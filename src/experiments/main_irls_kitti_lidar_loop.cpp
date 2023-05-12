#include <iostream>
#include <list>
#include <vector>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <set>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "cvo/CvoGPU.hpp"
#include "cvo/IRLS_State_CPU.hpp"
#include "cvo/IRLS_State_GPU.hpp"
#include "cvo/IRLS_State.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoFrame.hpp"
#include "cvo/CvoFrameGPU.hpp"
#include "cvo/IRLS.hpp"
#include "utils/VoxelMap.hpp"
#include "utils/data_type.hpp"
#include "graph_optimizer/PoseGraphOptimization.hpp"
#include "dataset_handler/KittiHandler.hpp"
#include "dataset_handler/PoseLoader.hpp"
#include "utils/LidarPointSelector.hpp"
#include "utils/LidarPointType.hpp"
#include "utils/ImageRGBD.hpp"
#include "utils/Calibration.hpp"

using namespace std;

extern template class cvo::VoxelMap<pcl::PointXYZRGB>;
extern template class cvo::Voxel<pcl::PointXYZRGB>;
extern template class cvo::Voxel<pcl::PointXYZI>;
extern template class cvo::VoxelMap<pcl::PointXYZI>;



void pose_graph_optimization( const cvo::aligned_vector<Eigen::Matrix4d> & tracking_poses,
                              const Eigen::Matrix4d & T_last_to_first,
                              cvo::aligned_vector<cvo::Mat34d_row> & BA_poses,
                              double cov_scale,
                              int num_neighbors_per_node){
  Eigen::Matrix<double, 6,6> information = Eigen::Matrix<double, 6,6>::Identity() / cov_scale / cov_scale;
  
  ceres::Problem problem;
  cvo::pgo::MapOfPoses poses;
  cvo::pgo::VectorOfConstraints constrains;
  /// copy from tracking_poses to poses and constrains
  for (int i = 0; i < tracking_poses.size(); i++) {
    cvo::pgo::Pose3d pose = cvo::pgo::pose3d_from_eigen<double, Eigen::ColMajor>(tracking_poses[i]);
    poses.insert(std::make_pair(i, pose));
  }
  for (int i = 0; i < tracking_poses.size(); i++) {
    for (int j = i+1; j < std::min((int)tracking_poses.size(), i+1+num_neighbors_per_node); j++) {
      Eigen::Matrix4d T_Fi_to_Fj = tracking_poses[i].inverse() * tracking_poses[j];
      cvo::pgo::Pose3d t_be = cvo::pgo::pose3d_from_eigen<double, Eigen::ColMajor>(T_Fi_to_Fj);
      std::cout<<__func__<<": Add constrain between "<<i<<" and "<<j<<"\n";
      cvo::pgo::Constraint3d constrain{i, j, t_be, information};
      constrains.push_back(constrain);
    }
  }
  /// the loop closing one
  Eigen::Matrix4d T_Fe_to_F0 = T_last_to_first;
  cvo::pgo::Pose3d t_be = cvo::pgo::pose3d_from_eigen<double, Eigen::ColMajor>(T_Fe_to_F0);
  cvo::pgo::Constraint3d constrain{static_cast<int>(tracking_poses.size()-1), 0, t_be, information};
  constrains.push_back(constrain);

  /// optimization
  cvo::pgo::BuildOptimizationProblem(constrains, &poses, &problem);
  cvo::pgo::SolveOptimizationProblem(&problem);

  /// copy PGO results to BA_poses
  std::cout<<"Global PGO results:\n";
  BA_poses.resize(tracking_poses.size());
  for (auto pose_pair : poses) {
    int i = pose_pair.first;
    cvo::pgo::Pose3d pose = pose_pair.second;
    BA_poses[i] = cvo::pgo::pose3d_to_eigen<double, Eigen::RowMajor>(pose).block(0,0,3,4);
    std::cout<<BA_poses[i]<<"\n";
  }
}


// extern template class Foo<double>;
Eigen::Matrix4d global_registration_last_frame_to_first_frame(cvo::CvoGPU & cvo_align,
                                                              const cvo::CvoPointCloud & p0,
                                                              const Eigen::Matrix4d & T0,
                                                              const cvo::CvoPointCloud & p_last,
                                                              const Eigen::Matrix4d & T_last) {
  Eigen::Matrix4f result;
  double time;
  Eigen::Matrix4f init_guess_inv = Eigen::Matrix4f::Identity();

  cvo::CvoParams & init_param = cvo_align.get_params();
  float ell_init = init_param.ell_init;
  float ell_decay_rate = init_param.ell_decay_rate;
  int ell_decay_start = init_param.ell_decay_start;
  init_param.ell_init = init_param.ell_init_first_frame;
  init_param.ell_decay_rate = init_param.ell_decay_rate_first_frame;
  init_param.ell_decay_start  = init_param.ell_decay_start_first_frame;
  cvo_align.write_params(&init_param);
  
  cvo_align.align(p_last, p0, init_guess_inv, result, nullptr, &time);
  
  init_param.ell_init = ell_init;
  init_param.ell_decay_rate = ell_decay_rate;
  init_param.ell_decay_start = ell_decay_start;
  cvo_align.write_params(&init_param);
  

  std::cout<<"Global Registration completes. Time is "<<time<<" seconds\n";
  return result.cast<double>();
}

void construct_loop_BA_problem(cvo::CvoGPU & cvo_align,
                               std::vector<cvo::CvoFrame::Ptr> frames,
                               std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> &  gt_poses,
                               int num_neighbors_per_node
                               ) {
  // read edges to construct graph
  std::list<std::pair<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr>> edges;
  std::list<cvo::BinaryState::Ptr> edge_states;
  //std::list<cvo::BinaryState::Ptr> edge_states_cpu;
  for (int i = 0; i < frames.size(); i++) {
    for (int j = i+1; j < std::min((int)frames.size(), i+1+num_neighbors_per_node); j++) {

      std::cout<<"first ind "<<i<<", second ind "<<j<<std::endl;
      std::pair<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr> p(frames[i], frames[j]);
      edges.push_back(p);
    
      const cvo::CvoParams & params = cvo_align.get_params();
      cvo::BinaryStateGPU::Ptr edge_state(new cvo::BinaryStateGPU(std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[i]),
                                                                  std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[j]),
                                                                  &params,
                                                                  cvo_align.get_params_gpu(),
                                                                  params.multiframe_num_neighbors,
                                                                  params.multiframe_ell_init
                                                                  ));
      edge_states.push_back((edge_state));
    }

  }

  /// loop closing constrains
  const cvo::CvoParams & params = cvo_align.get_params();
  cvo::BinaryStateGPU::Ptr edge_state_0_n(new cvo::BinaryStateGPU(std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[0]),
                                                              std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames.back()),
                                                              &params,
                                                              cvo_align.get_params_gpu(),
                                                              params.multiframe_num_neighbors,
                                                              params.multiframe_ell_init
                                                              ));
  edge_states.push_back((edge_state_0_n));
  cvo::BinaryStateGPU::Ptr edge_state_1_n(new cvo::BinaryStateGPU(std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[1]),
                                                              std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames.back()),
                                                              &params,
                                                              cvo_align.get_params_gpu(),
                                                              params.multiframe_num_neighbors,
                                                              params.multiframe_ell_init
                                                              ));
  edge_states.push_back((edge_state_1_n));
  cvo::BinaryStateGPU::Ptr edge_state_0_n1(new cvo::BinaryStateGPU(std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[0]),
                                                              std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[frames.size()-2]),
                                                              &params,
                                                              cvo_align.get_params_gpu(),
                                                              params.multiframe_num_neighbors,
                                                              params.multiframe_ell_init
                                                              ));
  edge_states.push_back((edge_state_0_n1));
  
  
  
  double time = 0;
  std::vector<bool> const_flags(frames.size(), false);
  const_flags[0] = true;
  std::cout<<"Total number of BA frames is "<<frames.size()<<"\n";
  
  auto start = std::chrono::system_clock::now();
  cvo::CvoBatchIRLS batch_irls_problem(frames, const_flags,
                                       edge_states, &cvo_align.get_params());
  std::string err_file = std::string("err_wrt_iters.txt");
  batch_irls_problem.solve(gt_poses, err_file);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> t_all = end - start;
  // cvo_align.align(frames, const_flags,
  //               edge_states, &time);

  std::cout<<"GPU Align ends. Total time is "<<double(t_all.count()) / 1000<<" seconds."<<std::endl;
}

std::shared_ptr<cvo::CvoPointCloud> downsample_lidar_points(bool is_edge_only,
                                                            pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in,
                                                            float leaf_size) {

  /*
  int expected_points = 5000;
  double intensity_bound = 0.4;
  double depth_bound = 4.0;
  double distance_bound = 40.0;
  
  LidarPointSelector lps(expected_points, intensity_bound, depth_bound, distance_bound, beam_num);

  // running edge detection + lego loam point selection
  pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_edge (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_surface (new pcl::PointCloud<pcl::PointXYZI>);
  std::vector<int> selected_edge_inds, selected_loam_inds;
  lps.edge_detection(pc, pc_out_edge, output_depth_grad, output_intenstity_grad, selected_edge_inds);
  
  lps.legoloam_point_selector(pc, pc_out_surface, edge_or_surface, selected_loam_inds);    
  //*pc_out += *pc_out_edge;
  //*pc_out += *pc_out_surface;
  //
  num_points_ = selected_indexes.size();
  */

  cvo::VoxelMap<pcl::PointXYZI> full_voxel(leaf_size);
  for (int k = 0; k < pc_in->size(); k++) {
    full_voxel.insert_point(&pc_in->points[k]);
  }
  std::vector<pcl::PointXYZI*> downsampled_results = full_voxel.sample_points();
  pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZI>);
  for (int k = 0; k < downsampled_results.size(); k++)
    downsampled->push_back(*downsampled_results[k]);
  std::shared_ptr<cvo::CvoPointCloud> ret(new cvo::CvoPointCloud(downsampled, 5000, 64, cvo::CvoPointCloud::PointSelectionMethod::FULL));
  return ret;
}


void write_traj_file(std::string & fname,
                     std::vector<cvo::CvoFrame::Ptr> & frames ) {
  std::ofstream outfile(fname);
  for (int i = 0; i< frames.size(); i++) {
    cvo::CvoFrame::Ptr ptr = frames[i];
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3,4>(0,0) = Eigen::Map<cvo::Mat34d_row>(ptr->pose_vec);
    Sophus::SO3d q(pose.block<3,3>(0,0));
    auto q_eigen = q.unit_quaternion().coeffs();
    Eigen::Vector3d t(pose.block<3,1>(0,3));
    outfile << t(0) <<" "<< t(1)<<" "<<t(2)<<" "
            <<q_eigen[0]<<" "<<q_eigen[1]<<" "<<q_eigen[2]<<" "<<q_eigen[3]<<std::endl;
    
  }
  outfile.close();
}

void write_traj_file(std::string & fname,
                     std::vector<std::string> & timestamps,
                     std::vector<cvo::CvoFrame::Ptr> & frames ) {
  std::ofstream outfile(fname);
  for (int i = 0; i< frames.size(); i++) {
    cvo::CvoFrame::Ptr ptr = frames[i];
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3,4>(0,0) = Eigen::Map<cvo::Mat34d_row>(ptr->pose_vec);
    Sophus::SO3d q(pose.block<3,3>(0,0));
    auto q_eigen = q.unit_quaternion().coeffs();
    Eigen::Vector3d t(pose.block<3,1>(0,3));
    outfile <<timestamps[i]<<" "<< t(0) <<" "<< t(1)<<" "<<t(2)<<" "
            <<q_eigen[0]<<" "<<q_eigen[1]<<" "<<q_eigen[2]<<" "<<q_eigen[3]<<std::endl;
    
  }
  outfile.close();
}

void write_transformed_pc(std::vector<cvo::CvoFrame::Ptr> & frames, std::string & fname,
                          int start_frame_ind=0, int end_frame_ind=1000000){
  pcl::PointCloud<pcl::PointXYZRGB> pc_all;
  pcl::PointCloud<pcl::PointXYZ> pc_xyz_all;

  for (int i = start_frame_ind; i <= std::min((int)frames.size(), end_frame_ind); i++) {
    auto ptr = frames[i];

    cvo::CvoPointCloud new_pc;
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3,4>(0,0) = Eigen::Map<cvo::Mat34d_row>(ptr->pose_vec);
    
    Eigen::Matrix4f pose_f = pose.cast<float>();
    cvo::CvoPointCloud::transform(pose_f, *ptr->points, new_pc);

    pcl::PointCloud<pcl::PointXYZRGB> pc_curr;
    pcl::PointCloud<pcl::PointXYZ> pc_xyz_curr;
    new_pc.export_to_pcd(pc_curr);
    new_pc.export_to_pcd(pc_xyz_curr);

    pc_all += pc_curr;
    pc_xyz_all += pc_xyz_curr;

  }
  //pcl::io::savePCDFileASCII(fname, pc_all);
  pcl::io::savePCDFileASCII(fname, pc_xyz_all);
}

int main(int argc, char** argv) {

  //  omp_set_num_threads(24);

  /// assume start_ind and last_ind has overlap
  
  cvo::KittiHandler kitti(argv[1], cvo::KittiHandler::DataType::LIDAR);
  string cvo_param_file(argv[2]);    
  int num_neighbors_per_node = std::stoi(argv[3]); // forward neighbors
  std::string tracking_traj_file(argv[4]);
  std::string BA_traj_file(argv[5]);
  int is_edge_only = std::stoi(argv[6]);
  std::cout<<"is edge only is "<<is_edge_only<<"\n";
  int start_ind = std::stoi(argv[7]);
  std::cout<<"input start_ind is  "<<start_ind<<"\n";
  int max_last_ind = std::stoi(argv[8]);
  std::cout<<"input last_ind is  "<<max_last_ind<<"\n";
  double cov_scale = std::stod(argv[9]);
  int skipped_frames = std::stoi(argv[10]);

  
  int last_ind = std::min(max_last_ind+1, kitti.get_total_number());
  std::cout<<"actual last ind is "<<last_ind<<"\n";

  std::cout<<"Finish reading all arguments\n";
  int total_iters = last_ind - start_ind + 1;

  cvo::CvoGPU cvo_align(cvo_param_file);
  string gt_pose_name;
  gt_pose_name = std::string(argv[1]) + "/poses.txt";

  /// read poses
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> gt_poses_raw(total_iters),
    gt_poses(total_iters),
    tracking_poses(total_iters);
  std::vector<cvo::Mat34d_row,
              Eigen::aligned_allocator<cvo::Mat34d_row>> BA_poses(total_iters, cvo::Mat34d_row::Zero());

  cvo::read_pose_file_kitti_format(gt_pose_name,
                                   start_ind,
                                   last_ind,
                                   gt_poses_raw);
  std::cout<<"gt poses size is "<<gt_poses_raw.size()<<"\n";
  Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();
  cvo::transform_vector_of_poses(gt_poses_raw, identity, gt_poses );
  
  cvo::read_pose_file_kitti_format(tracking_traj_file,
                                   start_ind,
                                   last_ind,
                                   tracking_poses);
  std::cout<<"init tracking pose size is "<<tracking_poses.size()<<"\n";
  assert(gt_poses.size() == tracking_poses.size());
  //Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();
 
  
  
  //for (int i = 0; i < tracking_poses.size(); i++)  {
  //  BA_poses[i].block(0,0,3,4) = tracking_poses[i].block(0,0,3,4); 
  //}
  
  
  // read point cloud
  std::vector<cvo::CvoFrame::Ptr> frames;
  std::vector<std::shared_ptr<cvo::CvoPointCloud>> pcs;
  for (int i = 0; i<gt_poses.size(); i++) {

    std::cout<<"new frame "<<i+start_ind<<" out of "<<gt_poses.size() + start_ind<<"\n";
    
    kitti.set_start_index(i+start_ind);
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_pcl(new pcl::PointCloud<pcl::PointXYZI>);
    if (-1 == kitti.read_next_lidar(pc_pcl)) 
      break;
    
    float leaf_size = cvo_align.get_params().multiframe_downsample_voxel_size;
    std::shared_ptr<cvo::CvoPointCloud> pc = downsample_lidar_points(is_edge_only,
                                                                     pc_pcl,
                                                                     leaf_size);
    pcs.push_back(pc);
  }


  /// global registration
  Eigen::Matrix4d T_last_to_first = global_registration_last_frame_to_first_frame(cvo_align,
                                                                                  *pcs[0], tracking_poses[0],
                                                                                  *pcs.back(), tracking_poses.back());
  std::cout<<"global registration result is \n"<<T_last_to_first<<"\n";

  /// pose graph optimization
  pose_graph_optimization(tracking_poses, T_last_to_first, BA_poses, cov_scale, num_neighbors_per_node);
  std::string pgo_fname("pgo.txt");
  cvo::write_traj_file<double, 3, Eigen::RowMajor>(pgo_fname, BA_poses);


  /// construct BA CvoFrame struct
  for (int i = 0; i<gt_poses.size(); i++) {
    std::cout<<"Copy "<<i<<"th point cloud to gpu \n";
    double * poses_data = BA_poses[i].data();
    cvo::CvoFrame::Ptr new_frame(new cvo::CvoFrameGPU(pcs[i].get(), poses_data, cvo_align.get_params().is_using_kdtree));
    frames.push_back(new_frame);
  }
  

  std::string f_name = std::string("before_BA_loop.pcd");
  write_transformed_pc(frames, f_name, 0, frames.size()-1);
  
  /// Multiframe alignment
  construct_loop_BA_problem(cvo_align, frames, gt_poses, num_neighbors_per_node);

  std::cout<<"Write stacked point cloud\n";
  f_name = std::string("after_BA_loop.pcd") ;
  write_transformed_pc(frames, f_name,0, frames.size()-1);
  std::cout<<"Write traj to file\n";
  write_traj_file(BA_traj_file,frames);
  std::string gt_fname("groundtruth.txt");
  cvo::write_traj_file<double, 4, Eigen::ColMajor>(gt_fname,gt_poses);
  std::string track_fname("tracking.txt");
  cvo::write_traj_file<double, 4, Eigen::ColMajor>(track_fname, tracking_poses);
  return 0;
}
