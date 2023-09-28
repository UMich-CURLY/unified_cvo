#include <iostream>
#include <list>
#include <vector>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_set>
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
//#include "argparse/argparse.hpp"
#include "dataset_handler/KittiHandler.hpp"
#include "dataset_handler/EthzHandler.hpp"
#include "utils/PoseLoader.hpp"
#include "utils/ImageDownsampler.hpp"
#include "cvo/gpu_init.hpp"
#include "utils/ImageRGBD.hpp"
#include "utils/Calibration.hpp"
#include "utils/SymbolHash.hpp"
#include "utils/g2o_parser.hpp"
#include "utils/LidarPointDownsampler.hpp"

using namespace std;

extern template class cvo::VoxelMap<pcl::PointXYZRGB>;
extern template class cvo::Voxel<pcl::PointXYZRGB>;
extern template class cvo::Voxel<pcl::PointXYZI>;
extern template class cvo::VoxelMap<pcl::PointXYZI>;


void  log_lc_pc_pairs( //const cvo::aligned_vector<cvo::Mat34d_row> &BA_poses,
                       const std::map<int, cvo::Mat34d_row> & BA_poses,
                       const std::vector<std::pair<int, int>> &loop_closures,
                       const std::map<int, std::shared_ptr<cvo::CvoPointCloud>>& pcs,
                       std::string &fname_prefix) {
  
  for (int i = 0; i < loop_closures.size(); i++) {
    int id1 = loop_closures[i].first;
    int id2 = loop_closures[i].second;
    cvo::CvoPointCloud pc1_T, pc2_T;
    Eigen::Matrix4f pose1 = Eigen::Matrix4f::Identity();
    pose1.block(0,0,3,4) = BA_poses.at(id1).cast<float>();
    Eigen::Matrix4f pose2 = Eigen::Matrix4f::Identity();
    pose2.block(0,0,3,4) = BA_poses.at(id2).cast<float>();
    
    cvo::CvoPointCloud::transform(pose1, *pcs.at(id1), pc1_T);
    cvo::CvoPointCloud::transform(pose2, *pcs.at(id2), pc2_T);

    cvo::CvoPointCloud pc;
    pc = pc1_T + pc2_T;
    pc.write_to_intensity_pcd(fname_prefix + std::to_string(id1)+"_"+std::to_string(id2)+".pcd");
  }
}



void parse_lc_file(std::vector<std::pair<int, int>> & loop_closures,
                   const std::string & loop_closure_pairs_file,
                   int start_ind) {
  std::ifstream f(loop_closure_pairs_file);
  std::cout<<"Read loop closure file "<<loop_closure_pairs_file<<"\n";
  cvo::BinaryCommutativeMap<int> added_edges;
  if (f.is_open()) {
    std::string line;
    std::getline(f, line);
    while(std::getline(f, line)) {
      std::istringstream ss(line);
      int id1, id2;
      ss>>id1>>id2;
      id1 -= start_ind;
      id2 -= start_ind;
      if (added_edges.exists(std::min(id1, id2), std::max(id1, id2)) == false ) {
      std::cout<<"parse line "<<line<<"\n";
      std::cout<<"read lc between "<<std::min(id1, id2)<<" and "<<std::max(id1, id2)<<"\n";
      loop_closures.push_back(std::make_pair(std::min(id1, id2), std::max(id1, id2)));
      added_edges.insert(std::min(id1, id2), std::max(id1, id2), 1);
      }
    }


    f.close();
    //char a;
    //std::cin>>a;
  } else {
    std::cerr<<"loop closure file "<<loop_closure_pairs_file<<" doesn't exist\n";
    exit(0);
  }
}




void pose_graph_optimization( const cvo::aligned_vector<Eigen::Matrix4d> & tracking_poses,
                              //const Eigen::Matrix4d & T_last_to_first,
                              const std::vector<std::pair<int, int>> & loop_closures,
                              const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> & lc_poses,
                              const std::string & lc_constrains_file_before_BA,
                              std::map<int, cvo::Mat34d_row> & BA_poses,
                              double cov_scale_t,
			      double cov_scale_r,
                              int num_neighbors_per_node,
                              std::set<int> & selected_inds,
                              int num_merging_sequential_frames){
  
  Eigen::Matrix<double, 6,6> information = Eigen::Matrix<double, 6,6>::Identity(); 
  information.block(0,0,3,3) = Eigen::Matrix<double, 3,3>::Identity() / cov_scale_t / cov_scale_t;
  information.block(3,3,3,3) = Eigen::Matrix<double, 3,3>::Identity() / cov_scale_r / cov_scale_r;
  
  ceres::Problem problem;
  cvo::pgo::MapOfPoses poses;
  cvo::pgo::VectorOfConstraints constrains;
  std::ofstream lc_g2o(lc_constrains_file_before_BA);

  std::cout<<"copy from tracking_poses to poses and constrains\n";
  //for (int i = 0; i < tracking_poses.size(); i++) {
  for (auto i : selected_inds) {
  //  if (selected_inds.size() && selected_inds.find(i) == selected_inds.end() )
  //    continue;
    cvo::pgo::Pose3d pose = cvo::pgo::pose3d_from_eigen<double, Eigen::ColMajor>(tracking_poses[i]);
    poses.insert(std::make_pair(i, pose));
    lc_g2o <<cvo::pgo::Pose3d::name()<<" "<<i<<" "<<pose<<"\n";
  }

  std::cout<<"add between factors for odom\n";
  for (auto iter_i = selected_inds.begin(); iter_i != selected_inds.end(); iter_i++) {
	  int i = *iter_i;
	  std::cout<<"i="<<i<<"\n";
  //for  (int i = 0; i < tracking_poses.size(); i++) {
    auto iter_j = iter_i;
    iter_j++;
    if (iter_j == selected_inds.end())
      continue;
    //for (; std::distance(iter_i, iter_j) <= num_neighbors_per_node && iter_j != selected_inds.end() ; iter_j++) {
    int j = *iter_j;
    std::cout<<"j="<<j<<"\n";

    Eigen::Matrix4d T_Fi_to_Fj = tracking_poses[i].inverse() * tracking_poses[j];
    cvo::pgo::Pose3d t_be = cvo::pgo::pose3d_from_eigen<double, Eigen::ColMajor>(T_Fi_to_Fj);
    std::cout<<__func__<<": Add constrain from "<<i<<" to "<<j<<"\n";
    cvo::pgo::Constraint3d constrain{i, j, t_be, information};
    constrains.push_back(constrain);
    // }
  }
  
  
  /// the loop closing pose factors
  std::cout<<__func__<<"Loop closure number is "<<loop_closures.size()<<"\n"; 
  for (int i =0 ; i < loop_closures.size(); i++) {
    Eigen::Matrix4d T_f1_to_f2 = lc_poses[i].cast<double>();
    cvo::pgo::Pose3d t_be = cvo::pgo::pose3d_from_eigen<double, Eigen::ColMajor>(T_f1_to_f2);
    auto pair = loop_closures[i];
    cvo::pgo::Constraint3d constrain{static_cast<int>(pair.first), static_cast<int>(pair.second),
                                     t_be, information};
    constrains.push_back(constrain);
    lc_g2o<<cvo::pgo::Constraint3d::name()<<" "<<constrain<<"\n";
    std::cout<<__func__<<": Add constrain from "<<pair.first<<" to "<<pair.second<<"\n";
  }
  lc_g2o.close();

  /// optimization
  cvo::pgo::BuildOptimizationProblem(constrains, &poses, &problem);
  cvo::pgo::SolveOptimizationProblem(&problem);

  /// copy PGO results to BA_poses
  std::cout<<"Global PGO results:\n";
  //BA_poses.resize(tracking_poses.size());
  //BA_poses.resize(selected_inds.size());
  BA_poses.clear();
  for (auto pose_pair : poses) {
    int i = pose_pair.first;
    cvo::pgo::Pose3d pose = pose_pair.second;
    BA_poses.insert(std::make_pair(i, cvo::pgo::pose3d_to_eigen<double, Eigen::RowMajor>(pose).block(0,0,3,4)));
    std::cout<<BA_poses[i]<<"\n";
  }
}


// extern template class Foo<double>;
void global_registration_batch(cvo::CvoGPU & cvo_align,
                               const std::vector<std::pair<int, int>> & loop_closures,
                               const std::map<int, std::shared_ptr<cvo::CvoPointCloud>> & pcs,
                               const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> & gt_poses,
                               const std::string & registration_result_file,
                               std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> & lc_poses_f1_to_f2
                               ) {

  cvo::CvoParams & init_param = cvo_align.get_params();
  float ell_init = init_param.ell_init;
  float ell_decay_rate = init_param.ell_decay_rate;
  int ell_decay_start = init_param.ell_decay_start;
  init_param.ell_init = init_param.ell_init_first_frame;
  init_param.ell_decay_rate = init_param.ell_decay_rate_first_frame;
  init_param.ell_decay_start  = init_param.ell_decay_start_first_frame;
  cvo_align.write_params(&init_param);

  lc_poses_f1_to_f2.resize(loop_closures.size());
  std::ofstream f(registration_result_file);
  double time = 0;
  for (int i = 0; i < loop_closures.size(); i++) {
    Eigen::Matrix4f result;
    double time_curr;
    Eigen::Matrix4f init_guess_inv = Eigen::Matrix4f::Identity();

    auto p = loop_closures[i];
    cvo_align.align(*pcs.at(p.first), *pcs.at(p.second), init_guess_inv, result, nullptr, &time_curr);

    lc_poses_f1_to_f2[i] = result;
    
    std::cout<<"Finish running global registration between "<<p.first<<" and "<<p.second<<", result is\n"
             <<result<<"\n ground truth between "<<p.first<<" and "<<p.second<<" is \n"
             <<gt_poses[p.first].inverse() * gt_poses[p.second]<<"\n\n";

    f<<"====================================\nFinish running global registration between "<<p.first<<" and "<<p.second<<", result is\n"
     <<result<<"\n ground truth between "<<p.first<<" and "<<p.second<<" is \n"
     <<gt_poses[p.first].inverse() * gt_poses[p.second]<<"\n";
    auto gt_pose_curr = (gt_poses[p.first].inverse() * gt_poses[p.second]).cast<float>();
    Sophus::SE3f gt_sophus(gt_pose_curr.block(0,0,3,3), gt_pose_curr.block(0,3,3,1));
    Sophus::SE3f result_sophus(result.block(0,0,3,3), result.block(0,3,3,1));
    f<<"Error wrt gt is "<<(gt_sophus.inverse() * result_sophus).log().norm()<<"\n";
    

    time += time_curr;
  }
  init_param.ell_init = ell_init;
  init_param.ell_decay_rate = ell_decay_rate;
  init_param.ell_decay_start = ell_decay_start;
  cvo_align.write_params(&init_param);
  f.close();
  std::cout<<"Global Registration completes. Time is "<<time<<" seconds\n";
  return; //result.cast<double>();
}

void construct_loop_BA_problem(cvo::CvoGPU & cvo_align,
                               const std::vector<std::pair<int, int>> & loop_closures,
                               
                               std::map<int, cvo::CvoFrame::Ptr> & frames,
                               std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> &  gt_poses,
                               int num_neighbors_per_node,
                               int num_merging_sequential_frames                               
                               ) {
  // read edges to construct graph
  std::list<std::pair<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr>> edges;
  std::list<cvo::BinaryState::Ptr> edge_states;
  cvo::BinaryCommutativeMap<int> added_edges;
  std::vector<cvo::CvoFrame::Ptr> frames_vec;  
  //std::list<cvo::BinaryState::Ptr> edge_states_cpu;

  //for (int i = 0; i < frames.size(); i++) {
    // for (int j = i+1; j < std::min((int)frames.size(), i+1+num_neighbors_per_node); j++) {

  std::unordered_set<int> added_frames;
  for (auto iter_i = frames.begin(); iter_i != frames.end(); iter_i++) {
    auto  & [i, frame_i_ptr] = *iter_i;
    if (added_frames.find(i) == added_frames.end()) {
      added_frames.insert(i);
      frames_vec.push_back(frame_i_ptr);
    }
    
    auto iter_j = iter_i;
    iter_j++;
    for (; std::distance(iter_i, iter_j) <= num_neighbors_per_node && iter_j != frames.end() ; iter_j++) {
      auto  & [j, frame_j_ptr] = *iter_j;
      
      if (added_edges.exists(i, j) == false) {
        if (frames.find(j) == frames.end()) continue;
        if (added_frames.find(j) == added_frames.end()) {
          added_frames.insert(j);
          frames_vec.push_back(frame_j_ptr);
        }
      
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
        added_edges.insert(i, j, 1);

        std::cout<<"Added edge between ind "<<i<<" and  ind "<<j<<" with edge ptr "<<edge_state.get()<<std::endl;
      }
    }

  }

  /// loop closing constrains
  std::cout<<"BA: Adding loop closing constrains\n";
  const cvo::CvoParams & params = cvo_align.get_params();
  for (int i = 0; i < loop_closures.size(); i++) {
    std::pair<int, int> p = loop_closures[i];
    //for (int j = -1; j < 2; j++) {
    int id1 = p.first;//p.first + j;
    int id2 = p.second; //p.second + j;
    // if (id1 < 0 || id1 > frames.size()-1 || id2 < 0 || id2 > frames.size() -1 )
    //continue;
      if (added_edges.exists(id1, id2))
        continue;
      cvo::BinaryStateGPU::Ptr edge_state(new cvo::BinaryStateGPU(std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[id1]),
                                                                  std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[id2]),
                                                                  &params,
                                                                  cvo_align.get_params_gpu(),
                                                                  params.multiframe_num_neighbors,
                                                                  params.multiframe_ell_init 
                                                                  ));
      edge_states.push_back(edge_state);
      added_edges.insert(id1, id2, 1);
      std::cout<<"Added edge between ind "<<id1<<" and  ind "<<id2<<" with edge ptr "<<edge_state.get()<<std::endl;      
      //}
  }




  
  double time = 0;
  std::vector<bool> const_flags(frames.size(), false);
  const_flags[0] = true;
  std::cout<<"Total number of BA frames is "<<frames.size()<<"\n";
  
  auto start = std::chrono::system_clock::now();
  cvo::CvoBatchIRLS batch_irls_problem(frames_vec, const_flags,
                                       edge_states, &cvo_align.get_params());
  std::string err_file = std::string("err_wrt_iters.txt");
  batch_irls_problem.solve(gt_poses, err_file);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> t_all = end - start;
  // cvo_align.align(frames, const_flags,
  //               edge_states, &time);

  std::cout<<"GPU Align ends. Total time is "<<double(t_all.count()) / 1000<<" seconds."<<std::endl;
}

void write_traj_file(std::string & fname,
                     std::map<int, cvo::CvoFrame::Ptr> & frames ) {
  std::ofstream outfile(fname);
  for (auto & [i, ptr] : frames) {
  //for (int i = 0; i< frames.size(); i++) {
    //cvo::CvoFrame::Ptr ptr = frames[i];
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3,4>(0,0) = Eigen::Map<cvo::Mat34d_row>(ptr->pose_vec);
    Sophus::SO3d q(pose.block<3,3>(0,0));
    auto q_eigen = q.unit_quaternion().coeffs();
    Eigen::Vector3d t(pose.block<3,1>(0,3));
    outfile << t(0) <<" "<< t(1)<<" "<<t(2)<<" "
            << q_eigen[0]<<" "<<q_eigen[1]<<" "<<q_eigen[2]<<" "<<q_eigen[3]<<std::endl;
    
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

void write_transformed_pc(std::map<int, cvo::CvoFrame::Ptr> & frames,
                          std::string & fname,
                          int start_frame_ind=0, int end_frame_ind=1000000){
  pcl::PointCloud<pcl::PointXYZRGB> pc_all;
  pcl::PointCloud<pcl::PointXYZ> pc_xyz_all;
  for (auto & [i, ptr] : frames) {
  //for (int i = start_frame_ind; i <= std::min((int)frames.size(), end_frame_ind); i++) {
  //auto ptr = frames[i];

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

void sample_frame_inds(int start_ind, int end_ind, int num_merged_frames,
                       const std::vector<std::pair<int, int>> &loop_closures,
                       std::set<int> & result_selected_frames
                       ) {
  result_selected_frames.clear();
  for (int i = start_ind; i <= end_ind; i+=(1+num_merged_frames)){
    result_selected_frames.insert(i);
  }

  for (auto && p : loop_closures) {
    if(result_selected_frames.find(p.first) == result_selected_frames.end())
      result_selected_frames.insert(p.first);
    if(result_selected_frames.find(p.second) == result_selected_frames.end())
      result_selected_frames.insert(p.second);
  }
}

int main(int argc, char** argv) {

  //  omp_set_num_threads(24);
  //argparse::ArgumentParser parser("irls_kitti_loop_closure_test");
  /// assume start_ind and last_ind has overlap

  std::cout<<"Start \n";
  std::string data_type(argv[1]);
  std::string data_path(argv[2]);
  
  std::unique_ptr<cvo::DatasetHandler> dataset;
  cvo::KittiHandler::DataType dtype;
  if (std::strcmp(data_type.c_str(), "kitti_stereo") == 0) {
    dtype = cvo::KittiHandler::DataType::STEREO;
    dataset.reset(new cvo::KittiHandler(data_path, dtype));    
  } else if (std::strcmp(data_type.c_str(), "kitti_lidar") == 0) {
    dtype = cvo::KittiHandler::DataType::LIDAR;
    dataset.reset(new cvo::KittiHandler(data_path, dtype));    
  } else if (std::strcmp(data_type.c_str(), "ethz" ) == 0) {
    dataset.reset(new cvo::EthzHandler(data_path, cvo::EthzHandler::FrameType::LOCAL));
  } else {
    ASSERT(false, "unknown dtype");
  }

  
  //cvo::KittiHandler kitti(argv[1], cvo::KittiHandler::DataType::LIDAR);
  string cvo_param_file(argv[3]);    
  int num_neighbors_per_node = std::stoi(argv[4]); // forward neighbors
  std::string tracking_traj_file(argv[5]);
  std::string loop_closure_input_file(argv[6]);
  std::string BA_traj_file(argv[7]);
  int is_edge_only = std::stoi(argv[8]);
  std::cout<<"is edge only is "<<is_edge_only<<"\n";
  int start_ind = std::stoi(argv[9]);
  std::cout<<"input start_ind is  "<<start_ind<<"\n";
  int max_last_ind = std::stoi(argv[10]);
  std::cout<<"input last_ind is  "<<max_last_ind<<"\n";
  double cov_scale_t = std::stod(argv[11]);
  double cov_scale_r = std::stod(argv[12]);
  int num_merging_sequential_frames = std::stoi(argv[13]);
  int is_pgo_only = std::stoi(argv[14]);
  int is_read_loop_closure_poses_from_file = std::stoi(argv[15]);
  
  int last_ind = std::min(max_last_ind+1, dataset->get_total_number()-1);
  std::cout<<"actual last ind is "<<last_ind<<"\n";

  std::cout<<"Finish reading all arguments\n";
  int total_iters = last_ind - start_ind + 1;

  //cvo::gpu_init(10);
  //std::cout<<"Launched gpu_init\n";
  std::string calib_file;
  calib_file =  data_path +"/cvo_calib.txt"; 
  cvo::Calibration calib(calib_file, cvo::Calibration::STEREO);
  
  cvo::CvoGPU cvo_align(cvo_param_file);
  string gt_pose_name;
  gt_pose_name = std::string(data_path) + "/poses.txt";

  /// read poses
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> gt_poses_raw(total_iters),
    gt_poses(total_iters),
    gt_pose_selected_vec,
    tracking_poses(total_iters);
  // std::vector<cvo::Mat34d_row,
  //            Eigen::aligned_allocator<cvo::Mat34d_row>> BA_poses(total_iters, cvo::Mat34d_row::Zero());
  std::map<int, cvo::Mat34d_row> BA_poses;


  /// read gt poses
  cvo::read_pose_file_kitti_format(gt_pose_name,
                                   start_ind,
                                   last_ind,
                                   gt_poses_raw);
  std::cout<<"gt poses size is "<<gt_poses_raw.size()<<"\n";
  Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();
  cvo::transform_vector_of_poses(gt_poses_raw, identity, gt_poses );

  /// read tracking files
  cvo::read_pose_file_kitti_format(tracking_traj_file,
                                   start_ind,
                                   last_ind,
                                   tracking_poses);
  std::cout<<"init tracking pose size is "<<tracking_poses.size()<<"\n";
  assert(gt_poses.size() == tracking_poses.size());

  // read loop closure files
  std::vector<std::pair<int, int>> loop_closures;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> lc_poses;
  if (is_read_loop_closure_poses_from_file) {
    cvo::ReadG2oFile(loop_closure_input_file, loop_closures, lc_poses) ;
  } else {
    parse_lc_file(loop_closures, loop_closure_input_file, start_ind);    
  }
  
  /// decide if we will skip frames
  std::set<int> result_selected_frames;
  sample_frame_inds(start_ind, last_ind, num_merging_sequential_frames, loop_closures, result_selected_frames);  

  /// select gt poses
  std::map<int, Eigen::Matrix4d> gt_pose_selected;
  for (auto ind : result_selected_frames)
    gt_pose_selected_vec.push_back(gt_poses[ind]);//insert(std::make_pair(ind, gt_poses[ind]));
  std::string gt_fname("groundtruth.txt");
  cvo::write_traj_file<double, 4, Eigen::ColMajor>(gt_fname,gt_poses, result_selected_frames);
  std::string track_fname("tracking.txt");
  cvo::write_traj_file<double, 4, Eigen::ColMajor>(track_fname, tracking_poses, result_selected_frames);
  
  
  
  // read point cloud
  std::map<int, cvo::CvoFrame::Ptr> frames;
  std::map<int, std::shared_ptr<cvo::CvoPointCloud>> pcs;
  if (!(is_pgo_only && is_read_loop_closure_poses_from_file)) {
    if (std::strcmp(data_type.c_str(), "kitti_lidar") == 0) {
      cvo::read_and_downsample_lidar_pc(result_selected_frames,
                                        *dataset,
                                        tracking_poses,                                        
                                        num_merging_sequential_frames,
                                        cvo_align.get_params().multiframe_downsample_voxel_size,
                                        is_edge_only,
                                        pcs);
    } else if (std::strcmp(data_type.c_str(), "kitti_stereo") == 0) {
      cvo::read_and_downsample_sequentail_stereo_frames(result_selected_frames, *dataset, calib,
                                                        tracking_poses,
                                                        num_merging_sequential_frames,
                                                        cvo_align.get_params().multiframe_downsample_voxel_size,
                                                        is_edge_only,
                                                        // results
                                                        pcs);

    } else if (std::strcmp(data_type.c_str(), "ethz") == 0) {
      cvo::read_and_downsample_sequentail_stereo_frames(result_selected_frames, *dataset, calib,
                                                        tracking_poses,
                                                        num_merging_sequential_frames,
                                                        cvo_align.get_params().multiframe_downsample_voxel_size,
                                                        true,
                                                        // results
                                                        pcs);
      
    }

  }
  

  /// global registration
  if (is_read_loop_closure_poses_from_file == false) {
    std::string g_reg_f("global.txt");    
    global_registration_batch(cvo_align,
                              loop_closures,
                              pcs,
                              gt_poses,
                              g_reg_f,
                              lc_poses);
  }

  /// pose graph optimization
  std::string lc_g2o("loop_closures.g2o");
  std::cout<<"Start PGO...\n";
  pose_graph_optimization(tracking_poses, loop_closures,
                          lc_poses, lc_g2o,
                          BA_poses, 
			  cov_scale_t, cov_scale_r, num_neighbors_per_node,
                          result_selected_frames,
                          num_merging_sequential_frames);
  std::cout<<"Finish PGO...\n";  
  std::string pgo_fname("pgo.txt");
  std::vector<cvo::Mat34d_row, Eigen::aligned_allocator<cvo::Mat34d_row>> pgo_poses;
  for (auto i : result_selected_frames) pgo_poses.push_back(BA_poses[i]);
  cvo::write_traj_file<double, 3, Eigen::RowMajor>(pgo_fname, pgo_poses);
  std::string lc_prefix(("loop_closure_"));
  if (pcs.size())
    log_lc_pc_pairs(BA_poses, loop_closures, pcs, lc_prefix);
  if (is_pgo_only) return 0;
  
  
  std::cout<<"Start construct BA CvoFrame\n";
  /// construct BA CvoFrame struct
  for (auto i : result_selected_frames) {
  //for (int i = 0; i<result_selected_frames.size(); i++) {
    std::cout<<"Copy "<<i<<"th point cloud to gpu \n";
    double * poses_data = BA_poses[i].data();
    cvo::CvoFrame::Ptr new_frame(new cvo::CvoFrameGPU(pcs[i].get(), poses_data, cvo_align.get_params().is_using_kdtree));
    frames.insert(std::make_pair(i, new_frame));
  }
  

  std::string f_name = std::string("before_BA_loop.pcd");
  write_transformed_pc(frames, f_name, 0, frames.size()-1);
  
  /// Multiframe alignment
  std::cout<<"Construct loop BA problem\n";
  ASSERT(frames.size() == gt_poses.size(), "frame size must be equal to gt_poses size");
  construct_loop_BA_problem(cvo_align,
                            loop_closures,
                            frames, gt_pose_selected_vec, num_neighbors_per_node,
                            num_merging_sequential_frames);

  std::cout<<"Write stacked point cloud\n";
  f_name = std::string("after_BA_loop.pcd") ;
  write_transformed_pc(frames, f_name,0, frames.size()-1);
  std::cout<<"Write traj to file\n";
  write_traj_file(BA_traj_file,frames);
  return 0;
}
