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
#include "dataset_handler/PoseLoader.hpp"
#include "utils/LidarPointSelector.hpp"
#include "utils/LidarPointType.hpp"
#include "utils/ImageRGBD.hpp"
#include "utils/Calibration.hpp"
#include "utils/SymbolHash.hpp"

using namespace std;

extern template class cvo::VoxelMap<pcl::PointXYZRGB>;
extern template class cvo::Voxel<pcl::PointXYZRGB>;
extern template class cvo::Voxel<pcl::PointXYZI>;
extern template class cvo::VoxelMap<pcl::PointXYZI>;


void  log_lc_pc_pairs( const cvo::aligned_vector<cvo::Mat34d_row> &BA_poses,
                       const std::vector<std::pair<int, int>> &loop_closures,
                       const std::vector<std::shared_ptr<cvo::CvoPointCloud>>& pcs,
                       std::string &fname_prefix) {
  
  for (int i = 0; i < loop_closures.size(); i++) {
    int id1 = loop_closures[i].first;
    int id2 = loop_closures[i].second;
    cvo::CvoPointCloud pc1_T, pc2_T;
    Eigen::Matrix4f pose1 = Eigen::Matrix4f::Identity();
    pose1.block(0,0,3,4) = BA_poses[id1].cast<float>();
    Eigen::Matrix4f pose2 = Eigen::Matrix4f::Identity();
    pose2.block(0,0,3,4) = BA_poses[id2].cast<float>();
    
    cvo::CvoPointCloud::transform(pose1, *pcs[id1], pc1_T);
    cvo::CvoPointCloud::transform(pose2, *pcs[id2], pc2_T);

    cvo::CvoPointCloud pc;
    pc = pc1_T + pc2_T;
    pc.write_to_intensity_pcd(fname_prefix + std::to_string(id1)+"_"+std::to_string(id2)+".pcd");
  }
}


// Reads a single pose from the input and inserts it into the map. Returns false
// if there is a duplicate entry.
bool ReadVertex(std::ifstream* infile,
                cvo::pgo::MapOfPoses* poses) {
  int id;
  cvo::pgo::Pose3d pose;
  *infile >> id >> pose;
  // Ensure we don't have duplicate poses.
  if (poses->find(id) != poses->end()) {
    std::cerr << "Duplicate vertex with ID: " << id;
    return false;
  }
  std::cout<<"Read pose "<<id<<"\n";
  (*poses)[id] = pose;
  return true;
}
// Reads the contraints between two vertices in the pose graph
void ReadConstraint(std::ifstream* infile,
                    cvo::pgo::VectorOfConstraints* constraints) {
  cvo::pgo::Constraint3d constraint;
  *infile >> constraint;
  std::cout<<"Read constrain: "<<constraint<<"\n";
  constraints->push_back(constraint);
}
// Reads a file in the g2o filename format that describes a pose graph
// problem. The g2o format consists of two entries, vertices and constraints.
//
// In 2D, a vertex is defined as follows:
//
// VERTEX_SE2 ID x_meters y_meters yaw_radians
//
// A constraint is defined as follows:
//
// EDGE_SE2 ID_A ID_B A_x_B A_y_B A_yaw_B I_11 I_12 I_13 I_22 I_23 I_33
//
// where I_ij is the (i, j)-th entry of the information matrix for the
// measurement.
//
//
// In 3D, a vertex is defined as follows:
//
// VERTEX_SE3:QUAT ID x y z q_x q_y q_z q_w
//
// where the quaternion is in Hamilton form.
// A constraint is defined as follows:
//
// EDGE_SE3:QUAT ID_a ID_b x_ab y_ab z_ab q_x_ab q_y_ab q_z_ab q_w_ab I_11 I_12 I_13 ... I_16 I_22 I_23 ... I_26 ... I_66 // NOLINT
//
// where I_ij is the (i, j)-th entry of the information matrix for the
// measurement. Only the upper-triangular part is stored. The measurement order
// is the delta position followed by the delta orientation.
bool ReadG2oFile(const std::string& filename,
                 std::vector<std::pair<int, int>> & loop_closures,
                 std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>&  lc_poses
                 ){
  
  cvo::pgo::MapOfPoses  poses;
  cvo::pgo::VectorOfConstraints constraints;

  std::ifstream infile(filename.c_str());
  if (!infile) {
    return false;
  }
  std::string data_type;
  while (infile.good()) {
    // Read whether the type is a node or a constraint.
    infile >> data_type;
    if (data_type == cvo::pgo::Pose3d::name()) {
      if (!ReadVertex(&infile, &poses)) {
        return false;
      }
    } else if (data_type == cvo::pgo::Constraint3d::name()) {
      ReadConstraint(&infile, &constraints);
    } else {
      std::cerr << "Unknown data type: " << data_type;
      return false;
    }
    // Clear any trailing whitespace from the line.
    infile >> std::ws;
  }

  
  std::cout<<__func__<<"Read from file, # of lc constrains is  "<<constraints.size()<<"\n";
  for (auto && constrain: constraints ) {
    int id1 = constrain.id_begin;
    int id2 = constrain.id_end;
    loop_closures.push_back(std::make_pair(id1, id2));
    Eigen::Matrix4f pose = cvo::pgo::pose3d_to_eigen<float, Eigen::ColMajor>(constrain.t_be);
    lc_poses.push_back(pose);
    std::cout<<__func__<<"Read from file, loop closure constrain between "<<id1<<" and "<<id2<<" is \n"<<pose<<"\n";
  }  
  return true;
}



void parse_lc_file(std::vector<std::pair<int, int>> & loop_closures,
                   const std::string & loop_closure_pairs_file,
                   int start_ind) {
  std::ifstream f(loop_closure_pairs_file);
  std::cout<<"Read loop closure file "<<loop_closure_pairs_file<<"\n";
  if (f.is_open()) {
    std::string line;
    std::getline(f, line);
    while(std::getline(f, line)) {
      std::istringstream ss(line);
      int id1, id2;
      ss>>id1>>id2;
      std::cout<<"read lc between "<<id1<<" and "<<id2<<"\n";
      id1 -= start_ind;
      id2 -= start_ind;
      loop_closures.push_back(std::make_pair(std::min(id1, id2), std::max(id1, id2)));
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
                              cvo::aligned_vector<cvo::Mat34d_row> & BA_poses,
                              double cov_scale_t,
			      double cov_scale_r,
                              int num_neighbors_per_node){
  Eigen::Matrix<double, 6,6> information = Eigen::Matrix<double, 6,6>::Identity(); 
  information.block(0,0,3,3) = Eigen::Matrix<double, 3,3>::Identity() / cov_scale_t / cov_scale_t;
  information.block(3,3,3,3) = Eigen::Matrix<double, 3,3>::Identity() / cov_scale_r / cov_scale_r;
  
  ceres::Problem problem;
  cvo::pgo::MapOfPoses poses;
  cvo::pgo::VectorOfConstraints constrains;
  std::ofstream lc_g2o(lc_constrains_file_before_BA);

  /// copy from tracking_poses to poses and constrains
  for (int i = 0; i < tracking_poses.size(); i++) {
    cvo::pgo::Pose3d pose = cvo::pgo::pose3d_from_eigen<double, Eigen::ColMajor>(tracking_poses[i]);
    poses.insert(std::make_pair(i, pose));
    lc_g2o <<cvo::pgo::Pose3d::name()<<" "<<i<<" "<<pose<<"\n";
  }
  
  for (int i = 0; i < tracking_poses.size(); i++) {
    //for (int j = i+1; j < std::min((int)tracking_poses.size(), i+1+num_neighbors_per_node); j++) {
    for (int j = i+1; j < std::min((int)tracking_poses.size(), i+2); j++) {
      Eigen::Matrix4d T_Fi_to_Fj = tracking_poses[i].inverse() * tracking_poses[j];
      cvo::pgo::Pose3d t_be = cvo::pgo::pose3d_from_eigen<double, Eigen::ColMajor>(T_Fi_to_Fj);
      std::cout<<__func__<<": Add constrain from "<<i<<" to "<<j<<"\n";
      cvo::pgo::Constraint3d constrain{i, j, t_be, information};
      constrains.push_back(constrain);
    }
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
  BA_poses.resize(tracking_poses.size());
  for (auto pose_pair : poses) {
    int i = pose_pair.first;
    cvo::pgo::Pose3d pose = pose_pair.second;
    BA_poses[i] = cvo::pgo::pose3d_to_eigen<double, Eigen::RowMajor>(pose).block(0,0,3,4);
    std::cout<<BA_poses[i]<<"\n";
  }
}


// extern template class Foo<double>;
void global_registration_batch(cvo::CvoGPU & cvo_align,
                               const std::vector<std::pair<int, int>> & loop_closures,
                               const std::vector<std::shared_ptr<cvo::CvoPointCloud>> & pcs,
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
    cvo_align.align(*pcs[p.first], *pcs[p.second], init_guess_inv, result, nullptr, &time_curr);

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
                               
                               std::vector<cvo::CvoFrame::Ptr> frames,
                               std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> &  gt_poses,
                               int num_neighbors_per_node
                               ) {
  // read edges to construct graph
  std::list<std::pair<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr>> edges;
  std::list<cvo::BinaryState::Ptr> edge_states;
  cvo::BinaryCommutativeMap<int> added_edges;
  //std::list<cvo::BinaryState::Ptr> edge_states_cpu;

  for (int i = 0; i < frames.size(); i++) {
    for (int j = i+1; j < std::min((int)frames.size(), i+1+num_neighbors_per_node); j++) {


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

  /// loop closing constrains
  std::cout<<"BA: Adding loop closing constrains\n";
  const cvo::CvoParams & params = cvo_align.get_params();
  for (int i = 0; i < loop_closures.size(); i++) {
    std::pair<int, int> p = loop_closures[i];
    for (int j = -1; j < 2; j++) {
      int id1 = p.first + j;
      int id2 = p.second + j;
      if (id1 < 0 || id1 > frames.size()-1 || id2 < 0 || id2 > frames.size() -1 )
	continue;
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
    }
  }
  
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


    int expected_points = 5000;
    double intensity_bound = 0.4;
    double depth_bound = 4.0;
    double distance_bound = 40.0;
    int kitti_beam_num = 64;
    cvo::LidarPointSelector lps(expected_points, intensity_bound, depth_bound, distance_bound, kitti_beam_num);
    /*
    // running edge detection + lego loam point selection
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_edge (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_surface (new pcl::PointCloud<pcl::PointXYZI>);
    std::vector<int> selected_edge_inds, selected_loam_inds;
    lps.edge_detection(pc, pc_out_edge, output_depth_grad, output_intenstity_grad, selected_edge_inds);
  
    lps.legoloam_point_selector(pc, pc_out_surface, edge_or_surface, selected_loam_inds);    
    // *pc_out += *pc_out_edge;
    // *pc_out += *pc_out_surface;
    //
    num_points_ = selected_indexes.size();
  */

  if (is_edge_only) {
    cvo::VoxelMap<pcl::PointXYZI> full_voxel(leaf_size);
    for (int k = 0; k < pc_in->size(); k++) {
      full_voxel.insert_point(&pc_in->points[k]);
    }
    std::vector<pcl::PointXYZI*> downsampled_results = full_voxel.sample_points();
    pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZI>);
    for (int k = 0; k < downsampled_results.size(); k++)
      downsampled->push_back(*downsampled_results[k]);
     std::shared_ptr<cvo::CvoPointCloud>  ret(new cvo::CvoPointCloud(downsampled, 5000, 64, cvo::CvoPointCloud::PointSelectionMethod::FULL));
     return ret;
  } else {
    
    /// edge points
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_edge (new pcl::PointCloud<pcl::PointXYZI>);  
    std::vector<int> selected_edge_inds;
    std::vector <double> output_depth_grad;
    std::vector <double> output_intenstity_grad;
    lps.edge_detection(pc_in, pc_out_edge, output_depth_grad, output_intenstity_grad, selected_edge_inds);
    std::unordered_set<int> edge_inds;
    for (auto && j : selected_edge_inds) edge_inds.insert(j);

    /// surface points
    std::vector<float> edge_or_surface;
    std::vector<int> selected_loam_inds;
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_loam (new pcl::PointCloud<pcl::PointXYZI>);        
    lps.loam_point_selector(pc_in, pc_out_loam, edge_or_surface, selected_loam_inds);

    /// declare voxel map
    cvo::VoxelMap<pcl::PointXYZI> edge_voxel(leaf_size); 
    cvo::VoxelMap<pcl::PointXYZI> surface_voxel(leaf_size / 2);

    /// edge and surface downsample
    for (int k = 0; k < pc_out_edge->size(); k++) 
      edge_voxel.insert_point(&pc_out_edge->points[k]);
    std::vector<pcl::PointXYZI*> edge_results = edge_voxel.sample_points();
    //for (int k = 0; k < pc_in->size(); k++)  {
    //  if (edge_or_surface[k] > 0 &&
    //      edge_inds.find(k) == edge_inds.end())
    //    surface_voxel.insert_point(&pc_in->points[k]);
   // }
    for (int k = 0; k < pc_out_loam->size(); k++)  {
      if (edge_or_surface[k] > 0 &&
          edge_inds.find(selected_loam_inds[k]) == edge_inds.end())
        surface_voxel.insert_point(&pc_out_loam->points[k]);
    }
    std::vector<pcl::PointXYZI*> surface_results = surface_voxel.sample_points();
    int total_selected_pts_num = edge_results.size() + surface_results.size();    
    std::shared_ptr<cvo::CvoPointCloud> ret(new cvo::CvoPointCloud(1, NUM_CLASSES));
    ret->reserve(total_selected_pts_num, 1, NUM_CLASSES);
    std::cout<<"edge voxel selected points "<<edge_results.size()<<std::endl;
    std::cout<<"surface voxel selected points "<<surface_results.size()<<std::endl;    

    /// push
    for (int k = 0; k < edge_results.size(); k++) {
      Eigen::VectorXf feat(1);
      feat(0) = edge_results[k]->intensity;
      Eigen::VectorXf semantics = Eigen::VectorXf::Zero(NUM_CLASSES);
      Eigen::VectorXf geo_t(2);
      geo_t << 1.0, 0;
      ret->add_point(k, edge_results[k]->getVector3fMap(),  feat, semantics, geo_t);
    }
    /// surface downsample
    for (int k = 0; k < surface_results.size(); k++) {
      // surface_pcl.push_back(*surface_results[k]);
      Eigen::VectorXf feat(1);
      feat(0) = surface_results[k]->intensity;
      Eigen::VectorXf semantics = Eigen::VectorXf::Zero(NUM_CLASSES);
      Eigen::VectorXf geo_t(2);
      geo_t << 0, 1.0;
      ret->add_point(k+edge_results.size(), surface_results[k]->getVector3fMap(), feat,
                     semantics, geo_t);
    }
    return ret;

  }

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
  //argparse::ArgumentParser parser("irls_kitti_loop_closure_test");
  /// assume start_ind and last_ind has overlap

  
  cvo::KittiHandler kitti(argv[1], cvo::KittiHandler::DataType::LIDAR);
  string cvo_param_file(argv[2]);    
  int num_neighbors_per_node = std::stoi(argv[3]); // forward neighbors
  std::string tracking_traj_file(argv[4]);
  std::string loop_closure_input_file(argv[5]);
  std::string BA_traj_file(argv[6]);
  int is_edge_only = std::stoi(argv[7]);
  std::cout<<"is edge only is "<<is_edge_only<<"\n";
  int start_ind = std::stoi(argv[8]);
  std::cout<<"input start_ind is  "<<start_ind<<"\n";
  int max_last_ind = std::stoi(argv[9]);
  std::cout<<"input last_ind is  "<<max_last_ind<<"\n";
  double cov_scale_t = std::stod(argv[10]);
  double cov_scale_r = std::stod(argv[11]);
  int skipped_frames = std::stoi(argv[12]);
  int is_pgo_only = std::stoi(argv[13]);
  int is_read_loop_closure_poses_from_file = std::stoi(argv[14]);
  
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
  std::string gt_fname("groundtruth.txt");
  cvo::write_traj_file<double, 4, Eigen::ColMajor>(gt_fname,gt_poses);
  std::string track_fname("tracking.txt");
  cvo::write_traj_file<double, 4, Eigen::ColMajor>(track_fname, tracking_poses);
  
  // read point cloud
  std::vector<cvo::CvoFrame::Ptr> frames;
  std::vector<std::shared_ptr<cvo::CvoPointCloud>> pcs;
  if (!(is_pgo_only && is_read_loop_closure_poses_from_file)) {
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
  }
  

  /// global registration
  std::vector<std::pair<int, int>> loop_closures;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> lc_poses;
  if (is_read_loop_closure_poses_from_file) {
    ReadG2oFile(loop_closure_input_file, loop_closures, lc_poses) ;
  } else {
    std::string g_reg_f("global.txt");    
    parse_lc_file(loop_closures, loop_closure_input_file, start_ind);    
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
			  cov_scale_t, cov_scale_r, num_neighbors_per_node);
  std::cout<<"Finish PGO...\n";  
  std::string pgo_fname("pgo.txt");
  cvo::write_traj_file<double, 3, Eigen::RowMajor>(pgo_fname, BA_poses);
  std::string lc_prefix(("loop_closure_"));
  if (pcs.size())
    log_lc_pc_pairs(BA_poses, loop_closures, pcs, lc_prefix);
  
  //std::cout<<"global registration result is \n"<<T_last_to_first<<"\n";
  //std::cout<<"groundtruth result is \n"<<gt_poses.back().inverse()*gt_poses[0]<<"\n";


  if (is_pgo_only) return 0;
  std::cout<<"Start construct BA CvoFrame\n";
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
  std::cout<<"Construct loop BA problem\n";
  construct_loop_BA_problem(cvo_align,
                            loop_closures,
                            frames, gt_poses, num_neighbors_per_node);

  std::cout<<"Write stacked point cloud\n";
  f_name = std::string("after_BA_loop.pcd") ;
  write_transformed_pc(frames, f_name,0, frames.size()-1);
  std::cout<<"Write traj to file\n";
  write_traj_file(BA_traj_file,frames);
  return 0;
}
