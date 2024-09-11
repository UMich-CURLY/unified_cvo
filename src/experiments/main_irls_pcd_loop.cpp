#include <iostream>
#include <list>
#include <vector>
#include <utility>
#include <string>
#include <fstream>
#include <filesystem>
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
//#include <pcl/filters/farthest_point_sampling.h>

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

void write_loop_closure_pcds(std::map<int, cvo::CvoFrame::Ptr> & frames,
                             std::vector<std::pair<int, int>> & loop_closures,
                             bool is_recording_frames,
                             const std::string & name_prefix) {
  if (is_recording_frames) {
    for (auto && [ind, frame ]:frames) {
      frame->points->write_to_pcd(name_prefix + "_"+std::to_string(ind)+".pcd");
    }
  }
  for (auto & p : loop_closures) {

    int f1 = p.first;
    int f2 = p.second;
    
    cvo::CvoPointCloud new_pc;
    Eigen::Matrix4d pose1 = Eigen::Matrix4d::Identity();
    pose1.block<3,4>(0,0) = Eigen::Map<cvo::Mat34d_row>(frames[f1]->pose_vec);
    Eigen::Matrix4d pose2 = Eigen::Matrix4d::Identity();
    pose2.block<3,4>(0,0) = Eigen::Map<cvo::Mat34d_row>(frames[f2]->pose_vec);
    
    Eigen::Matrix4f pose_f = (pose1.inverse() * pose2).cast<float>();
    cvo::CvoPointCloud::transform(pose_f, *frames[f2]->points, new_pc);
    new_pc += *frames[f1]->points;

    new_pc.write_to_color_pcd(name_prefix + "_loop_"+std::to_string(f1)+"_"+std::to_string(f2)+".pcd");
    
  }
}


void pose_graph_optimization( const cvo::aligned_vector<Eigen::Matrix4d> & tracking_poses,
                              const std::vector<std::pair<int, int>> & loop_closures,
                              const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> & lc_poses,
                              const std::string & lc_constrains_file_before_BA,
                              std::map<int, cvo::Mat34d_row> & BA_poses,
                              double cov_scale_t,
			      double cov_scale_r,
                              int num_neighbors_per_node,
                              std::set<int> & selected_inds,
                              int num_merging_sequential_frames,
                              const std::string & pgo_result_file,
			      bool is_running_pgo){
  
  Eigen::Matrix<double, 6,6> information = Eigen::Matrix<double, 6,6>::Identity(); 
  information.block(0,0,3,3) = Eigen::Matrix<double, 3,3>::Identity() / cov_scale_t / cov_scale_t;
  information.block(3,3,3,3) = Eigen::Matrix<double, 3,3>::Identity() / cov_scale_r / cov_scale_r;
  
  ceres::Problem problem;
  cvo::pgo::MapOfPoses poses;
  cvo::pgo::VectorOfConstraints constrains;
  std::ofstream lc_g2o(lc_constrains_file_before_BA);
  std::ofstream pgo_g2o(pgo_result_file);

  std::cout<<"copy from tracking_poses to poses and constrains\n";
  for (auto i : selected_inds) {
    cvo::pgo::Pose3d pose = cvo::pgo::pose3d_from_eigen<double, Eigen::ColMajor>(tracking_poses[i]);
    poses.insert(std::make_pair(i, pose));
    lc_g2o <<cvo::pgo::Pose3d::name()<<" "<<i<<" "<<pose<<"\n";
    pgo_g2o <<cvo::pgo::Pose3d::name()<<" "<<i<<" "<<pose<<"\n";    
  }

  std::cout<<"add between factors for odom\n";
  for (auto iter_i = selected_inds.begin(); iter_i != selected_inds.end(); iter_i++) {
	  int i = *iter_i;
	  std::cout<<"i="<<i<<"\n";
    auto iter_j = iter_i;
    iter_j++;
    if (iter_j == selected_inds.end())
      continue;

    int j = *iter_j;
    std::cout<<"j="<<j<<"\n";

    Eigen::Matrix4d T_Fi_to_Fj = tracking_poses[i].inverse() * tracking_poses[j];
    cvo::pgo::Pose3d t_be = cvo::pgo::pose3d_from_eigen<double, Eigen::ColMajor>(T_Fi_to_Fj);
    std::cout<<__func__<<": Add constrain from "<<i<<" to "<<j<<"\n";
    cvo::pgo::Constraint3d constrain{i, j, t_be, information};
    constrains.push_back(constrain);
    pgo_g2o<<cvo::pgo::Constraint3d::name()<<" "<<constrain<<"\n";        
  }
  
  
  /// the loop closing pose factors
  std::cout<<__func__<<"Loop closure number is "<<loop_closures.size()<<"\n";
  for (int i =0 ; i < loop_closures.size(); i++) {
    auto pair = loop_closures[i];    
    Eigen::Matrix4d T_f1_to_f2;
    if (lc_poses.size() > i)
      T_f1_to_f2 = lc_poses[i].cast<double>();
    else
      T_f1_to_f2 = tracking_poses[pair.first].inverse() * tracking_poses[pair.second];
    cvo::pgo::Pose3d t_be = cvo::pgo::pose3d_from_eigen<double, Eigen::ColMajor>(T_f1_to_f2);

    cvo::pgo::Constraint3d constrain{static_cast<int>(pair.first), static_cast<int>(pair.second),
                                     t_be, information};
    constrains.push_back(constrain);
    lc_g2o<<cvo::pgo::Constraint3d::name()<<" "<<constrain<<"\n";
    pgo_g2o<<cvo::pgo::Constraint3d::name()<<" "<<constrain<<"\n";    
    std::cout<<__func__<<": Add constrain from "<<pair.first<<" to "<<pair.second<<"\n";
  }
  lc_g2o.close();
  pgo_g2o.close();

  /// optimization
  if (is_running_pgo) {
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
  } else {

    for (auto ind : selected_inds) {
      Eigen::Matrix<double, 3, 4> pose = tracking_poses[ind].block(0,0,3,4);
      Eigen::Matrix<double, 3, 4, Eigen::RowMajor> pose_row = pose;
      BA_poses.insert(std::make_pair(ind, pose_row));
    }
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
  init_param.is_global_angle_registration = 1;
  cvo_align.write_params(&init_param);

  lc_poses_f1_to_f2.resize(loop_closures.size());
  std::ofstream f(registration_result_file);
  double time = 0;
  for (int i = 0; i < loop_closures.size(); i++) {
    Eigen::Matrix4f result;
    double time_curr;
    Eigen::Matrix4f init_guess_inv = Eigen::Matrix4f::Identity();
    auto p = loop_closures[i];    
    std::cout<<__func__<<": global reg "<<p.first<<" and "<<p.second<<"\n";
    cvo_align.align(*pcs.at(p.first), *pcs.at(p.second), init_guess_inv, result, nullptr, &time_curr);

    lc_poses_f1_to_f2[i] = result;
    
    std::cout<<"====================================\nFinish running global registration between "<<p.first<<" and "<<p.second<<", result is\n"<<result<<"\n";
    time += time_curr;

    cvo::CvoPointCloud old_pc;
    cvo::CvoPointCloud::transform(init_guess_inv, *pcs.at(p.second), old_pc);
    old_pc += *pcs.at(p.first);
    old_pc.write_to_color_pcd("cvo_before_loop_"+std::to_string(p.first)+"_"+std::to_string(p.second)+".pcd");


    cvo::CvoPointCloud new_pc;
    cvo::CvoPointCloud::transform(result, *pcs.at(p.second), new_pc);
    new_pc += *pcs.at(p.first);
    new_pc.write_to_color_pcd("cvo_after_loop_"+std::to_string(p.first)+"_"+std::to_string(p.second)+".pcd");

    
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
    for (int j = 0; j < 1; j++) {
      int id1 = p.first + j;
      int id2 = p.second + j;
      if (//id1 < 0 || id1 > frames.size()-1 || id2 < 0 || id2 > frames.size() -1
          frames.find(id1) == frames.end() || frames.find(id2) == frames.end())
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
  cvo::CvoBatchIRLS batch_irls_problem(frames_vec, const_flags,
                                       edge_states, &cvo_align.get_params());
  std::string err_file = std::string("err_wrt_iters.txt");
  batch_irls_problem.solve();//gt_poses, err_file);
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
void write_traj_file_kitti_format(std::string & fname,
                                  std::map<int, cvo::CvoFrame::Ptr> & frames ) {
  std::ofstream outfile(fname);
  for (auto & [i, ptr] : frames) {
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3,4>(0,0) = Eigen::Map<cvo::Mat34d_row>(ptr->pose_vec);
    for (int j = 0; j < 12; j++) {
      outfile <<pose(j / 4, j % 4);
      if (j == 11)
        outfile<<"\n";
      else
        outfile<<" ";
    }

    
  }
  outfile.close();
}

/*
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
*/
void write_transformed_pc(std::map<int, cvo::CvoFrame::Ptr> & frames,
                          std::string & fname,
                          int start_frame_ind=0, int end_frame_ind=1000000){
  pcl::PointCloud<pcl::PointXYZRGB> pc_all;
  pcl::PointCloud<cvo::CvoPoint> pc_xyz_all;
  for (auto & [i, ptr] : frames) {
  //for (int i = start_frame_ind; i <= std::min((int)frames.size(), end_frame_ind); i++) {
  //auto ptr = frames[i];

    cvo::CvoPointCloud new_pc;
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3,4>(0,0) = Eigen::Map<cvo::Mat34d_row>(ptr->pose_vec);
    
    Eigen::Matrix4f pose_f = pose.cast<float>();
    cvo::CvoPointCloud::transform(pose_f, *ptr->points, new_pc);

    pcl::PointCloud<pcl::PointXYZRGB> pc_curr;
    pcl::PointCloud<cvo::CvoPoint> pc_xyz_curr;
    //new_pc.export_semantics_to_color_pcd(pc_curr);
    new_pc.export_to_pcd<pcl::PointXYZRGB>(pc_curr);
    new_pc.export_to_pcd(pc_xyz_curr);

    pc_all += pc_curr;
    pc_xyz_all += pc_xyz_curr;

  }
  std::string fname_color = fname + ".semantic_color.pcd";
  pcl::io::savePCDFileASCII(fname_color, pc_all);
  pcl::io::savePCDFileASCII(fname, pc_xyz_all);
}




void sample_frame_inds(int start_ind, int end_ind, int num_merged_frames,
                       const std::vector<std::pair<int, int>> &loop_closures,
                       const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> tracking_poses,                  
                       std::set<int> & result_selected_frames
                       ) {
  result_selected_frames.clear();
  for (int i = start_ind; i <= end_ind; i+=(1+num_merged_frames)){
    result_selected_frames.insert(i);
  }
  ASSERT(loop_closures.size() > 0, "lc edges must be non-empty");
  for (auto && p : loop_closures) {
    if(result_selected_frames.find(p.first) == result_selected_frames.end()) {
      result_selected_frames.insert(p.first);
	std::cout<<"Insert lc frame "<<p.first<<"\n";
    }
    if(result_selected_frames.find(p.second) == result_selected_frames.end()) {
      result_selected_frames.insert(p.second);
	std::cout<<"Insert lc frame "<<p.second<<"\n";
    }
  }
}




int main(int argc, char** argv) {

  //  omp_set_num_threads(24);
  //argparse::ArgumentParser parser("irls_kitti_loop_closure_test");
  /// assume start_ind and last_ind has overlap

  std::cout<<"Start \n";
  std::string data_type(argv[1]); /// ply or pcd
  std::string data_path(argv[2]); /// folder that holds all semantic pcds
  string cvo_param_file(argv[3]);    
  int num_neighbors_per_node = std::stoi(argv[4]); // forward neighbors
  std::string tracking_traj_file(argv[5]);
  std::string loop_closure_input_file(argv[6]);
  std::string BA_traj_file(argv[7]);
  int is_read_loop_closure_poses_from_file = std::stoi(argv[8]); /// 0: from text file.  1: from g2o file. 2: gen by program
  //std::cout<<"is edge only is "<<is_edge_only<<"\n";
  int start_ind = std::stoi(argv[9]);
  std::cout<<"input start_ind is  "<<start_ind<<"\n";
  int max_last_ind = std::stoi(argv[10]);
  std::cout<<"input last_ind is  "<<max_last_ind<<"\n";
  double cov_scale_t = std::stod(argv[11]);
  double cov_scale_r = std::stod(argv[12]);
  int num_merging_sequential_frames = std::stoi(argv[13]);
  int is_doing_pgo = std::stoi(argv[14]);
  int is_read_pcd = std::stoi(argv[15]); //// 0: not read. 1: read and downsample; 2: read downsampled
  int is_store_pcd_each_frame = std::stoi(argv[16]);
  int is_global_registration = std::stoi(argv[17]);
  int is_doing_ba = std::stoi(argv[18]);
  int is_save_pcd = std::stoi(argv[19]);
  //int is_semantic = std::stoi(argv[19]);

  std::size_t num_pcds = (std::size_t)std::distance(std::filesystem::directory_iterator{std::filesystem::path(data_path)}, std::filesystem::directory_iterator{});
  int last_ind = std::min(max_last_ind, (int)num_pcds-1);
  std::cout<<"actual last ind is "<<last_ind<<", total pcds is "<<num_pcds<<"\n";

  std::cout<<"Finish reading all arguments\n";
  int total_iters = last_ind - start_ind + 1;

 
  std::cout<<"Init CvoGPU\n";
  cvo::CvoGPU cvo_align(cvo_param_file);

  /// read poses
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> tracking_poses,
    gt_poses;
  std::map<int, cvo::Mat34d_row> BA_poses;

  /// read tracking files
  std::cout<<"Read traj init file "<<tracking_traj_file<<"\n";
  cvo::read_pose_file_kitti_format(tracking_traj_file,
                                   start_ind,
                                   last_ind,
                                   tracking_poses
                                   );
  std::cout<<"init tracking pose size is "<<tracking_poses.size()<<"\n";

  // read loop closure files
  std::vector<std::pair<int, int>> loop_closures;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> lc_poses;
  if (is_read_loop_closure_poses_from_file == 1) {
    cvo::ReadG2oFile(loop_closure_input_file, loop_closures, lc_poses) ;
  } else if (is_read_loop_closure_poses_from_file == 0){
    parse_lc_file(loop_closures, loop_closure_input_file, start_ind);    
  }
  
  
  /// decide if we will skip frames
  std::set<int> result_selected_frames;
  sample_frame_inds(start_ind, last_ind, num_merging_sequential_frames, loop_closures, tracking_poses, result_selected_frames);
  std::cout<<"loop closures size is "<<loop_closures.size()<<"\n";
  std::string track_fname("tracking.txt");
  cvo::write_traj_file_kitti_format<double, 4, Eigen::ColMajor>(track_fname, tracking_poses, result_selected_frames);
  
  // read point cloud

  std::map<int, std::shared_ptr<cvo::CvoPointCloud>> pcs;
  if (is_read_pcd == 2) {
    for (auto i : result_selected_frames) {
      pcl::PointCloud<cvo::CvoPoint>::Ptr pc_pcl(new pcl::PointCloud<cvo::CvoPoint>);

      if (cvo_align.get_params().is_using_semantics)
        pcl::io::loadPCDFile<cvo::CvoPoint>(data_path+"/"+std::to_string(i)+".pcd", *pc_pcl);
      else {

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::io::loadPCDFile<pcl::PointXYZRGB>(data_path+"/"+std::to_string(i)+".pcd", *pc_rgb);
        pcl::PointSeg_from_PointXYZRGB<FEATURE_DIMENSIONS,NUM_CLASSES,pcl::PointXYZRGB>(*pc_rgb, *pc_pcl);
        
      }
      std::shared_ptr<cvo::CvoPointCloud> ret(new cvo::CvoPointCloud(FEATURE_DIMENSIONS, NUM_CLASSES));
      for (int k = 0; k < pc_pcl->size(); k++) {
        cvo::CvoPoint p = (*pc_pcl)[k];
        ret->push_back(p);
      }
      pcs.insert(std::make_pair(i, ret));
    }
  } else if (is_read_pcd == 1) {
    for (auto i : result_selected_frames) {
      pcl::PointCloud<cvo::CvoPoint>::Ptr pc_local(new pcl::PointCloud<cvo::CvoPoint>);
      for (int j = 0; j < 1+num_merging_sequential_frames; j++){
        int index_j = i+j;
        pcl::PointCloud<cvo::CvoPoint>::Ptr pc_pcl(new pcl::PointCloud<cvo::CvoPoint>);
        if (cvo_align.get_params().is_using_semantics)
          pcl::io::loadPCDFile<cvo::CvoPoint>(data_path+"/"+std::to_string(index_j)+".pcd", *pc_pcl);
        else {
          pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
          pcl::io::loadPCDFile<pcl::PointXYZRGB>(data_path+"/"+std::to_string(i)+".pcd", *pc_rgb);
          pcl::PointSeg_from_PointXYZRGB<FEATURE_DIMENSIONS,NUM_CLASSES,pcl::PointXYZRGB>(*pc_rgb, *pc_pcl);
        
        }
        //if (j > 0) {
        Eigen::Matrix4f pose_fi_to_fj = (tracking_poses[i].inverse() * tracking_poses[j+i]).cast<float>();
#pragma omp parallel for 
        for (int k = 0; k < pc_pcl->size(); k++) {
          auto & p = pc_pcl->at(k);
          
          p.getVector3fMap() = pose_fi_to_fj.block(0,0,3,3) * p.getVector3fMap() + pose_fi_to_fj.block(0,3,3,1);
          if (k == 0) {
            auto & p_c = (*pc_pcl)[k];            
            Eigen::Matrix3f feat_map;
            feat_map(0) = static_cast<float>(p_c.r);
            feat_map(1) = static_cast<float>(p_c.g);
            feat_map(2) = static_cast<float>(p_c.b);
            std::cout<<"Read rgb: "<<feat_map * 255<<"\n";          
            std::cout<<"Read rgb end\n " ;
          }

        }
          //}
        *pc_local += *pc_pcl;
      }

      /// downsample
      std::cout<<"frame "<<i<<" before downsample points "<<pc_local->size()<<std::endl;
      /*
      pcl::PointCloud<cvo::CvoPoint>::Ptr pcd_downsampled(new pcl::PointCloud<cvo::CvoPoint>);
      pcl::PassThrough<cvo::CvoPoint> pass;
      pass.setInputCloud (pc_local);
      pass.setFilterFieldName ("z");
      pass.setFilterLimits (0.0, 3.0);
      pass.filter (*pcd_downsampled);
      
      pcl::FarthestPointSampling downsample;
      downsample.setSample(1024);
      downsample.setInputCloud (pcd_downsampled);
      downsample.filter (*pcd_downsampled);
      
      std::shared_ptr<cvo::CvoPointCloud> ret(new cvo::CvoPointCloud(*pcd_downsampled));
      //pcs.push_back(pc);
      */
      
      cvo::VoxelMap<cvo::CvoPoint> edge_voxel(cvo_align.get_params().multiframe_downsample_voxel_size); 
      for (int k = 0; k < pc_local->size(); k++) {
        if ( (*pc_local)[k].getVector3fMap().norm() < 10.0 ) {
          edge_voxel.insert_point(&(*pc_local)[k]);
        }
      }
      std::vector<cvo::CvoPoint*> edge_results = edge_voxel.sample_points();
      std::shared_ptr<cvo::CvoPointCloud> ret(new cvo::CvoPointCloud(FEATURE_DIMENSIONS, NUM_CLASSES));
      std::cout<<"frame "<<i<<" selected points "<<edge_results.size()<<std::endl;
      for (int k = 0; k < edge_results.size(); k++) {
        cvo::CvoPoint p = *edge_results[k];
        //Eigen::Map<Eigen::Matrix<float, FEATURE_DIMENSIONS, 1>> feat_map(p.features);
        if (k == 0) {
          Eigen::Matrix3f feat_map;
          feat_map(0) = static_cast<float>(p.r);
          feat_map(1) = static_cast<float>(p.g);
          feat_map(2) = static_cast<float>(p.b);
          std::cout<<"Read rgb: "<<feat_map * 255<<"\n";          
          std::cout<<"Read rgb end\n " ;
        }
        //feat_map.normalize();
        ret->push_back(p);
      }
      

      pcs.insert(std::make_pair(i, ret));

      if (is_save_pcd) {
        if (cvo_align.get_params().is_using_semantics) {
          pcl::PointCloud<pcl::PointXYZRGB> pc_semantic_downsampled;
          ret->export_semantics_to_color_pcd(pc_semantic_downsampled);
          pcl::io::savePCDFile(std::to_string(i)+"_color.pcd", pc_semantic_downsampled);
        }
        pcl::PointCloud<cvo::CvoPoint> pc_cvo_downsampled;
        ret->export_to_pcd<cvo::CvoPoint>(pc_cvo_downsampled);
        //pcl::io::savePCDFileASCII(std::to_string(i)+".pcd", pc_cvo_downsampled);
        pcl::io::savePCDFile(std::to_string(i)+".pcd", pc_cvo_downsampled);
      }
    }

  }


  ///  visualizion: lay out the point cloud based on tracking poses
  std::cout<<"Merge all with tracking poses\n";
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_all_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
  for (auto && [ind, pc]: pcs) {
    Eigen::Matrix4f tracking_pose = tracking_poses[ind].cast<float>();
    pcl::PointCloud<pcl::PointXYZRGB> pc_curr;
    pc_curr.resize(pc->size());
    #pragma omp parallel for
    for (int j = 0; j < pc->size(); j++)  {
      pc_curr[j].getVector3fMap() = tracking_pose.block(0,0,3,3) * pc->at(j) + tracking_pose.block(0,3,3,1);
      pc_curr[j].rgb = pc->point_at(j).rgb;
    }
    (*pc_all_ptr) += pc_curr;
  }
  pcl::io::savePCDFileASCII ("tracking.pcd", *pc_all_ptr);
  std::cout<<"Just wrote to tracking.pcd";

  
  

  /// global registration
  if (is_read_pcd
      && is_global_registration) {
    std::cout<<"Start global reg";    
    std::string g_reg_f("global.txt");    
    global_registration_batch(cvo_align,
                              loop_closures,
                              pcs,
                              gt_poses,
                              g_reg_f,
                              lc_poses);
    std::cout<<__func__<<": End global reg";        
  }
  

  /// pose graph optimization
  if(is_doing_pgo) {
    std::string lc_g2o("loop_closures.g2o");
    std::string pgo_g2o("pgo.g2o");    
    std::cout<<"Start PGO...\n";
    pose_graph_optimization(tracking_poses, loop_closures,
                            lc_poses, lc_g2o,
                            BA_poses, 
                            cov_scale_t, cov_scale_r, num_neighbors_per_node,
                            result_selected_frames,
                            num_merging_sequential_frames,
                            pgo_g2o,
			    is_doing_pgo);


    /// log pgo results
    std::cout<<"Finish PGO...\n";  
    std::string pgo_fname("pgo.txt");
    std::vector<cvo::Mat34d_row, Eigen::aligned_allocator<cvo::Mat34d_row>> pgo_poses;
    for (auto i : result_selected_frames) pgo_poses.push_back(BA_poses[i]);
    cvo::write_traj_file_kitti_format<double, 3, Eigen::RowMajor>(pgo_fname, pgo_poses);
    std::string lc_prefix(("loop_closure_"));
    if (pcs.size())
      log_lc_pc_pairs(BA_poses, loop_closures, pcs, lc_prefix);
  }

  /// construct BA CvoFrame struct  
  std::cout<<"Start construct BA CvoFrame\n";
  std::map<int, cvo::CvoFrame::Ptr> frames;  
  for (auto i : result_selected_frames) {
    std::cout<<"Copy "<<i<<"th point cloud to gpu \n";
    double * poses_data = BA_poses[i].data();
    cvo::CvoFrame::Ptr new_frame(new cvo::CvoFrameGPU(pcs[i].get(), poses_data, cvo_align.get_params().is_using_kdtree));
    frames.insert(std::make_pair(i, new_frame));
  }
  std::string f_name = std::string("before_BA_loop.pcd");
  write_transformed_pc(frames, f_name, 0, frames.size()-1);

  /// Multiframe alignment  
  if (is_doing_ba) {

    //if (is_read_loop_closure_poses_from_file == 2) {
    for (auto iter_i = result_selected_frames.begin(); iter_i != result_selected_frames.end(); iter_i++) {
      int i = *iter_i;
    
      auto iter_j = iter_i;
      iter_j++;
      for (int k = 0; k < num_neighbors_per_node && iter_j != result_selected_frames.end(); k++)
        iter_j++;

      //for (; std::distance(iter_i, iter_j) <= num_neighbors_per_node && iter_j != frames.end() ; iter_j++) {
      for (; iter_j != result_selected_frames.end() ; iter_j++) {
        int j = *iter_j;
        double dist = (tracking_poses[i].block<3,1>(0,3) - tracking_poses[j].block<3,1>(0,3)).norm();
        if (dist < 0.2 ){
          std::cout<<"loop: dist betwee "<<i<<" and "<<j<<" is "<<dist<<"\n";
          loop_closures.push_back(std::make_pair(i,j));
        }
      }
    }
      //}    
    std::cout<<"Construct loop BA problem\n";
    write_loop_closure_pcds( frames, loop_closures, true, "before_ba_");
    construct_loop_BA_problem(cvo_align,
                              loop_closures,
                              frames, gt_poses, num_neighbors_per_node,
                              num_merging_sequential_frames);
    write_loop_closure_pcds( frames, loop_closures, false, "after_ba_");
    std::cout<<"Write stacked point cloud\n";
    f_name = std::string("after_BA_loop.pcd") ;
    write_transformed_pc(frames, f_name,0, frames.size()-1);
    std::cout<<"Write traj to file\n";
    write_traj_file_kitti_format(BA_traj_file,frames);
  }
  return 0;
}
