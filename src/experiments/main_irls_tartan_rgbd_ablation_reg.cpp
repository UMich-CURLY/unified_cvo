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
#include "dataset_handler/TartanAirHandler.hpp"
#include "utils/PoseLoader.hpp"
#include "utils/ImageRGBD.hpp"
#include "utils/Calibration.hpp"
#include "utils/ImageDownsampler.hpp"

using namespace std;

extern template class cvo::VoxelMap<pcl::PointXYZRGB>;
extern template class cvo::Voxel<pcl::PointXYZRGB>;
// extern template class Foo<double>;


void construct_BA_problem(cvo::CvoGPU & cvo_align,
                          std::map<int, cvo::CvoFrame::Ptr> frames,
                          std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> &  gt_poses,
                          int start_frame_ind,
                          int num_BA_frames
                          ) {
  // read edges to construct graph
  //int start_frame_ind = last_frame_ind - num_BA_frames + 1;
  std::list<std::pair<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr>> edges;
  std::list<cvo::BinaryState::Ptr> edge_states;
  //std::list<cvo::BinaryState::Ptr> edge_states_cpu;
  //for (int i = start_frame_ind; i < start_frame_ind + num_BA_frames; i++) {
  for (auto && [i, ptr1]: frames) {
    //for (int j = i+1; j < start_frame_ind + num_BA_frames; j++) {
    for (auto && [j, ptr2]: frames) {
      if (j <= i) continue;

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
  double time = 0;
  std::vector<bool> const_flags(frames.size(), false);
  const_flags[0] = true;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> gt_poses_sub;
  for (auto && [j, ptr] : frames)
    gt_poses_sub.push_back(gt_poses[j]);
  
  auto start = std::chrono::system_clock::now();
  std::vector<cvo::CvoFrame::Ptr> frames_vec;
  for (auto && [ind, ptr]: frames) frames_vec.push_back(ptr);
  cvo::CvoBatchIRLS batch_irls_problem(frames_vec, const_flags,
                                       edge_states, &cvo_align.get_params());
  std::string err_file = std::string("err_wrt_iters_") + std::to_string(start_frame_ind)+ ".txt";
  batch_irls_problem.solve(gt_poses_sub, err_file);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> t_all = end - start;
  // cvo_align.align(frames, const_flags,
  //               edge_states, &time);

  std::cout<<"GPU Align ends. Total time is "<<double(t_all.count()) / 1000<<" seconds."<<std::endl;
  
}



void write_traj_file(std::string & fname,
                     std::map<int, cvo::CvoFrame::Ptr> & frames ) {
  std::ofstream outfile(fname);
  //for (int i = 0; i< frames.size(); i++) {
  for (auto && [i, ptr] : frames) {
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


void transform_vector_of_poses(std::vector<Sophus::SE3d> & poses_in,
                               const Sophus::SE3d & pose_anchor,
                               std::vector<Sophus::SE3d> & poses_out
                               ) {

  Sophus::SE3d Tb =   poses_in[0].inverse() * pose_anchor;
  
  poses_out.resize(poses_in.size());
  for (int i = 1; i < poses_in.size(); i++) {
    poses_out[i] = Tb.inverse() * poses_in[i] * Tb;
  }
  if (poses_out.size() > 0) {
    poses_out[0] = pose_anchor;
  }
    
}


double log_err_of_all_poses(std::vector<Sophus::SE3d> & poses_old,
                            std::vector<Sophus::SE3d> & poses_new) {
  double change = 0;
  for (int i = 0; i < poses_old.size(); i++) {
    change += (poses_old[i].inverse() * poses_new[i]).log().squaredNorm();
  }
  return change;
}

double frobenius_err_of_all_poses(std::vector<Sophus::SE3d> & poses_old,
                                  std::vector<Sophus::SE3d> & poses_new) {
  double change = 0;
  for (int i = 0; i < poses_old.size(); i++) {
    change += (poses_old[i].matrix() -  poses_new[i].matrix()).squaredNorm();
  }
  return change;
} 


enum POSE_ERR_METRIC {LOG, FROBENIUS};

double eval_pose(POSE_ERR_METRIC metric,
                 const std::map<int, cvo::CvoFrame::Ptr> & frames,
                 const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> & gt_poses_all) {
  std::vector<Sophus::SE3d> poses;
  std::vector<Sophus::SE3d> gt;

  
  for (auto && [ind, frame] : frames) {
    Eigen::Matrix4d pose = frame->pose_cpu();
    Sophus::SE3d T(pose.block<3,3>(0,0), pose.block<3,1>(0,3));
    poses.push_back(T);
    Sophus::SE3d T_gt(gt_poses_all[ind].block<3,3>(0,0), gt_poses_all[ind].block<3,1>(0,3));
    gt.push_back(T_gt);
  }

  //transform_vector_of_poses(gt,
  //                          poses[0],
  //                            gt
  //                          );
  

  if (metric == LOG) {
    return log_err_of_all_poses(poses, gt);
  } else {
    return frobenius_err_of_all_poses(poses, gt);
  }
  
}

void write_transformed_pc(std::map<int, cvo::CvoFrame::Ptr> & frames, std::string & fname) {

  pcl::PointCloud<pcl::PointXYZRGB> pc_all;
  pcl::PointCloud<pcl::PointXYZ> pc_xyz_all;

  //for (int i = start_frame_ind; i <= std::min((int)frames.size(), end_frame_ind); i++) {
  for (auto && [ind, ptr]: frames) {
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

  enum NoiseType {SEMANTIC=0, GEOMETRIC};

  cvo::TartanAirHandler tartan(argv[1]);
  tartan.set_depth_folder_name("deep_depth");
  string cvo_param_file(argv[2]);    
  string calib_file_name(argv[3]);
  int num_BA_frames = std::stoi(argv[4]);
  std::string result_file_folder(argv[5]);
  int sky_label = std::stoi(argv[6]);
  std::cout<<"sky_label is  "<<sky_label<<"\n";
  int noise_type_int = std::stoi(argv[7]);
  float noise_sigma = std::stof(argv[8]);
  float max_depth = std::stof(argv[9]);
  int every_n_frames_in_BA = std::stoi(argv[10]);
  int total_BA_per_seq = std::stoi(argv[11]);
  int last_ind = std::stoi(argv[12]);


  NoiseType noise_type = static_cast<NoiseType>(noise_type_int);

  cvo::CvoGPU cvo_align(cvo_param_file);
  string calib_file, gt_pose_name;
  calib_file = string(argv[1] ) +"/" + calib_file_name;
  gt_pose_name = std::string(argv[1]) + "/pose_left.txt";
  cvo::Calibration calib(calib_file, cvo::Calibration::RGBD);

  std::cout<<"Finish reading all arguments\n";
  last_ind = std::min(last_ind, tartan.get_total_number() - num_BA_frames + 1);  
  int total_iters = std::min(tartan.get_total_number(), last_ind + 1) ;
  int num_skipping_frames = total_iters / total_BA_per_seq;

  /// read poses
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> gt_poses_gt_frame, gt_poses(total_iters);


  cvo::Mat34d_row init_pose;
  init_pose << 1,0,0,0,0,1,0,0,0,0,1,0;
  std::vector<cvo::Mat34d_row,
              Eigen::aligned_allocator<cvo::Mat34d_row>> BA_poses(total_iters, init_pose);

  cvo::read_pose_file_tartan_format(gt_pose_name,
                                    0,
                                    total_iters -1,
                                    gt_poses_gt_frame);

  /// change frame from gt frame to camera frame
  Eigen::Matrix4d our_frame_from_gt_cam_frame;
  for (int j = 0; j < gt_poses_gt_frame.size(); j++) {
    Eigen::Matrix3d m;
    m = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ())
      * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitX());
    Eigen::Matrix4d ned_from_us = Eigen::Matrix4d::Identity();
    ned_from_us.block<3,3>(0,0) = m.inverse();
    /// convert from NED to our coordinate system    
    gt_poses[j] = ned_from_us * gt_poses_gt_frame[j] * ned_from_us.inverse();
    if (j == 0) {
      our_frame_from_gt_cam_frame = gt_poses[0].inverse();
      gt_poses[0] = Eigen::Matrix4d::Identity();
    }  else {
      gt_poses[j] = (our_frame_from_gt_cam_frame * gt_poses[j]).eval();
    }
  }

  // read point cloud
  std::map<int, std::shared_ptr<cvo::CvoPointCloud>> pcs;
  std::map<int, std::shared_ptr<cvo::CvoPointCloud>> pcs_full;
  for (int j = 0; j< total_iters ; j+=num_skipping_frames) {
    for (int i = 0; i < num_BA_frames; i++) {
      int ind = j+i*every_n_frames_in_BA;
      if (ind >= total_iters) continue;
      std::cout<<"Read frame "<<ind<<"\n";
      tartan.set_start_index(ind);
      cv::Mat rgb;
      vector<float> depth, semantics;
      tartan.read_next_rgbd_without_sky(rgb, depth, NUM_CLASSES, semantics, sky_label, noise_sigma, max_depth);
    
      std::shared_ptr<cvo::ImageRGBD<float>> raw(new cvo::ImageRGBD<float>(rgb, depth, NUM_CLASSES, semantics));
    
      std::shared_ptr<cvo::CvoPointCloud> pc_full = std::make_shared<cvo::CvoPointCloud> (*raw,  calib, cvo::CvoPointCloud::FULL);
      std::shared_ptr<cvo::CvoPointCloud> pc_edge_raw(new cvo::CvoPointCloud(*raw, calib, cvo::CvoPointCloud::DSO_EDGES));
    
      float leaf_size = cvo_align.get_params().multiframe_downsample_voxel_size;  
      std::shared_ptr<cvo::CvoPointCloud> pc = cvo::rgbd_downsampling_single_frame(pc_full, pc_edge_raw,
                                                                                   leaf_size);
      pcs[ind] = pc;
      pcs_full[ind] = pc_full;
    }
  }

  std::map<int, cvo::CvoFrame::Ptr> frames;
  std::map<int, std::shared_ptr<cvo::CvoFrame>> frames_full;  
  for (int i = 0; i< total_iters; i+=num_skipping_frames) {

    /// construct CvoFrame
    frames.clear();
    frames_full.clear();
    for (int j = 0; j < num_BA_frames; j++) {
      int ind = i+j*every_n_frames_in_BA;
      if (ind >= total_iters) break;
      double * poses_data = nullptr;
      Eigen::Matrix<double, 3, 4, Eigen::RowMajor> gt_i_34;//# = gt_i_row.block<3,4>(0,0);
      if (gt_poses.size()) {
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> gt_i_row = gt_poses[i];
	gt_i_34 = gt_i_row.block<3,4>(0,0);
        poses_data = gt_i_34.data();
      } else 
        poses_data = BA_poses[ind].data();
      cvo::CvoFrame::Ptr new_frame(new cvo::CvoFrameGPU(pcs[ind].get(), poses_data, cvo_align.get_params().is_using_kdtree));
      cvo::CvoFrame::Ptr new_full_frame(new cvo::CvoFrameGPU(pcs_full[ind].get(), poses_data,  cvo_align.get_params().is_using_kdtree));
      std::cout<<"Construct CvoFrame pose: \n"<<new_frame->pose_cpu()<<"\n";
      frames[ind] = (new_frame);
      frames_full[ind] = (new_full_frame);
    }

    // multiframe BA
    int start_frame = i;//-num_BA_frames+1;
    int end_frame = i + (num_BA_frames-1) * every_n_frames_in_BA;
    if (end_frame >= total_iters) break;
    std::string f_name_full = std::string("before_BA_full_") + std::to_string(i) + std::string(".pcd") ;
    std::string f_name = std::string("before_BA_") + std::to_string(i) + std::string(".pcd") ;
    write_transformed_pc(frames, f_name);

    /// Multiframe alignment
    construct_BA_problem(cvo_align, frames, gt_poses, start_frame, num_BA_frames);

    std::cout<<"copy result to frames_full\n";
    //for (int j = i-num_BA_frames+1; j < end_frame+1; j++)
    for (auto && [j, ptr] : frames ) 
      memcpy(frames_full[j]->pose_vec, frames[j]->pose_vec, sizeof(double)*12);
    std::cout<<"write results to pcd\n";
    f_name = std::string("after_BA_") + std::to_string(i) + std::string(".pcd") ;
    f_name_full = std::string("after_BA_full_") + std::to_string(i) + std::string(".pcd") ;
    //write_transformed_pc(frames_full, f_name_full, start_frame, end_frame);
    //if (i == 0)
    write_transformed_pc(frames, f_name);


    std::string result_fro_err_file = result_file_folder + "/fro_err.txt";
    std::string result_log_err_file = result_file_folder + "/log_err.txt";
    std::string result_pose_file = result_file_folder+ "/"+ std::to_string(start_frame) + ".txt";
    std::string result_gt_file = result_file_folder+ "/gt_"+ std::to_string(start_frame) + ".txt";
    std::ofstream f;
    f.open(result_fro_err_file, std::ifstream::app);
    f<<eval_pose(FROBENIUS, frames, gt_poses)<<"\n";
    f.close();
    f.open(result_log_err_file, std::ifstream::app);
    f<<eval_pose(LOG, frames, gt_poses)<<"\n";
    f.close();
    write_traj_file(result_pose_file, frames );
    cvo::write_traj_file<double, 4, Eigen::ColMajor>(result_gt_file, gt_poses, i, i+num_BA_frames-1);


    pcs[start_frame].reset();
    pcs_full[start_frame].reset();
    dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[start_frame])->clear_points();
    dynamic_pointer_cast<cvo::CvoFrameGPU>(frames_full[start_frame])->clear_points();
      

  }


  return 0;
}
