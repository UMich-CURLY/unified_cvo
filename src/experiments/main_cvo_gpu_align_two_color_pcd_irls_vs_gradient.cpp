#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
#include <pcl/common/io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/features/normal_3d.h>

//#include "dataset_handler/KittiHandler.hpp"
#include "utils/ImageStereo.hpp"
#include "utils/Calibration.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoGPU.hpp"
#include "cvo/CvoParams.hpp"
#include "cvo/IRLS_State_GPU.hpp"
#include "cvo/IRLS_State.hpp"
#include "cvo/CvoFrameGPU.hpp"
#include "cvo/IRLS.hpp"
#include "utils/VoxelMap.hpp"
#include <Eigen/Geometry>
#include <Eigen/src/Core/util/Constants.h>
#include <iostream>
#include <list>
#include <cmath>
#include <fstream>
//#include <experimental/filesystem>
#include "utils/def_assert.hpp"
#include "utils/GassianMixture.hpp"
#include "utils/eigen_utils.hpp"
//#include "utils/Augmentation.hpp"
//#include "utils/pcl_utils.hpp"
#include <pcl/common/io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <boost/filesystem.hpp>
#include <vector>
#include <utility>
#include <random>
#include <string>
#include <fstream>
#include <sstream>
#include <set>
#include <Eigen/Dense>
#include "cvo/CvoGPU.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoFrame.hpp"
#include "cvo/CvoFrameGPU.hpp"
#include "cvo/IRLS_State.hpp"
#include "cvo/IRLS_State_GPU.hpp"
#include "utils/VoxelMap.hpp"
#include "utils/VoxelMap_impl.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


using namespace std;
using namespace boost::filesystem;

extern template class cvo::VoxelMap<pcl::PointXYZRGB>;
extern template class cvo::Voxel<pcl::PointXYZRGB>;



Eigen::Matrix4f  construct_loop_BA_problem(cvo::CvoGPU & cvo_align,
                                           std::shared_ptr<cvo::CvoPointCloud> pc1,
                                           std::shared_ptr<cvo::CvoPointCloud> pc2,
                                           const Eigen::Matrix4f & init_pose,
                                           double & time,
                                           int & num_iters
                               ) {

  Eigen::Matrix<double, 3, 4, Eigen::RowMajor> pose1, pose2;
  Eigen::Matrix<double, 4, 4, Eigen::ColMajor> pose2_c = init_pose.cast<double>();
  pose2 = pose2_c.block(0,0,3,4);
  pose1 << 1.0,0,0,0,0,1.0,0,0,0,0,1.0,0;

  cvo::CvoFrame::Ptr frame1(new cvo::CvoFrameGPU(pc1.get(), pose1.data(), cvo_align.get_params().is_using_kdtree));
  cvo::CvoFrame::Ptr frame2(new cvo::CvoFrameGPU(pc2.get(), pose2.data(), cvo_align.get_params().is_using_kdtree));

  // read edges to construct graph
  //std::list<std::pair<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr>> edges;
  std::list<cvo::BinaryState::Ptr> edge_states;
  // cvo::BinaryCommutativeMap<int> added_edges;
  std::vector<cvo::CvoFrame::Ptr> frames_vec;
  frames_vec.push_back(frame1);
  frames_vec.push_back(frame2);
  //std::list<cvo::BinaryState::Ptr> edge_states_cpu;

  //for (int i = 0; i < frames.size(); i++) {
    // for (int j = i+1; j < std::min((int)frames.size(), i+1+num_neighbors_per_node); j++) {

  const cvo::CvoParams & params = cvo_align.get_params();
  cvo::BinaryStateGPU::Ptr edge_state(new cvo::BinaryStateGPU(std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frame1),
                                                              std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frame2),
                                                              &params,
                                                              cvo_align.get_params_gpu(),
                                                              params.multiframe_num_neighbors,
                                                              params.multiframe_ell_init// * 4
                                                              ));
  edge_states.push_back(edge_state);
  //double time = 0;
  std::vector<bool> const_flags(2, false);
  const_flags[0] = true;
  
  auto start = std::chrono::system_clock::now();
  cvo::CvoBatchIRLS batch_irls_problem(frames_vec, const_flags,
                                       edge_states, &cvo_align.get_params());
  std::string err_file = std::string("err_wrt_iters.txt");

  
  num_iters = batch_irls_problem.solve();
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> t_all = end - start;
  // cvo_align.align(frames, const_flags,
  //               edge_states, &time);

  std::cout<<"GPU Align ends. Total time is "<<double(t_all.count()) / 1000<<" seconds."<<std::endl;
  time = static_cast<double>( t_all.count() )/1000;

}


Eigen::Vector3f get_pc_mean(const cvo::CvoPointCloud & pc) {
  Eigen::Vector3f p_mean_tmp = Eigen::Vector3f::Zero();
  for (int k = 0; k < pc.num_points(); k++)
//    p_mean_tmp = (p_mean_tmp + pc.positions()[k]).eval();
    p_mean_tmp = (p_mean_tmp + pc.at(k)).eval();
  p_mean_tmp = (p_mean_tmp) / pc.num_points();    
  return p_mean_tmp;
}



void gen_random_poses(std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> & poses, int num_poses,
                      float max_angle_axis=1.0, // max: [0, 1),
                      float max_trans=0.5
                      ) {
  poses.resize(num_poses);
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<> dist(0, max_angle_axis);
  std::uniform_real_distribution<> dist_trans(0,max_trans);
  for (int i = 0; i < num_poses; i++) {
    if (i != 0) {
      Eigen::Matrix3f rot;
      Eigen::Vector3f axis;
      axis << (float)dist_trans(e2), (float)dist_trans(e2), (float)dist_trans(e2);
      axis = axis.normalized().eval();

      Eigen::Vector3f angle_vec = axis * dist(e2) * M_PI;
      rot = Eigen::AngleAxisf(angle_vec[0], Eigen::Vector3f::UnitX())
        * Eigen::AngleAxisf(angle_vec[1],  Eigen::Vector3f::UnitY())
        * Eigen::AngleAxisf(angle_vec[2], Eigen::Vector3f::UnitZ());
      //rot = Eigen::AngleAxisf(max_angle_axis * M_PI, axis);
      poses[i] = Eigen::Matrix4f::Identity();
      poses[i].block<3,3>(0,0) = rot;

      //poses[i](0,3) = dist_trans(e2);
      //poses[i](1,3) = dist_trans(e2);
      //poses[i](2,3) = dist_trans(e2);
    } else {
      poses[i] = Eigen::Matrix4f::Identity();
    }
    //rot = Eigen::AngleAxisf(dist(e2)*M_PI, axis );


    std::cout<<"random pose "<<i<<" is \n"<<poses[i]<<"\n";
  }
}


float eval_poses(std::vector< Eigen::Matrix4f,
                    Eigen::aligned_allocator<Eigen::Matrix4f>>& estimates,
                std::vector<  Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> & gt
                ){
  //assert(fs::exists(fname));
  //std::ofstream err_f(fname,std::fstream::out |   std::ios::app);
  float total_err = 0;  
  for (int i = 0; i < estimates.size(); i++) {

    float err = (estimates[i].inverse() * gt[i]).log().norm();
    total_err += err;
  }
  return total_err;
  //err_f << total_err<<"\n";
  //err_f.close();
  //std::cout<<"Total: "<<counter<<" success out of "<<gt.size()<<"\n";
}



int main(int argc, char *argv[]) {
  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  //cvo::KittiHandler kitti(argv[1], 0);
  std::string source_file(argv[1]);
  //std::string target_file(argv[2]);
  string cvo_param_file(argv[2]);
  float ell = -1;
  //if (argc > 4)
  ell = std::stof(argv[3]);
  int is_using_irls = std::stoi(argv[4]);
  int num_runs = std::stoi(argv[5]);
  cvo::CvoGPU cvo_align(cvo_param_file );
  cvo::CvoParams & init_param = cvo_align.get_params();
  //init_param.ell_init = dist; //init_param.ell_init_first_frame;

  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_pcd(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::io::loadPCDFile(source_file, *source_pcd);


  cvo::VoxelMap<pcl::PointXYZRGB> voxel_map(cvo_align.get_params().multiframe_downsample_voxel_size);
  for (int l = 0; l < source_pcd->size(); l++) {
    voxel_map.insert_point(&source_pcd->at(l));
  }
  std::vector<pcl::PointXYZRGB*> sampled =  voxel_map.sample_points();
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_pcd_downsampled(new pcl::PointCloud<pcl::PointXYZRGB>);
  for (auto p : sampled)
    source_pcd_downsampled->push_back(*p);
  std::shared_ptr<cvo::CvoPointCloud> source(new cvo::CvoPointCloud(*source_pcd_downsampled));  


  //if (argc > 4)
  init_param.ell_init = ell;
  init_param.ell_decay_rate = init_param.ell_decay_rate_first_frame;
  init_param.ell_decay_start  = init_param.ell_decay_start_first_frame;
  cvo_align.write_params(&init_param);

  std::cout<<"write ell! ell init is "<<cvo_align.get_params().ell_init<<std::endl;

  std::vector< Eigen::Matrix4f,
               Eigen::aligned_allocator<Eigen::Matrix4f>> gt_all,
    grad_all, irls_all;

  std::vector<double> grad_time, irls_time;
  std::vector<int> grad_iters, irls_iters;
    
  gen_random_poses(gt_all, num_runs, 30 );
  std::vector<Sophus::SE3f> poses_gt;
  std::transform(gt_all.begin(), gt_all.end(), std::back_inserter(poses_gt),
                 [&](const Eigen::Matrix4f & in){
                   Sophus::SE3f pose(in);
                   return pose.inverse();
                 });
    
  for (int k = 0; k < num_runs; k++) {

    // generate gt
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_pcd_transformed(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud (*source_pcd_downsampled, *source_pcd_transformed, gt_all[k].inverse());
    std::shared_ptr<cvo::CvoPointCloud> target(new cvo::CvoPointCloud(*source_pcd_transformed));
  
    Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();  // from source frame to the target frame

    Eigen::Matrix4f result, init_guess_inv;
    Eigen::Matrix4f identity_init = Eigen::Matrix4f::Identity();
    init_guess_inv = init_guess.inverse();    

    printf("Start align... num_fixed is %d, num_moving is %d\n", source->num_points(), target->num_points());
    std::cout<<std::flush;

    double this_time = 0;
    int num_iters = 0;

    
    result = construct_loop_BA_problem(cvo_align,
                                       source, target,
                                       init_guess,
                                       this_time,
                                       num_iters);
    irls_all.push_back(result);
    irls_time.push_back(this_time);
    irls_iters.push_back(num_iters);

    
    //cvo_align.align(*source, *target, init_guess_inv, result, nullptr,&this_time, &num_iters);
    //grad_all.push_back(result);
    //grad_time.push_back(this_time);
    //grad_iters.push_back(num_iters);
  }

  //float err_grad = eval_poses(grad_all,
  //                             gt_all);
   
  float err_irls = eval_poses(irls_all,
                              gt_all);

  //std::cout<<"err grad is "<<err_grad<<" with time "<<std::accumulate(grad_time.begin(), grad_time.end(), 0.0) / num_runs<<" and iterations is "<<std::accumulate(grad_iters.begin(), grad_iters.end(), 0.0)<<"\n";
  std::cout<<"err irls is "<<err_irls<<" with time "<<std::accumulate(irls_time.begin(), irls_time.end(), 0.0) / num_runs<<" and iterations is "<<std::accumulate(irls_iters.begin(), irls_iters.end(), 0.0)<<"\n";
   

  
  //cvo_align.align(*source, *target, init_guess, result);
    /*  
  std::cout<<"Transform is "<<result <<"\n\n";
  pcl::PointCloud<pcl::PointXYZRGB> pcd_old, pcd_new;
  cvo::CvoPointCloud new_pc(3, 19), old_pc(3, 19);
  cvo::CvoPointCloud::transform(init_guess, * target, old_pc);
  cvo::CvoPointCloud::transform(result, *target, new_pc);
  std::cout<<"Just finished transform\n";
  cvo::CvoPointCloud sum_old = old_pc + *source;
  cvo::CvoPointCloud sum_new = new_pc  + *source ;
  std::cout<<"Just finished CvoPointCloud concatenation\n";
  std::cout<<"num of points before and after alignment is "<<sum_old.num_points()<<", "<<sum_new.num_points()<<"\n";
  sum_old.export_to_pcd(pcd_old);
  sum_new.export_to_pcd(pcd_new);
  std::cout<<"Just export to pcd\n";
  std::string fname("before_align.pcd");
  pcl::io::savePCDFileASCII(fname, pcd_old);
  fname= "after_align.pcd";
  pcl::io::savePCDFileASCII(fname, pcd_new);
  // append accum_tf_list for future initialization
  std::cout<<"Average registration time is "<<this_time<<std::endl;
    */

  return 0;
}
