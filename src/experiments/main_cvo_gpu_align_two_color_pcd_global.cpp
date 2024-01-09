#include <algorithm>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <chrono>
#include <boost/filesystem.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>

///#include "utils/pcl_utils.hpp"
//#include "dataset_handler/KittiHandler.hpp"
#include "utils/ImageStereo.hpp"
#include "utils/Calibration.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoGPU.hpp"
#include "cvo/CvoParams.hpp"
#include "cvo/IRLS_State_CPU.hpp"
#include "cvo/IRLS_State.hpp"
#include "utils/VoxelMap.hpp"


using namespace std;
using namespace boost::filesystem;

extern template class cvo::VoxelMap<pcl::PointXYZRGB>;
extern template class cvo::Voxel<pcl::PointXYZRGB>;



namespace cvo {

  
  template <typename PointT>
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr calculate_fpfh_pcl(typename pcl::PointCloud< PointT>::Ptr cloud,
                                                                float normal_radius,
                                                                float fpfh_radius){

    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal> ());

    // Create the normal estimation class, and pass the input dataset to it
    typename pcl::NormalEstimation<PointT, pcl::Normal> ne;
    ne.setInputCloud (cloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    typename pcl::search::KdTree<PointT>::Ptr tree_normal (new typename pcl::search::KdTree<PointT> ());
    ne.setSearchMethod (tree_normal);

    // Output datasets
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

    // Use all neighbors in a sphere of radius 3cm
    ne.setRadiusSearch (normal_radius);

    // Compute the features
    ne.compute (*cloud_normals);


    // Create the FPFH estimation class, and pass the input dataset+normals to it

    typename  pcl::FPFHEstimation<PointT, pcl::Normal, pcl::FPFHSignature33> fpfh;

    fpfh.setInputCloud (cloud);
    fpfh.setInputNormals (cloud_normals);

    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    typename  pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    fpfh.setSearchMethod (tree);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs (new pcl::PointCloud<pcl::FPFHSignature33> ());
    // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!

    fpfh.setRadiusSearch (fpfh_radius);

    // Compute the features
    fpfh.compute (*fpfhs);

    return fpfhs;
  }


  template <typename PointT>
  std::shared_ptr<cvo::CvoPointCloud> load_pcd_and_fpfh(const std::string & source_file,
                                                        float radius_normal,
                                                        float radius_fpfh) {
    
    typename pcl::PointCloud<PointT>::Ptr source_pcd(new typename pcl::PointCloud<PointT>);
    pcl::io::loadPCDFile(source_file, *source_pcd);
    //std::cout<<"Read  source "<<source_pcd->size()<<" points\n";
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_source = calculate_fpfh_pcl<pcl::PointXYZRGB>(source_pcd, radius_normal, radius_fpfh);
    std::shared_ptr<cvo::CvoPointCloud> source(new cvo::CvoPointCloud(*source_pcd));
    for (int i = 0 ; i < source->size(); i++)  {
      memcpy((*source)[i].label_distribution, (*fpfh_source)[i].histogram, sizeof(float)*33  );
    }
    return source;
    
  }

}



float rand_rad() {
  float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  return r * M_PI * 2;
}

Eigen::Matrix4f gen_rand_pose(float max_translation) {
  Eigen::Matrix3f rot;
  
  rot = Eigen::AngleAxisf( rand_rad(), Eigen::Vector3f::UnitX())
    * Eigen::AngleAxisf( rand_rad(), Eigen::Vector3f::UnitY())
    * Eigen::AngleAxisf( rand_rad(), Eigen::Vector3f::UnitZ());

  Eigen::Vector3f t;
  t << static_cast <float> (rand()) / static_cast <float> (RAND_MAX),
    static_cast <float> (rand()) / static_cast <float> (RAND_MAX),
    static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  t = (t.normalized()).eval();
  t = (t*max_translation).eval();

  Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
  pose.block<3,3>(0,0) = rot;
  pose.block<3,1>(0,3) = t;
  return pose;
}

std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> gen_rand_init_pose(int discrete_rpy_num) {

  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> rot_all;
  rot_all.reserve(discrete_rpy_num * discrete_rpy_num * discrete_rpy_num);
  for (int i = 0; i < discrete_rpy_num; i++) {
    for ( int j = 0; j < discrete_rpy_num; j++) {
      for (int k = 0; k < discrete_rpy_num; k++) {

        Eigen::Matrix3f rot;
        
        rot = Eigen::AngleAxisf( i / (float)discrete_rpy_num * M_PI, Eigen::Vector3f::UnitZ())
          * Eigen::AngleAxisf( j / (float)discrete_rpy_num * M_PI, Eigen::Vector3f::UnitY())
          * Eigen::AngleAxisf( k / (float)discrete_rpy_num * M_PI, Eigen::Vector3f::UnitZ());
        rot_all.emplace_back(rot);
        //std::cout<<"push "<<rot<<"\n";
      }
    }
  }
  return rot_all;
}
 

Eigen::Vector3f get_pc_mean(const cvo::CvoPointCloud & pc) {
  Eigen::Vector3f p_mean_tmp = Eigen::Vector3f::Zero();
  for (int k = 0; k < pc.num_points(); k++)
//    p_mean_tmp = (p_mean_tmp + pc.positions()[k]).eval();
    p_mean_tmp = (p_mean_tmp + pc.at(k)).eval();
  p_mean_tmp = (p_mean_tmp) / pc.num_points();    
  return p_mean_tmp;
}

void save_two_cvo_pc(const Eigen::Matrix4f & tmp_init_guess,
                     std::shared_ptr<cvo::CvoPointCloud> source,
                     std::shared_ptr<cvo::CvoPointCloud> target,
                     std::string & name
                     ) {
  cvo::CvoPointCloud  old_pc(3, 19);
  cvo::CvoPointCloud::transform(tmp_init_guess, * target, old_pc);
  cvo::CvoPointCloud sum_old = old_pc + *source;
  pcl::PointCloud<pcl::PointXYZRGB> pcd_old;  
  sum_old.export_to_pcd(pcd_old);
  std::string fname = name; //("before_align")+std::to_string(ip)+".pcd";
  
  pcl::io::savePCDFileASCII(fname, pcd_old);
  
}


int main(int argc, char *argv[]) {
  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  //cvo::KittiHandler kitti(argv[1], 0);
  std::string source_file(argv[1]);
  std::string target_file(argv[2]);
  string cvo_param_file(argv[3]);
  int rot_discrete_num = std::stoi(argv[4]);
  srand ( time(NULL) );
  Eigen::Matrix4f idd = Eigen::Matrix4f::Identity();
  std::string name(                 "before_init_idd.pcd");  
  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> init_rots = gen_rand_init_pose(rot_discrete_num);
  std::cout<<"init rot size "<<init_rots.size()<<"\n"<<std::flush;
  //float ell = -1;
  //
  //	  ell = std::stof(argv[4]);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_pcd(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::io::loadPCDFile(source_file, *source_pcd);
  std::cout<<"Read  source "<<source_pcd->size()<<" points\n";
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_source = cvo::calculate_fpfh_pcl<pcl::PointXYZRGB>(source_pcd, 0.02, 0.03);
  std::shared_ptr<cvo::CvoPointCloud> source(new cvo::CvoPointCloud(*source_pcd));  pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_pcd(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::io::loadPCDFile(target_file, *target_pcd);
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_target = cvo::calculate_fpfh_pcl<pcl::PointXYZRGB>(target_pcd, 0.02, 0.03);  
  std::cout<<"Read  target "<<target_pcd->size()<<" points\n";
  std::shared_ptr<cvo::CvoPointCloud> target_tmp(new cvo::CvoPointCloud(*target_pcd));
  std::shared_ptr<cvo::CvoPointCloud> target(new cvo::CvoPointCloud());
  cvo::CvoPointCloud::transform(gen_rand_pose(1.0), *target_tmp, *target);
  
  name = "before_init_raw.pcd";
  save_two_cvo_pc(idd,
                  source,
                  target,
                  name 
                  );

  /// subtract mean
  Eigen::Vector3f source_mean = get_pc_mean(*source);
  for (int i = 0 ; i < source->size(); i++)  {
    (*source)[i].getVector3fMap() = ((*source)[i].getVector3fMap() - source_mean ).eval();
    memcpy((*source)[i].label_distribution, (*fpfh_source)[i].histogram, sizeof(float)*33  );
  }
  Eigen::Vector3f target_mean = get_pc_mean(*target);
  for (int i = 0 ; i < target->size(); i++)  {
    (*target)[i].getVector3fMap() = ((*target)[i].getVector3fMap() - target_mean ).eval();
    memcpy((*target)[i].label_distribution, (*fpfh_target)[i].histogram, sizeof(float)*33  );
  }
  float dist = (source_mean - target_mean).norm();
  std::cout<<"source mean is "<<source_mean<<", target mean is "<<target_mean<<", dist is "<<dist<<std::endl;


  //Eigen::Vector3f dist_vec = source_mean - target_mean;

  name = "before_init_idd_centered.pcd";
  save_two_cvo_pc(idd,
                  source,
                  target,
                  name
                  );

  
  cvo::CvoGPU cvo_align(cvo_param_file );
  cvo::CvoParams & init_param = cvo_align.get_params();
  //init_param.ell_init = dist; //init_param.ell_init_first_frame;
  //init_param.ell_init = init_param.ell_init_first_frame ;
  init_param.ell_decay_rate = init_param.ell_decay_rate_first_frame;
  init_param.ell_decay_start  = init_param.ell_decay_start_first_frame;
  init_param.MAX_ITER = 10000;
  
  cvo_align.write_params(&init_param);

  std::cout<<"write ell! ell init is "<<cvo_align.get_params().ell_init<<std::endl;

  auto start = std::chrono::system_clock::now();
  Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();  // from source frame to the target frame
  float curr_max_ip = 0;
  for (auto && init_rot : init_rots) {
    Eigen::Matrix4f tmp_init_guess =  Eigen::Matrix4f::Identity();
    tmp_init_guess.block<3,3>(0,0) = init_rot;

    Eigen::Matrix4f init_inv = tmp_init_guess.inverse();
    float ip = cvo_align.function_angle(*source, * target, init_inv, init_param.ell_init_first_frame);
    std::cout<<"Init guess\n"<<tmp_init_guess<<", ip is "<<ip<<"\n";
    name =  std::string("before_init_")+std::to_string(ip)+".pcd";
    save_two_cvo_pc(tmp_init_guess,
                    source,
                    target,
                    name);
    
    if (ip > curr_max_ip) {
      curr_max_ip = ip;
      init_guess = tmp_init_guess;
    }
  }
  auto end = std::chrono::system_clock::now();
  double elapsed =
    std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout<<"Chosen init guess is "<<init_guess<<", with inner product "<<curr_max_ip<<", the init search takes "<<elapsed<< "ms\n";


  Eigen::Matrix4f result, init_guess_inv;
  init_guess_inv = init_guess.inverse();    
  printf("Start align... num_fixed is %d, num_moving is %d\n", source->num_points(), target->num_points());
  std::cout<<std::flush;

  double this_time = 0;
  cvo_align.align(*source, *target, init_guess_inv, result, nullptr,&this_time);


  
  //cvo_align.align(*source, *target, init_guess, result);
    
  std::cout<<"Transform is "<<result <<"\n\n";
  cvo::CvoPointCloud new_pc(3, 19), old_pc(3, 19);
  cvo::CvoPointCloud::transform(init_guess, * target, old_pc);
  cvo::CvoPointCloud::transform(result, *target, new_pc);
  std::cout<<"Just finished transform\n";  
  cvo::CvoPointCloud sum_old = old_pc + *source;
  cvo::CvoPointCloud sum_new = new_pc  + *source ;
  std::cout<<"Just finished CvoPointCloud concatenation\n";
  std::cout<<"num of points before and after alignment is "<<sum_old.num_points()<<", "<<sum_new.num_points()<<"\n";
  pcl::PointCloud<pcl::PointXYZRGB> pcd_old, pcd_new;  
  sum_old.export_to_pcd(pcd_old);
  sum_new.export_to_pcd(pcd_new);
  std::cout<<"Just export to pcd\n";
  std::string fname("before_align.pcd");
  pcl::io::savePCDFileASCII(fname, pcd_old);
  fname= "after_align.pcd";
  pcl::io::savePCDFileASCII(fname, pcd_new);
  // append accum_tf_list for future initialization
  std::cout<<"Average registration time is "<<this_time<<std::endl;

  save_two_cvo_pc(result,
                  source,
                  target,
                  fname
                  );

  


  return 0;
}
