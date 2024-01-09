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
namespace fs = boost::filesystem;

namespace cvo {


  

  void crop_point_cloud(const CvoPointCloud & pc_in,
                        CvoPointCloud & new_pc,
                        float crop_ratio) {
    int num_pts = pc_in.size();
    crop_ratio = std::min(crop_ratio, 1-crop_ratio);
    int num_pts_to_crop = (int)(num_pts * crop_ratio);

    new_pc.clear();
    std::vector<int> sorted_inds(num_pts);
    std::vector<float> x(num_pts);
    for (int i = 0; i < pc_in.size(); i++) {
      sorted_inds[i] = i;
      x[i] = pc_in.xyz_at(i)(0);
    }
    std::sort(sorted_inds.begin(), sorted_inds.end(), [&](int a, int b){ return x[a] < x[b];});
    for (int j = 0; j < 10; j++)std::cout<<sorted_inds[j]<<",";
    std::cout<<std::endl;
    
    int head_to_tail = rand() % 2;    
    if (head_to_tail == 0) {
      /// head to be cropped
      for (int j = num_pts_to_crop; j < num_pts; j++) {
        new_pc.push_back(pc_in.point_at(sorted_inds[j]));
      }
    } else {
      /// tail
      for (int j = 0; j < num_pts - num_pts_to_crop; j++) {
        new_pc.push_back(pc_in.point_at(sorted_inds[j]));
      }
    }
  }

  
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

}

Eigen::Vector3f get_pc_mean(const cvo::CvoPointCloud & pc) {
  Eigen::Vector3f p_mean_tmp = Eigen::Vector3f::Zero();
  for (int k = 0; k < pc.num_points(); k++)
//    p_mean_tmp = (p_mean_tmp + pc.positions()[k]).eval();
    p_mean_tmp = (p_mean_tmp + pc.at(k)).eval();
  p_mean_tmp = (p_mean_tmp) / pc.num_points();    
  return p_mean_tmp;
}


void write_transformed_pc_pair(const cvo::CvoPointCloud & pc1, const cvo::CvoPointCloud & pc2,
                               const Eigen::Matrix4f & T_frame_1_to_2,
                               const std::tuple<uint8_t, uint8_t, uint8_t> & color1,
                               const std::tuple<uint8_t, uint8_t, uint8_t> & color2,
                               std::string & fname    
                               ) {
  pcl::PointCloud<pcl::PointXYZRGB> pc_all;

  // pc1
  for (int j = 0; j <  pc1.size(); j++) {
    pcl::PointXYZRGB p;
    p.getVector3fMap() = pc1.at(j);
    p.r = std::get<0>(color1);
    p.g = std::get<1>(color1);
    p.b = std::get<2>(color1);
    pc_all.push_back(p);
  }

  /// pc2
  cvo::CvoPointCloud new_pc(0,0);
  Eigen::Matrix4f pose_f = T_frame_1_to_2.cast<float>();
  cvo::CvoPointCloud::transform(T_frame_1_to_2, pc2, new_pc);
  for (int j = 0; j <  pc2.size(); j++) {
    pcl::PointXYZRGB p;
    p.getVector3fMap() = new_pc.at(j);
    p.r = std::get<0>(color2);
    p.g = std::get<1>(color2);
    p.b = std::get<2>(color2);
    pc_all.push_back(p);
  }

  pcl::io::savePCDFileASCII(fname, pc_all);
}



template <typename PointT>
void add_gaussian_mixture_noise(pcl::PointCloud<pcl::PointNormal> & input,
                                pcl::PointCloud<PointT> & output,
                                float ratio,
                                float sigma,
                                float uniform_range,
                                bool is_using_viewpoint) {

  cvo::GaussianMixtureDepthGenerator gaussion_mixture(ratio, sigma, uniform_range);

  output.resize(input.size());
  for (int i = 0; i < input.size(); i++) {
    if (is_using_viewpoint) {
      // using a far away view point
    } else {
      // using normal direction
      auto pt = input[i];
      Eigen::Vector3f normal_dir;
      normal_dir << pt.normal_x, pt.normal_y, pt.normal_z;
      Eigen::Vector3f center_pt = pt.getVector3fMap();
      Eigen::Vector3f result = gaussion_mixture.sample(center_pt, normal_dir);
      pcl::PointXYZ new_pt;
      output[i].getVector3fMap() = result;
      if (i == 0) {
        std::cout<<"transform "<<pt.getVector3fMap().transpose()<<" to "<<result.transpose()<<std::endl;
      }
    }
  }

  
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
template <typename T>
float pcd_rescale(typename pcl::PointCloud<T>::Ptr pcd, float scale=-1.0 ){
  Eigen::MatrixXf pcd_eigen(3, pcd->size());
  for (int j = 0; j < pcd->size(); j++) {
    pcd_eigen.col(j) = pcd->at(j).getVector3fMap();
  }

  if (scale < 0)
    scale = (pcd_eigen.rowwise().maxCoeff() - pcd_eigen.rowwise().minCoeff()).norm();
  std::cout << "scale = " << scale << std::endl;
  pcd_eigen /= scale;

  for (int j = 0; j < pcd->size(); j++)
    pcd->at(j).getVector3fMap() = pcd_eigen.col(j);
  return scale;
}


void gen_rand_colors(  std::vector<std::tuple<uint8_t, uint8_t, uint8_t> > & colors) {
  for (auto && color_curr_frame : colors) {
    std::get<0>(color_curr_frame) = (uint8_t)(rand() % 255);
    std::get<1>(color_curr_frame) = (uint8_t)(rand() % 255);
    std::get<2>(color_curr_frame) = (uint8_t)(rand() % 255);
  }
}

void eval_poses(std::vector<Sophus::SE3f> & estimates,
                std::vector<Sophus::SE3f> & gt,
                std::string & fname
                ){
  assert(filesystem::exists(fname));
  std::ofstream err_f(fname,std::fstream::out |   std::ios::app);
  float total_err = 0;
  for (int i = 0; i < gt.size(); i ++) {
    float err = (estimates[i].inverse() * gt[i]).log().norm();
    total_err += err;
  }
  err_f << total_err<<"\n";
  err_f.close();
  //std::cout<<"Total: "<<counter<<" success out of "<<gt.size()<<"\n";
}

void eval_poses(const Eigen::Matrix4f & estimates,
                const Eigen::Matrix4f & gt,
                std::string & fname
                ){
  assert(filesystem::exists(fname));
  std::ofstream err_f(fname,std::fstream::out |   std::ios::app);
  float total_err = 0;
  float err = (estimates.inverse() * gt).log().norm();
  total_err += err;
  err_f << total_err<<"\n";
  err_f.close();
  //std::cout<<"Total: "<<counter<<" success out of "<<gt.size()<<"\n";
}


int main(int argc, char** argv) {

   srand(time(NULL));
  omp_set_num_threads(24);
  std::string in_pcd_fname(argv[1]);
  std::string cvo_param_file(argv[2]);
  float max_trans = std::stof(argv[3]);
  float max_angle_per_axis = std::stof(argv[4]);
  int num_runs = std::stoi(argv[5]);
  std::string exp_folder(argv[6]);
  int is_adding_outliers = std::stoi(argv[7]);
  float ratio = 0.8;
  float sigma = 0.01;
  float uniform_range = 0.1;
  //if (is_adding_outliers) {
  ratio = std::stof(argv[8]);
  sigma = std::stof(argv[9]);
  uniform_range = std::stof(argv[10]);

  float normal_radius = std::stof(argv[11]);
  float fpfh_radius = std::stof(argv[12]);

  float crop_ratio = std::stof(argv[13]);
    //}
  int num_frames = 2;

  /// assign one color to one frame
  std::vector<std::tuple<uint8_t, uint8_t, uint8_t> > colors;
  colors.push_back({255,0,0});
  colors.push_back({0,0,255});
  ///gen_rand_colors(colors);

  
  cvo::CvoGPU cvo_align(cvo_param_file);
  const Eigen::Matrix4f pose_idd = Eigen::Matrix4f::Identity();

  pcl::PointCloud<pcl::PointXYZ>::Ptr raw_pcd(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointNormal>::Ptr raw_pcd_normal(new pcl::PointCloud<pcl::PointNormal>);
  pcl::io::loadPCDFile<pcl::PointNormal> (in_pcd_fname, *raw_pcd_normal);

  fs::path exp_folder_dir(exp_folder);
  fs::create_directories(exp_folder_dir);
  std::string fname(exp_folder + "/cvo_err_global.txt");
  std::ofstream err_f(fname);
  err_f.close();
  float scale = -1.0;
  for (int k = 0; k < num_runs; k++) {

    /// create exp log folder
    fs::path exp_curr_dir = exp_folder_dir / fs::path(std::to_string(k));
    std::string exp_curr_dir_str = exp_curr_dir.string();
    std::cout<<"Current exp folder is "<<exp_curr_dir_str<<"\n";
    fs::create_directories(exp_curr_dir);

    /// gen init poses for each frame and log them as gt pose
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> tracking_poses;
    gen_random_poses(tracking_poses, num_frames, max_angle_per_axis );
    std::ofstream pose_f(exp_curr_dir_str + "/gt_poses.txt");
    for (int l = 0; l < tracking_poses.size(); l++ ) {
      std::string pose_curr = cvo::mat_to_line<float, Eigen::ColMajor>(tracking_poses[l]);
      pose_f << pose_curr<<std::endl;
    }
    pose_f.close();


    std::vector<Sophus::SE3f> poses_gt;
    std::transform(tracking_poses.begin(), tracking_poses.end(), std::back_inserter(poses_gt),
                   [&](const Eigen::Matrix4f & in){
                     Sophus::SE3f pose(in);
                     return pose.inverse();
                   });

    /// read point cloud
    //std::vector<cvo::CvoFrame::Ptr> frames;
    std::vector<std::shared_ptr<cvo::CvoPointCloud>> pcs;
    for (int i = 0; i<num_frames; i++) {
      // Create the filtering object
      pcl::PointCloud<pcl::PointNormal>::Ptr raw_pcd_curr;//(new pcl::PointCloud<pcl::PointXYZ>);
      if (is_adding_outliers) {
        raw_pcd_curr.reset(new pcl::PointCloud<pcl::PointNormal>);
        add_gaussian_mixture_noise(*raw_pcd_normal,
                                   *raw_pcd_curr,
                                   ratio,
                                   sigma,
                                   uniform_range,
                                   false
                                   );
      } else {
        raw_pcd_curr = raw_pcd_normal;
      }

      // for visualzation
      copyPointCloud(*raw_pcd_curr, *raw_pcd);
      pcl::io::savePCDFileASCII(exp_curr_dir_str + "/" + std::to_string(i)+"normal.pcd", *raw_pcd);      
      scale = pcd_rescale<pcl::PointXYZ>(raw_pcd, scale);
      
      pcl::PointCloud<pcl::PointXYZ>::Ptr raw_pcd_transformed(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::transformPointCloud (*raw_pcd, *raw_pcd_transformed, tracking_poses[i]);


      /// downsample
      std::cout<<"Before downsample, num of points of frame "<<i<<" is "<<raw_pcd->size()<<"\n";
      cvo::VoxelMap<pcl::PointXYZ> voxel_map(cvo_align.get_params().multiframe_downsample_voxel_size);
      for (int l = 0; l < raw_pcd_transformed->size(); l++) {
        voxel_map.insert_point(&raw_pcd_transformed->at(l));
      }
      std::vector<pcl::PointXYZ*> sampled =  voxel_map.sample_points();
      pcl::PointCloud<pcl::PointXYZ>::Ptr raw_pcd_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
      for (auto p : sampled)
        raw_pcd_downsampled->push_back(*p);
      std::cout<<"after downsampling, num of points is "<<raw_pcd_transformed->size()<<std::endl;
      pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh = cvo::calculate_fpfh_pcl<pcl::PointXYZ>(raw_pcd_downsampled, 0.03, 0.05);
      
      std::shared_ptr<cvo::CvoPointCloud> pc (new cvo::CvoPointCloud(*raw_pcd_downsampled));
      pc->add_semantics(33);
      for (int k = 0 ; k < pc->size(); k++)  {
        
        memcpy((*pc)[k].label_distribution, (*fpfh)[k].histogram, sizeof(float)*33  );
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>> label_map((*pc)[k].label_distribution, 33);
        label_map.normalize();
        /*
        if (k == 0) {
        for (int l = 0; l < 33; l++)
          std::cout<<(*fpfh)[k].histogram[l]<<",";
        std::cout<<"\n";
        }*/
      }

      if (crop_ratio > 0.001) {
        std::shared_ptr<cvo::CvoPointCloud> pc_cropped(new cvo::CvoPointCloud(pc->num_features(), pc->num_classes() ));
        cvo::crop_point_cloud(*pc, *pc_cropped, crop_ratio);

        pc_cropped.swap(pc);
        std::cout<<"crop from "<<pc_cropped->size()<<" to "<<pc->size()<<"\n";        
      }

      std::string pcd_name(exp_curr_dir_str + "/" + std::to_string(0)+".pcd");
      pc->write_to_pcd(pcd_name);
      
      pcs.push_back(pc);
      
      /// construct frame
      //cvo::Mat34d_row pose;
      //Eigen::Matrix<double, 4,4, Eigen::RowMajor> pose44 = Eigen::Matrix<double, 4,4, Eigen::RowMajor>::Identity();
      //pose = pose44.block<3,4>(0,0);
      //cvo::CvoFrame::Ptr new_frame(new cvo::CvoFrameGPU(pc.get(), pose.data()));
      //frames.push_back(new_frame);

    }


    float ip = cvo_align.function_angle(*(pcs[0]), *(pcs[1]), poses_gt[1].inverse().matrix(), cvo_align.get_params().ell_init_first_frame);
    std::cout<<"At gt, the inner product is "<<ip<<"\n";


    //std::vector<cvo::Mat34d_row, Eigen::aligned_allocator<cvo::Mat34d_row>> tracking_poses;
    std::cout<<"write to before_BA.pcd\n";
    std::string f_name(exp_curr_dir_str + "/before_BA_");
    f_name += std::to_string(k)+".pcd";
    //write_transformed_pc(frames, f_name, colors);
    write_transformed_pc_pair(*(pcs[0]), *(pcs[1]),
                              pose_idd,
                              colors[0], colors[1],
                              f_name);
    

    
    std::cout<<"write to gt\n";
    /*
    for (int i = 0; i < num_frames; i++ ) {
      cvo::Mat34d_row pose;
      Eigen::Matrix<double, 4,4, Eigen::RowMajor> pose44 = tracking_poses[i].cast<double>().inverse();
      memcpy(pcs[i]->pose_vec, pose44.data(), sizeof(double) * 12);
    }
    */
    f_name =  exp_curr_dir_str + "/gt_BA_"+std::to_string(k)+".pcd" ;
    //write_transformed_pc(frames, f_name, colors);
    write_transformed_pc_pair(*(pcs[0]), *(pcs[1]),
                              poses_gt[1].matrix(),
                              colors[0], colors[1],
                              f_name);
    

    /*
    for (int i = 0; i < num_frames; i++ ) {
      cvo::Mat34d_row pose;
      Eigen::Matrix<double, 4,4, Eigen::RowMajor> pose44 = Eigen::Matrix<double, 4,4, Eigen::RowMajor>::Identity();
      memcpy(frames[i]->pose_vec, pose44.data(), sizeof(double) * 12);
    }
    */

    /// start registration
    std::cout<<"Centerizing\n";
    Eigen::Vector3f mean1, mean2;
    cvo::CvoPointCloud pc1_center(*pcs[0]),pc2_center(*pcs[1]);
    mean1 = get_pc_mean(*pcs[0]);
    for (int j = 0 ; j < pcs[0]->size(); j++) 
      pc1_center[j].getVector3fMap() = (pcs[0]->at(j) - mean1 ).eval();
    mean2 = get_pc_mean(*pcs[1]);
    for (int j = 0 ; j < pcs[1]->size(); j++) 
      pc2_center[j].getVector3fMap() = (pcs[1]->at(j) - mean2 ).eval();
    f_name =  exp_curr_dir_str + "/before_BA_centered_"+std::to_string(k)+".pcd" ;
    write_transformed_pc_pair(pc1_center, pc2_center,
                              pose_idd, //poses_gt[1].matrix(),
                              colors[0], colors[1],
                              f_name);

    
    std::cout<<"start align\n";
    Eigen::Matrix4f init_inv = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f result;
    double time = 0;
    cvo_align.align(pc1_center, pc2_center, init_inv,
                    result,  nullptr, &time);

    f_name=exp_curr_dir_str + "/after_BA_bunny_" + std::to_string(k)+"_centered.pcd";
    write_transformed_pc_pair(pc1_center, pc2_center,
                              result,
                              colors[0], colors[1],
                              f_name);

    
    //Sophus::SE3f result_sophus(result.block<3,3>(0,0), result.block<3,1>(0,0));
    Eigen::Vector3f t_actual = (result.block<3,1>(0,3) - result.block<3,3>(0,0) * mean2 + mean1).eval();
    result.block<3,1>(0,3) = t_actual;    

    std::cout<<"Align ends. Total time is "<<time<<std::endl<<"result is "<<result<<"\n";
    f_name=exp_curr_dir_str + "/after_BA_bunny_" + std::to_string(k)+".pcd";
    write_transformed_pc_pair(*(pcs[0]), *(pcs[1]),
                              result,
                              colors[0], colors[1],
                              f_name);

    eval_poses(result,
               poses_gt[1].matrix(),
               fname);


  }
  return 0;
}
