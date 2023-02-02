#include <Eigen/Geometry>
#include <Eigen/src/Core/util/Constants.h>
#include <iostream>
#include <list>
#include <cmath>
#include <fstream>
#include <filesystem>
#include "utils/def_assert.hpp"
#include "utils/GassianMixture.hpp"
#include "utils/eigen_utils.hpp"
#include <pcl/common/io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/impl/point_types.hpp>
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
#include <pcl/io/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <filesystem>
namespace fs = std::filesystem;

void write_transformed_pc(std::vector<cvo::CvoFrame::Ptr> & frames, std::string & fname,
                          std::vector<std::tuple<uint8_t, uint8_t, uint8_t> > & colors) {
  pcl::PointCloud<cvo::CvoPoint> pc_all;
  for (int i = 0; i <  frames.size(); i++) {
    auto ptr = frames[i];
    cvo::CvoPointCloud new_pc(0,0);
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3,4>(0,0) = Eigen::Map<cvo::Mat34d_row>(ptr->pose_vec);
    
    Eigen::Matrix4f pose_f = pose.cast<float>();
    cvo::CvoPointCloud::transform(pose_f, *ptr->points, new_pc);

    pcl::PointCloud<cvo::CvoPoint> pc_curr_xyz;
    new_pc.export_to_pcd(pc_curr_xyz);
    pcl::PointCloud<cvo::CvoPoint> pc_curr;
    for (int j = 0; j <  pc_curr_xyz.size(); j++) {
      cvo::CvoPoint p;
      p.getVector3fMap() = pc_curr_xyz[j].getVector3fMap();
      p.r = std::get<0>(colors[i]);
      p.g = std::get<1>(colors[i]);
      p.b = std::get<2>(colors[i]);
      pc_curr.push_back(p);
    }

    pc_all += pc_curr;
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
                      float max_angle_axis // In degrees
                      ) {
  poses.resize(num_poses);
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<> dist(0, max_angle_axis);
  std::uniform_real_distribution<> dist_trans(0,0.3);
  for (int i = 0; i < num_poses; i++) {
    if (i != 0) {
      Eigen::Matrix3f rot;
      Eigen::Vector3f axis;
      axis << (float)dist_trans(e2), (float)dist_trans(e2), (float)dist_trans(e2);
      axis = axis.normalized().eval();
      
      //rot = Eigen::AngleAxisf(dist(e2)*M_PI, Eigen::Vector3f::UnitX())
      //  * Eigen::AngleAxisf(dist(e2)*M_PI,  Eigen::Vector3f::UnitY())
      //  * Eigen::AngleAxisf(dist(e2)*M_PI, Eigen::Vector3f::UnitZ());
      rot = Eigen::AngleAxisf(max_angle_axis * 0.017453 , axis);
      poses[i] = Eigen::Matrix4f::Identity();
      poses[i].block<3,3>(0,0) = rot;
      poses[i](0,3) = dist_trans(e2);
      poses[i](1,3) = dist_trans(e2);
      poses[i](2,3) = dist_trans(e2);
    } else {
      poses[i] = Eigen::Matrix4f::Identity();
    }
    //rot = Eigen::AngleAxisf(dist(e2)*M_PI, axis );


    std::cout<<"random pose "<<i<<" is \n"<<poses[i]<<"\n";
  }
}
template <typename T>
void pcd_rescale(typename pcl::PointCloud<T>::Ptr pcd ){
  Eigen::MatrixXf pcd_eigen(3, pcd->size());
  for (int j = 0; j < pcd->size(); j++) {
    pcd_eigen.col(j) = pcd->at(j).getVector3fMap();
  }

  float scale = (pcd_eigen.rowwise().maxCoeff() - pcd_eigen.rowwise().minCoeff()).norm();
  std::cout << "scale = " << scale << std::endl;
  pcd_eigen /= scale;

  for (int j = 0; j < pcd->size(); j++)
    pcd->at(j).getVector3fMap() = pcd_eigen.col(j);
}


void gen_rand_colors(  std::vector<std::tuple<uint8_t, uint8_t, uint8_t> > & colors) {
  for (auto && color_curr_frame : colors) {
    std::get<0>(color_curr_frame) = (uint8_t)(rand() % 255);
    std::get<1>(color_curr_frame) = (uint8_t)(rand() % 255);
    std::get<2>(color_curr_frame) = (uint8_t)(rand() % 255);
  }
}

void eval_poses_se3_norm(std::vector<Sophus::SE3f> & estimates,
                         std::vector<Sophus::SE3f> & gt,
                         std::string & fname
                         ){
  assert(std::filesystem::exists(fname));
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


void mat_to_row(const Sophus::SE3f & mat,
                std::vector<float> & row) {
  row.resize(16);
  Eigen::Matrix4f m = mat.matrix();
  for (int i = 0; i < 16; i++) {
    row[i] = m(i / 4, i % 4);
  }
  
}

void eval_poses_SE3_frobenius_norm(std::vector<Sophus::SE3f> & estimates,
                                   std::vector<Sophus::SE3f> & gt,
                                   std::string & err_fname,
                                   std::string & err_framewise_fname,
                                   std::string & pose_fname
                                   ){
  // assert(std::filesystem::exists(fname));
  std::ofstream err_f(err_fname,std::fstream::out |   std::ios::app);
  std::ofstream err_framewise(err_framewise_fname);
  std::ofstream pose_f(pose_fname);
  float total_err = 0;
  for (int i = 0; i < gt.size(); i ++) {
    /// element wise norm 
    float err = ((estimates[i].inverse() * gt[i]).matrix() - Eigen::Matrix4f::Identity()).norm();
    err_framewise << err<<"\n";

    /// log poses
    std::vector<float> mat_row_major;
    mat_to_row(estimates[i], mat_row_major);
    for (int j = 0; j < mat_row_major.size(); j++ )
      pose_f << mat_row_major[j]<<" ";
    pose_f << "\n";

    total_err += err;
  }
  err_f << total_err<<"\n";
  err_f.close();
  err_framewise.close();
  pose_f.close();
  //std::cout<<"Total: "<<counter<<" success out of "<<gt.size()<<"\n";
}


int main(int argc, char** argv) {

  omp_set_num_threads(24);
  std::string in_pcd_fname(argv[1]);
  std::string cvo_param_file(argv[2]);
  int num_frames = std::stoi(argv[3]);
  float max_angle_per_axis = std::stof(argv[4]);
  int num_runs = std::stoi(argv[5]);
  std::string exp_folder(argv[6]);
  int is_adding_outliers = std::stoi(argv[7]);
  float ratio = 0.8;
  float sigma = 0.01;
  float uniform_range = 0.5;
  cvo::CvoGPU cvo_align(cvo_param_file);
  if (is_adding_outliers) {
    ratio = std::stof(argv[8]); /// affects the outlier ratio
    sigma = std::stof(argv[9]); /// affects the noise scale value size
    uniform_range = std::stof(argv[10]); /// affets
  }
//   int is_logging_data_only = std::stoi(argv[11]);

//   std::vector<std::tuple<uint8_t, uint8_t, uint8_t> > colors(num_frames);
//   gen_rand_colors(colors);

//   cvo::CvoGPU cvo_align(cvo_param_file);

  /// load input point cloud (transfomed from prev step)
  pcl::PointCloud<cvo::CvoPoint>::Ptr raw_pcd(new pcl::PointCloud<cvo::CvoPoint>);
  // Add function to load 4 different point cloud 
  std::vector<pcl::PointCloud<cvo::CvoPoint>::Ptr> raw_pcd_normal;


  /// experiment result folder
  fs::path exp_folder_dir(exp_folder);
  fs::create_directories(exp_folder_dir);
  std::string full_error_fname(exp_folder + "/cvo_err_bunny_semantics.txt");
  std::ofstream err_f(full_error_fname);
  err_f.close();
  std::string full_time_fname(exp_folder + "/cvo_time_bunny_semantics.txt");
  std::ofstream err_time_f(full_time_fname);
  err_time_f.close();

  /// start running 
  for (int k = 0; k < num_runs; k++) {


    // /// convert to Sophus format for ground truth
    // std::vector<Sophus::SE3f> poses_gt;
    // std::transform(tracking_poses.begin(), tracking_poses.end(), std::back_inserter(poses_gt),
    //                [&](const Eigen::Matrix4f & in){
    //                  Sophus::SE3f pose(in);
    //                  return pose.inverse();
    //                });


    /// generate point cloud
    std::vector<cvo::CvoFrame::Ptr> frames;
    std::vector<std::shared_ptr<cvo::CvoPointCloud>> pcs;
    std::string foldername = in_pcd_fname + '/' +std::to_string(k) + '/';
    for (int i = 0; i < num_frames; i++) {
        pcl::PointCloud<cvo::CvoPoint>::Ptr dst(new pcl::PointCloud<cvo::CvoPoint>);
        std::string fullName = foldername  + std::to_string(i) + "normal.pcd";
        std::cout << "load point cloud " << fullName << std::endl;
        pcl::io::loadPCDFile (fullName, *dst);
        pcl::PointCloud<cvo::CvoPoint>::Ptr raw_pcd(new pcl::PointCloud<cvo::CvoPoint>);
        raw_pcd = dst;
        std::shared_ptr<cvo::CvoPointCloud> pc (new cvo::CvoPointCloud(*raw_pcd));  
        cvo::Mat34d_row pose;
        cvo::CvoFrame::Ptr new_frame(new cvo::CvoFrameGPU(pc.get(), pose.data()));
        frames.push_back(new_frame);
        pcs.push_back(pc);

        // covert point cloud 
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr result(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (auto pt:dst->points){
          pcl::PointXYZRGB newpt;
          newpt.x = pt.x;
          newpt.y = pt.y;
          newpt.z = pt.z;
          newpt.r = pt.features[0] * 255;
          newpt.g = pt.features[1] * 255;
          newpt.b = pt.features[2] * 255;
          result->push_back(newpt);
        }
        std::cout <<"add new point" << result->size() << std::endl;
        pcl::io::savePCDFileASCII(foldername + std::to_string(i)+"normal_color.pcd", *result);  
    }
  
  }
  return 0;
}
