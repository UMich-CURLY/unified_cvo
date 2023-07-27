#include <algorithm>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
#include "dataset_handler/KittiHandler.hpp"
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include "utils/ImageStereo.hpp"
#include "utils/Calibration.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoGPU.hpp"
#include "cvo/CvoParams.hpp"
#include "utils/VoxelMap.hpp"
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <random>
using namespace std;
using namespace boost::filesystem;


struct normal_random_variable
{
  normal_random_variable(Eigen::Matrix3f const& covar)
    : normal_random_variable(Eigen::Vector3f::Zero(covar.rows()), covar)
  {}

  normal_random_variable(Eigen::Vector3f const& mean, Eigen::Matrix3f const& covar)
    : mean(mean)
  {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(covar);
    transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
  }

  Eigen::Vector3f mean;
  Eigen::Matrix3f transform;

  Eigen::Vector3f operator()() const
  {
    static std::mt19937 gen{ std::random_device{}() };
    static std::normal_distribution<> dist;
    Eigen::Vector3f v;
    return mean + transform * v.unaryExpr([&](auto x) { return static_cast<float>( dist(gen) ); });
  }
};

pcl::visualization::PCLVisualizer::Ptr rgbNormalVis(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud) {
  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::PointXYZRGBNormal> ne;
  ne.setInputCloud (cloud);

  // Create an empty kdtree representation, and pass it to the normal estimation object.
  // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
  ne.setSearchMethod (tree);

  // Output datasets
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_normals (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

  // Use all neighbors in a sphere of radius 3cm
  ne.setRadiusSearch (2);

  // Compute the features
  ne.compute (*cloud_normals);
  

  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (255, 255, 255);

  // sample points
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr samples(new pcl::PointCloud<pcl::PointXYZRGB>);
  for (int k = 0; k < cloud_normals->size(); k++) {

    auto p_center = (*cloud_normals)[k];
    auto p_xyz = (*cloud)[k];
    Eigen::Matrix3f cov;
    Eigen::Vector3f z(p_center.normal[0], p_center.normal[1], p_center.normal[2]);  //p_center.getVector3fMap().normalized();
    Eigen::Vector3f x = p_xyz.getVector3fMap();
    //x(2) = (-1)/z(2)*(z(0) + z(1));
    Eigen::Vector3f y = z.cross(x);
    x = y.cross(z);
      
    Eigen::Matrix3f U;
    U.col(0) = x.normalized();
    U.col(1) = y.normalized();
    U.col(2) = z.normalized();

    Eigen::Matrix3f eig = Eigen::Matrix3f::Identity();
    eig(0,0) = 1;
    eig(1,1) = 1;
    eig(2,2) = 0.05;
    cov = U * eig * U.transpose();    

    // sample 50 points
    normal_random_variable sample (p_xyz.getVector3fMap(), cov);
    for (int i = 0; i < 200; i++) {
      Eigen::Vector3f p_sampled = sample();

      pcl::PointXYZRGB p_sampled_pcl;// = p_center;
      p_sampled_pcl.getArray3fMap() = p_sampled;
      p_sampled_pcl.r = p_xyz.r;
      p_sampled_pcl.g = p_xyz.g;
      p_sampled_pcl.b = p_xyz.b;
      samples->push_back(p_sampled_pcl);
      std::cout<<"sample "<<p_sampled_pcl.getArray3fMap().transpose()<<" around "<<p_xyz.getVector3fMap()<<"\n";      
    }
    std::cout<<"\n";
  }

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "raw cloud");
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_sampled(samples);
  viewer->addPointCloud<pcl::PointXYZRGB> (samples, rgb_sampled, "sampled depth cloud");
  
  
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "raw cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sampled depth cloud");
  
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
  

}

pcl::visualization::PCLVisualizer::Ptr rgbVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  std::cout<<"start view...\n";
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  std::cout<<"set viewer background\n";
  // sample points
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr samples(new pcl::PointCloud<pcl::PointXYZRGB>);
  for (auto p_center: *cloud) {


    Eigen::Matrix3f cov;
      Eigen::Vector3f z = p_center.getVector3fMap().normalized();
      Eigen::Vector3f x(1,1,1);
      x(2) = (-1)/z(2)*(z(0) + z(1));
      Eigen::Vector3f y = z.cross(x);

      Eigen::Matrix3f U;
      U.col(0) = x;
      U.col(1) = y;
      U.col(2) = z;

      Eigen::Matrix3f eig = Eigen::Matrix3f::Identity();
      eig(0,0) = 0.005;
      eig(1,1) = 0.005;
      eig(2,2) = 0.4;

      cov = U * eig * U.transpose();    


    // sample 50 points
    normal_random_variable sample (p_center.getVector3fMap(), cov);
    for (int i = 0; i < 200; i++) {
      Eigen::Vector3f p_sampled = sample();

      pcl::PointXYZRGB p_sampled_pcl = p_center;
      p_sampled_pcl.getArray3fMap() = p_sampled;
      samples->push_back(p_sampled_pcl);
      std::cout<<"sample "<<p_sampled_pcl.getArray3fMap().transpose()<<" around "<<p_center.getVector3fMap()<<"\n";      
    }
    std::cout<<"\n";
  }

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "raw cloud");
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_sampled(samples);
  viewer->addPointCloud<pcl::PointXYZRGB> (samples, rgb_sampled, "sampled depth cloud");
  
  
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "raw cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sampled depth cloud");
  
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}

int main(int argc, char *argv[]) {
  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  cvo::KittiHandler kitti(argv[1], cvo::KittiHandler::DataType::STEREO);
  int total_iters = kitti.get_total_number();
  string calib_file;
  calib_file = string(argv[1] ) +"/cvo_calib.txt"; 
  cvo::Calibration calib(calib_file);
  int start_frame = std::stoi(argv[2]);
  kitti.set_start_index(start_frame);

  float leaf_size = std::stof(argv[3]);
  int directional_or_surface = std::stoi(argv[4]);  // 0: directional.  1: surface.

  Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();  // from source frame to the target frame
  //init_guess(2,3)=2.22;
  Eigen::Matrix4f accum_mat = Eigen::Matrix4f::Identity();

  cv::Mat source_left, source_right;
  //std::vector<float> semantics_source;
  //kitti.read_next_stereo(source_left, source_right, 19, semantics_source);
  kitti.read_next_stereo(source_left, source_right);
  std::cout<<"read source raw...\n";
  std::shared_ptr<cvo::ImageStereo> source_raw(new cvo::ImageStereo(source_left, source_right));
  std::cout<<"build source CvoPointCloud...\n";
  std::shared_ptr<cvo::CvoPointCloud> source(new cvo::CvoPointCloud(*source_raw, calib
                                                                    , cvo::CvoPointCloud::FULL
                                                                    ));

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>), cloud_downsampled(new pcl::PointCloud<pcl::PointXYZRGB>);
  source->export_to_pcd<pcl::PointXYZRGB>(*cloud);

  cvo::VoxelMap<pcl::PointXYZRGB> voxel(leaf_size);
  for (int k = 0; k < cloud->size(); k++) {
    voxel.insert_point(&cloud->points[k]);
  }

  std::vector<pcl::PointXYZRGB*> downsampled_results = voxel.sample_points();
  std::cout<<"downsampeld to "<<downsampled_results.size()<<" points from "<<cloud->size()<<" points\n";  
  for (auto p: downsampled_results) {
    cloud_downsampled->push_back(*p);
  }

  //pcl::visualization::PCLVisualizer::Ptr view_ptr;
  if (directional_or_surface) {
    auto view_ptr = rgbVis(cloud_downsampled);
    while (!view_ptr->wasStopped ())
    {
      view_ptr->spinOnce (100);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

  }
  else {
    auto view_ptr = rgbNormalVis(cloud_downsampled);
    while (!view_ptr->wasStopped ())
    {
      view_ptr->spinOnce (100);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    

  }
  return 0;
}
