#include <iostream>
#include <vector>
#include <filesystem>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
#include <tbb/tbb.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include "utils/CvoPointCloud.hpp"
#include "mapping/bkioctomap.h"
#include "utils/PoseLoader.hpp"
#include <omp.h>
using namespace std;
using namespace boost::filesystem;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_row;

int main(int argc, char *argv[]) {

  omp_set_num_threads(12); // Use 4 threads for all consecutive parallel regions

  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  std::cout<<argc<<std::endl;
  int num_class= 0;

  std::string pcd_dir(argv[1]);
  std::string output_file(argv[2]);
  int start_frame = stoi(argv[3]);
  int num_frames = stoi(argv[4]);
  std::string pose_fname = std::string(argv[5]);
  
  vector<string> files;
  std::cout<<" cycle through the directory\n";
  int total_num = 0;
  /*
  for(auto & p : boost::filesystem::directory_iterator( p ) ) {
    // If it's not a directory, list it. If you want to list directories too, just remove this check.
    if (is_regular_file(p.path())) {
      // assign current file name to current_file and echo it out to the console.
      string current_file = p.path().string();
      files.push_back(string(argv[1]) + "/" + to_string(total_num) + ".pcd" );
      total_num += 1;
      cout <<"reading "<< current_file << endl; 
    }
    }*/
  std::cout<<"Mapping...\n";
  // Mapping
  // Set parameters
  int block_depth = 1;
  double sf2 = 1.0;
  double ell = 1.0;
  float prior = 0.0f;
  float var_thresh = 1.0f;
  double free_thresh = 0.65;
  double occupied_thresh = 0.9;
  double resolution = 0.05;
  double free_resolution = 1;
  double ds_resolution = -1;
  double max_range = -1;

  // Read camera poses
  // Eigen::MatrixXf camera_poses;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> poses;
  //std::string camera_pose_file = string(argv[1]) + "/" + "poses.txt";
  std::string camera_pose_file = pose_fname;//"poses.txt";
  //read_camera_poses(camera_pose_file, camera_poses);
  cvo::read_pose_file_kitti_format(pose_fname,
                                   start_frame,
                                   num_frames-start_frame-1,
                                   poses);
  
  std::cout<<"Just read poses\n";
  // Build map
  std::vector<cvo::CvoPointCloud> pc_vec(files.size(), cvo::CvoPointCloud(cvo::CvoPoint::FEATURE_DIMENSION, cvo::CvoPoint::LABEL_DIMENSION));
  semantic_bki::SemanticBKIOctoMap map_csm(resolution, block_depth, cvo::CvoPoint::LABEL_DIMENSION + 1, sf2, ell, prior, var_thresh, free_thresh, occupied_thresh);
  int i = 0;
  for (int i = start_frame; i < num_frames-start_frame; i++) {

    //pc_vec[i].read_cvo_pointcloud_from_file(f);
    std::string fname = pcd_dir + "/" + std::to_string(i)+".pcd";

    if (!std::filesystem::exists(fname))
      break;
    
    std::cout << "Reading " << fname << std::endl;    
    pcl::PointCloud<cvo::CvoPoint>::Ptr pc_pcl(new pcl::PointCloud<cvo::CvoPoint>);
    pcl::io::loadPCDFile<cvo::CvoPoint>(fname, *pc_pcl);
    for (int j = 0; j < pc_pcl->size(); j++) {
      auto & p = pc_pcl->at(j);
      p.features[0] = static_cast<float>(p.b) / 255.0;
      p.features[1] = static_cast<float>(p.g) / 255.0;
      p.features[2] = static_cast<float>(p.r) / 255.0;

    }
    pcl::io::savePCDFileASCII(pcd_dir+"/../segmented_color_pcd/"+std::to_string(i)+".pcd", *pc_pcl);
    
    
    // transform point cloud
    Eigen::Matrix4f transform = poses[i].cast<float>();
    cvo::CvoPointCloud transformed_pc(cvo::CvoPoint::FEATURE_DIMENSION, cvo::CvoPoint::LABEL_DIMENSION);
    cvo::CvoPointCloud::transform(transform, *pc_pcl, transformed_pc);

    if (i == start_frame) {
      transformed_pc.write_to_color_pcd(std::to_string(i)+"color.pcd");
      transformed_pc.write_to_label_pcd(std::to_string(i)+"semantic.pcd");
    }
    
    semantic_bki::point3f origin;
    origin.x() = transform(0, 3);
    origin.y() = transform(1, 3);
    origin.z() = transform(2, 3);

    // insert point cloud
    map_csm.insert_pointcloud_csm(&transformed_pc, origin, ds_resolution, free_resolution, max_range);

  }
  
  // Map to CVOPointCloud
  std::cout<<" cvt to pcd \n";
  cvo::CvoPointCloud cloud_out(&map_csm,  cvo::CvoPoint::FEATURE_DIMENSION, cvo::CvoPoint::LABEL_DIMENSION);
  //pc_vec[0].write_to_color_pcd(output_file + "_" + "input_color.pcd");
  //pc_vec[0].write_to_label_pcd(output_file + "_" + "input_semantics.pcd");
  cloud_out.write_to_color_pcd(output_file + "_" + "test_color.pcd");
  cloud_out.write_to_label_pcd(output_file + "_" + "test_semantics.pcd");

  return 0;
}
