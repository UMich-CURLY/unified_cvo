#include <iostream>
#include <vector>
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

using namespace std;
using namespace boost::filesystem;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_row;

bool read_camera_poses(const std::string camera_pose_name, Eigen::MatrixXf& camera_poses) {
  if (std::ifstream(camera_pose_name)) {
    std::vector<std::vector<float>> camera_poses_v;
    std::ifstream fPoses;
    fPoses.open(camera_pose_name.c_str());
    int counter = 0;
    while (!fPoses.eof()) {
      std::vector<float> camera_pose_v;
      std::string s;
      std::getline(fPoses, s);
      if (!s.empty()) {
        std::stringstream ss;
        ss << s;
        float t;
        for (int i = 0; i < 12; ++i) {
          ss >> t;
          camera_pose_v.push_back(t);
        }
        camera_poses_v.push_back(camera_pose_v);
        counter++;
      }
    }
    fPoses.close();
    camera_poses.resize(counter, 12);
    for (int c = 0; c < counter; ++c) {
      for (int i = 0; i < 12; ++i)
        camera_poses(c, i) = camera_poses_v[c][i];
    }
    return true;
  } else {
    std::cout << "Cannot open camera pose file " << camera_pose_name << std::endl;
    return false;
  }
}

Eigen::Matrix4f get_current_pose(const Eigen::MatrixXf& camera_poses, const int scan_id) {
  Eigen::VectorXf curr_pose_v = camera_poses.row(scan_id);
  Eigen::MatrixXf curr_pose = Eigen::Map<MatrixXf_row>(curr_pose_v.data(), 3, 4);
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  transform.block(0, 0, 3, 4) = curr_pose;
  return transform;
}

int main(int argc, char *argv[]) {
  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  std::cout<<argc<<std::endl;
  int num_class= 0;
  string n_class_str;
  if (argc > 5) {
    n_class_str = argv[5];
    num_class = stoi(n_class_str);
  }
  
  path p (argv[1] );
  //std::ofstream output_file(argv[2]);
  std::string output_file(argv[2]);
  int start_frame = stoi(argv[3]);
  int num_frames = stoi(argv[4]);
  
  vector<string> files;
  std::cout<<" cycle through the directory\n";
  int total_num = 0;
  for(auto & p : boost::filesystem::directory_iterator( p ) ) {
    // If it's not a directory, list it. If you want to list directories too, just remove this check.
    if (is_regular_file(p.path())) {
      // assign current file name to current_file and echo it out to the console.
      string current_file = p.path().string();
      files.push_back(string(argv[1]) + "/" + to_string(total_num) + ".txt" );
      total_num += 1;
      cout <<"reading "<< current_file << endl; 

    }
  }
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
  Eigen::MatrixXf camera_poses;
  //std::string camera_pose_file = string(argv[1]) + "/" + "poses.txt";
  std::string camera_pose_file = "poses.txt";
  read_camera_poses(camera_pose_file, camera_poses);
  std::cout<<"Just read poses\n";
  // Build map
  std::vector<cvo::CvoPointCloud> pc_vec(files.size());
  semantic_bki::SemanticBKIOctoMap map_csm(resolution, block_depth, num_class + 1, sf2, ell, prior, var_thresh, free_thresh, occupied_thresh);
  int i = 0;
  for ( auto &f: files) {
    std::cout << "Reading " << f << std::endl;
    pc_vec[i].read_cvo_pointcloud_from_file(f);
    
    // transform point cloud
    Eigen::Matrix4f transform = get_current_pose(camera_poses, i);

    cvo::CvoPointCloud transformed_pc;
    cvo::CvoPointCloud::transform(transform, pc_vec[i], transformed_pc);
    //pc_vec[i].transform(transform);
    semantic_bki::point3f origin;
    origin.x() = transform(0, 3);
    origin.y() = transform(1, 3);
    origin.z() = transform(2, 3);

    // insert point cloud
    map_csm.insert_pointcloud_csm(&transformed_pc, origin, ds_resolution, free_resolution, max_range);
    ++i;
    if (i == num_frames) break;
  }
  
  // Map to CVOPointCloud
  cvo::CvoPointCloud cloud_out(&map_csm, num_class);
  pc_vec[0].write_to_color_pcd(output_file + "/" + "input_color.pcd");
  pc_vec[0].write_to_label_pcd(output_file + "/" + "input_semantics.pcd");
  cloud_out.write_to_color_pcd(output_file + "/" + "test_color.pcd");
  cloud_out.write_to_label_pcd(output_file + "/" + "test_semantics.pcd");

  return 0;
}
