#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
#include <tbb/tbb.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include "utils/CvoPointCloud.hpp"
#include "mapping/bkioctomap.h"
#include "utils/Calibration.hpp"
#include "utils/ImageRGBD.hpp"
#include "dataset_handler/TumHandler.hpp"
#include "dataset_handler/PoseLoader.hpp"

using namespace std;
using namespace boost::filesystem;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_row;

void map_to_pc(  semantic_bki::SemanticBKIOctoMap & map,
                 cvo::CvoPointCloud & pc){

  int num_pts = 0;
  for (auto it = map.begin_leaf(); it != map.end_leaf(); ++it) {
    if (it.get_node().get_state() == semantic_bki::State::OCCUPIED) {
      num_pts++;
    }
  }
    //num_classes_ = num_classes;
  std::vector<std::vector<float>> features;
  std::vector<std::vector<float>> labels;
  pc.reserve(num_pts, 5, 0);
  int ind = 0;
  for (auto it = map.begin_leaf(); it != map.end_leaf(); ++it) {
    if (it.get_node().get_state() == semantic_bki::State::OCCUPIED) {
      // position
      semantic_bki::point3f p = it.get_loc();
      Eigen::Vector3f xyz;
      xyz << p.x(), p.y(), p.z();
      //positions_.push_back(xyz);
      // features
      std::vector<float> feature_vec(5, 0);
      it.get_node().get_features(feature_vec);
      Eigen::VectorXf feature = Eigen::Map<Eigen::VectorXf>(feature_vec.data(), 5);
      
      //features.push_back(feature);
      // labels
      //std::vector<float> label(num_classes_, 0);
      //it.get_node().get_occupied_probs(label);
      //labels.push_back(label);
      Eigen::VectorXf label, geometric_type;
      pc.add_point(ind, xyz, feature,label, geometric_type );
    }
  }
}

int main(int argc, char *argv[]) {
  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  std::cout<<argc<<std::endl;
  int num_class= 1;
  string n_class_str;
  //if (argc > 5) {
  //  n_class_str = argv[5];
  //num_class = //stoi(n_class_str);
  //}

  cvo::TumHandler tum(argv[1]);
  int total_iters = tum.get_total_number();
  vector<string> vstrRGBName = tum.get_rgb_name_list();
  string cvo_param_file(argv[2]);
  string calib_file;
  calib_file = string(argv[1] ) +"/cvo_calib.txt"; 
  //std::string output_dir(argv[3]);
  int start_frame = stoi(argv[3]);
  int num_frames = stoi(argv[4]);
  std::string odom_file(argv[5]);
  
  vector<string> files;
  std::cout<<" cycle through the directory\n";
  int total_num = 0;
  int last_frame = std::min(start_frame+num_frames-1, total_iters-1);
  tum.set_start_index(start_frame);
  cvo::Calibration  calib(calib_file, cvo::Calibration::RGBD);

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
  std::vector<Eigen::Matrix4d,
              Eigen::aligned_allocator<Eigen::Matrix4d>> poses;
  cvo::read_pose_file_tum_format(odom_file, start_frame, last_frame, poses);
  std::cout<<"Just read poses\n";
  
  // Build map
  //std::vector<cvo::CvoPointCloud> pc_vec(files.size());
  semantic_bki::SemanticBKIOctoMap map_csm(resolution, block_depth, num_class + 1, sf2, ell, prior, var_thresh, free_thresh, occupied_thresh);
  cvo::CvoPointCloud pc_full(5,0);
  int i = 0;
  for (; i+start_frame <= last_frame; i++) {
    
    std::cout << "Read Frame " << i+start_frame << std::endl;
    //pc_vec[i].read_cvo_pointcloud_from_file(f);
    cv::Mat source_rgb, source_dep;
    tum.read_next_rgbd(source_rgb, source_dep);
    std::vector<uint16_t> source_dep_data(source_dep.begin<uint16_t>(), source_dep.end<uint16_t>());

    //std::shared_ptr<cvo::Frame> source(new cvo::Frame(start_frame, source_rgb, source_dep,
    //19, semantics_source, 
    //                                                  calib, 1));
    std::shared_ptr<cvo::ImageRGBD<uint16_t>> source_raw(new cvo::ImageRGBD(source_rgb, source_dep_data));
    std::shared_ptr<cvo::CvoPointCloud> pc_original(new cvo::CvoPointCloud(*source_raw,
                                                                           calib
                                                                           //,cvo::CvoPointCloud::CANNY_EDGES
                                                                           ));
    
    
    // transform point cloud
    Eigen::Matrix4f transform = poses[i].cast<float>();
    cvo::CvoPointCloud transformed_pc(5,0);
    cvo::CvoPointCloud::transform(transform, *pc_original, transformed_pc);
    pc_full += transformed_pc;

    semantic_bki::point3f origin;
    origin.x() = transform(0, 3);
    origin.y() = transform(1, 3);
    origin.z() = transform(2, 3);

    // insert point cloud
    map_csm.insert_pointcloud_csm(&transformed_pc, origin, ds_resolution, free_resolution, max_range);
  }
  
  // Map to CVOPointCloud
  //cvo::CvoPointCloud cloud_out(&map_csm, num_class);
  cvo::CvoPointCloud pc_map(5,0);
  map_to_pc(map_csm, pc_map);
  //pc_vec[0].write_to_color_pcd(output_dir + "/" + "input_color.pcd");
  //pc_vec[0].write_to_label_pcd(output_dir + "/" + "input_semantics.pcd");
  pc_map.write_to_color_pcd("map.pcd");
  pc_full.write_to_color_pcd("stacked_pc.pcd");
  //cloud_out.write_to_label_pcd(output_dir + "/" + "test_semantics.pcd");

  return 0;
}
