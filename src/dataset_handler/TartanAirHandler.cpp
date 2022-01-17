#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cassert>
#include <boost/filesystem.hpp>
#include "dataset_handler/TartanAirHandler.hpp"
#include "cnpy.h"
using namespace std;
using namespace boost::filesystem;

namespace cvo {
  TartanAirHandler::TartanAirHandler(std::string tartan_traj_folder){
    // use left camera only, rgbd
    const string depth_pth = tartan_traj_folder + "/depth_left";
    const string image_pth = tartan_traj_folder + "/image_left";
    // count number of files in both dirs
    int depth_count = 0;
    directory_iterator end_it;
    for (directory_iterator it(depth_pth); it != end_it; it++) {
      depth_count++;
    }
    int image_count = 0;
    for (directory_iterator it(image_pth); it != end_it; it++) {
      image_count++;
    }
    assert (depth_count == image_count);
    total_size = depth_count;
    curr_index = 0;
    folder_name = tartan_traj_folder;
    cout << "Found " << total_size << " image pairs\n";
  }

  TartanAirHandler::~TartanAirHandler() {}
  
  int TartanAirHandler::read_next_rgbd(cv::Mat & rgb_img, cv::Mat & dep_img) {
    if (curr_index >= total_size)
      return -1;
    // format curr_index
    stringstream ss;
    ss << setw(6) << setfill('0') << curr_index;
    string index_str = ss.str();
    // read rgb image
    string img_pth = folder_name + "/image_left/" + index_str + "_left.png";
    rgb_img = cv::imread(img_pth, cv::ImreadModes::IMREAD_COLOR);
    // read depth npy
    string dep_pth = folder_name + "/depth_left/" + index_str + "_left_depth.npy";
    cnpy::NpyArray dep_arr = cnpy::npy_load(dep_pth);
    float* dep_data = dep_arr.data<float>();
    int dim1 = dep_arr.shape[0];
    int dim2 = dep_arr.shape[1];
    cv::Mat raw_dep(cv::Size(dim2, dim1), CV_32FC1, dep_data);
    // set high depth pixels (sky) to nan
    for (int r = 0; r < raw_dep.rows; r++) {
      for (int c = 0; c < raw_dep.cols; c++) {
        if (raw_dep.at<float>(r, c) < 10000) continue;
        raw_dep.at<float>(r, c) = std::nanf("1");
      }
    }
    // scale by 5000 and convert to uint16_t
    raw_dep = raw_dep * 500.0f;
    raw_dep.convertTo(dep_img, CV_16UC1);
    return 0;
  }

  int TartanAirHandler::read_next_rgbd(cv::Mat & rgb_img, std::vector<float> & dep_vec) {
    if (curr_index >= total_size)
      return -1;
    // format curr_index
    stringstream ss;
    ss << setw(6) << setfill('0') << curr_index;
    string index_str = ss.str();
    // read rgb image
    string img_pth = folder_name + "/image_left/" + index_str + "_left.png";
    rgb_img = cv::imread(img_pth, cv::ImreadModes::IMREAD_COLOR);
    // read depth npy
    string dep_pth = folder_name + "/depth_left/" + index_str + "_left_depth.npy";
    cnpy::NpyArray dep_arr = cnpy::npy_load(dep_pth);
    float* dep_data = dep_arr.data<float>();
    int dim1 = dep_arr.shape[0];
    int dim2 = dep_arr.shape[1];
    cv::Mat raw_dep(cv::Size(dim2, dim1), CV_32FC1, dep_data);
    // set high depth pixels (sky) to nan
    for (int r = 0; r < raw_dep.rows; r++) {
      for (int c = 0; c < raw_dep.cols; c++) {
        if (raw_dep.at<float>(r, c) < 10000) continue;
        raw_dep.at<float>(r, c) = std::nanf("1");
      }
    }
    // scale by 5000 and flatten to vector
    raw_dep = raw_dep * 5000.0f;
    dep_vec.clear();
    dep_vec = vector<float>(raw_dep.begin<float>(), raw_dep.end<float>());
    return 0;
  }

  int TartanAirHandler::read_next_stereo(cv::Mat & left, cv::Mat & right) {
    if (curr_index >= total_size)
      return -1;
    // format curr_index
    stringstream ss;
    ss << setw(6) << setfill('0') << curr_index;
    string index_str = ss.str();
    // read l, r images
    string left_pth = folder_name + "/image_left/" + index_str + "_left.png";
    string right_pth = folder_name + "/image_right/" + index_str + "_right.png";
    left = cv::imread(left_pth, cv::ImreadModes::IMREAD_COLOR);
    right = cv::imread(right_pth, cv::ImreadModes::IMREAD_COLOR);
    if (left.data == nullptr || right.data == nullptr) {
      cerr << "Image doesn't read successfully: " << left_pth << ", " << right_pth << "\n";
      return -1;
    }
    return 0;
  }

  void TartanAirHandler::next_frame_index() {
    curr_index++;
  }


  void TartanAirHandler::set_start_index(int start) {
    curr_index = start;
  }

  int TartanAirHandler::get_current_index() {
    return curr_index;
  }

  int TartanAirHandler::get_total_number() {
    return total_size;
  }

}
