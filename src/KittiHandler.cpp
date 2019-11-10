#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cassert>
#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include "dataset_handler/KittiHandler.hpp"
#include "utils/debug_visualization.hpp"
using namespace std;
using namespace boost::filesystem;

namespace cvo {
  KittiHandler::KittiHandler(std::string kitti_folder) {
    curr_index = 0;
    folder_name = kitti_folder;
    debug_plot = true;

    string left_color_folder = folder_name + "/image_2/";
    path kitti(left_color_folder.c_str());
    for (auto & p : directory_iterator(kitti)) {
      if (is_regular_file(p.path())) {
        string curr_file = p.path().filename().string();
        size_t last_ind = curr_file.find_last_of(".");
        string raw_name = curr_file.substr(0, last_ind);
        names.push_back(raw_name);
      }
    }
    sort(names.begin(), names.end());
    cout<<"Kitti contains "<<names.size()<<" files\n";
  }

  int KittiHandler::read_next_stereo(cv::Mat & left,
                                     cv::Mat & right) {
    if (curr_index >= names.size())
      return -1;

    string left_name = folder_name + "/image_2/" + names[curr_index] + ".png";
    string right_name = folder_name + "/image_3/" + names[curr_index] + ".png";
    left = cv::imread(left_name, cv::ImreadModes::IMREAD_COLOR );
    right = cv::imread(right_name, cv::ImreadModes::IMREAD_COLOR );

    if (left.data == nullptr || right.data == nullptr) {
      cerr<<"Image doesn't read successfully: "<<left_name<<", "<<right_name<<"\n";
      return -1;
    }
    curr_index ++;
    return 0;
  }

  int KittiHandler::read_next_stereo(cv::Mat &left, cv::Mat &right,
                                     int num_semantic_class,
                                     vector<float> & semantics) {

    if (read_next_stereo(left, right))
      return -1;
    
    string semantic_name = folder_name + "/image_semantic/" + names[curr_index-1] + ".bin";
    int num_bytes = sizeof(float) * num_semantic_class * left.total();
    semantics.resize(left.total() * num_semantic_class);
    infile.open(semantic_name.c_str(),std::ios::in| std::ifstream::binary);
    if (infile.is_open()) {
      infile.read( reinterpret_cast<char *>(semantics.data()), num_bytes );
      infile.close();
    } else {
      cerr<<"Semantic Image doesn't read successfully: "<<semantic_name<<"\n"<<std::flush;
      return -1;
      
    }

    if (debug_plot)
      visualize_semantic_image("last_semantic.png", semantics.data(), num_semantic_class, left.cols, left.rows );

    return 0;
  }

  int KittiHandler::read_next_lidar_mono(cv::Mat & image,
                                         pcl::PointCloud<pcl::PointXYZ>::Ptr pc  ) {
    std::cerr<<"lidar-mono not implemented in KittiHandler\n";
    assert(0);
    
  }


  void KittiHandler::set_start_index(int start) {
    curr_index = start;
    
  }

  int KittiHandler::get_current_index() {
    return curr_index;
    
  }

  int KittiHandler::get_total_number() {
    return names.size();
    
  }
  
}
