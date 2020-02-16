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
    return 0;
  }

  int KittiHandler::read_next_stereo(cv::Mat &left, cv::Mat &right,
                                     int num_semantic_class,
                                     vector<float> & semantics) {

    if (read_next_stereo(left, right))
      return -1;
    
    string semantic_name = folder_name + "/image_semantic/" + names[curr_index] + ".bin";
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

  int KittiHandler::read_next_lidar(pcl::PointCloud<pcl::PointXYZI>::Ptr pc  ) {

    if (curr_index >= names.size())
      return -1;
    
    string lidar_bin_path = folder_name + "/velodyne/" + names[curr_index] + ".bin";
    
    std::ifstream fLidar(lidar_bin_path.c_str(),std::ios::in|std::ios::binary);

    fLidar.seekg (0, fLidar.end);
    int fLidar_length = fLidar.tellg();
    fLidar.seekg (0, fLidar.beg);
    int num_lidar_points = fLidar_length / 4;
    // std::cout << "fLidar_length: " << fLidar_length << ", num_lidar_points: " << num_lidar_points << std::endl;

    std::vector<float> lidar_points;

    if (fLidar.is_open()){
      int num_bytes = sizeof(float) * fLidar_length;
      
      lidar_points.resize(fLidar_length);
      infile.open(lidar_bin_path.c_str(),std::ios::in| std::ifstream::binary);

      infile.read( reinterpret_cast<char *>(lidar_points.data()), num_bytes );
      infile.close();

      // Eigen::Affine3f tf_change_basis = Eigen::Affine3f::Identity();
      Eigen::Affine3f rx = Eigen::Affine3f::Identity();
      Eigen::Affine3f ry = Eigen::Affine3f(Eigen::AngleAxisf(-M_PI/2.0, Eigen::Vector3f::UnitY()));
      Eigen::Affine3f rz = Eigen::Affine3f(Eigen::AngleAxisf(M_PI/2.0, Eigen::Vector3f::UnitZ()));
      Eigen::Affine3f tf_change_basis = rz * ry * rx;

      for(int r=0; r<num_lidar_points; ++r){          
          pcl::PointCloud<pcl::PointXYZI>::PointType temp_pcl;
          pcl::PointCloud<pcl::PointXYZI>::PointType pcl_after_tf;
          temp_pcl.x = lidar_points[r*4+0];
          temp_pcl.y = lidar_points[r*4+1];
          temp_pcl.z = lidar_points[r*4+2];

          Eigen::Vector4f temp_pt(temp_pcl.x, temp_pcl.y, temp_pcl.z, 1);
          Eigen::Vector4f after_tf_pt = tf_change_basis.matrix() * temp_pt;
          pcl_after_tf.x = after_tf_pt(0);
          pcl_after_tf.y = after_tf_pt(1);
          pcl_after_tf.z = after_tf_pt(2);
          // pcl_after_tf.x = -temp_pcl.y;
          // pcl_after_tf.y = -temp_pcl.z;
          // pcl_after_tf.z = temp_pcl.x;

          pcl_after_tf.intensity = lidar_points[r*4+3];
          // std::cout << "DEGUS-Kitti: point before " << to_string(r) << ", x=" << to_string(temp_pcl.x) << ", y=" << to_string(temp_pcl.y) 
          //                                    << ", z=" << to_string(temp_pcl.z) << ", intensity=" << to_string(temp_pcl.intensity) << std::endl;
          // std::cout << "DEGUS-Kitti: point after  " << to_string(r) << ", x=" << to_string(pcl_after_tf.x) << ", y=" << to_string(pcl_after_tf.y) 
          //                                    << ", z=" << to_string(pcl_after_tf.z) << ", intensity=" << to_string(pcl_after_tf.intensity) << std::endl;
          
          if(pcl_after_tf.x == 0 && pcl_after_tf.y == 0 && pcl_after_tf.z == 0 && pcl_after_tf.intensity == 0){
            break;
          }

          pc->push_back(pcl_after_tf);
      }     
    }
    else {
      cerr<<"Lidar doesn't read successfully: "<<lidar_bin_path<<"\n"<<std::flush;
      return -1;
    }
    return 0;
  }

  void KittiHandler::next_frame_index() {
    curr_index ++;
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
