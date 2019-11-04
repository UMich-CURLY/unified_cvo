#include <string>
#include <fstream>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <pcl/io/pcd_io.h>
#include "utils/CvoPointCloud.hpp"

namespace cvo{

  CvoPointCloud::CvoPointCloud(const cv::Mat & left_image,
                               const cv::Mat & right_image) {
    
    
  }
  
  CvoPointCloud::CvoPointCloud(const semantic_bki::SemanticBKIOctoMap& map,
                               const int num_classes) {
    num_classes_ = num_classes;
    std::vector<std::vector<float>> features;
    std::vector<std::vector<float>> labels;
    for (auto it = map.begin_leaf(); it != map.end_leaf(); ++it) {
      if (it.get_node().get_state() == semantic_bki::State::OCCUPIED) {
        // position
        semantic_bki::point3f p = it.get_loc();
        Vec3f xyz;
        xyz << p.x(), p.y(), p.z();
        positions_.push_back(xyz);
        // features
        std::vector<float> feature(5, 0);
        it.get_node().get_features(feature);
        features.push_back(feature);
        // labels
        std::vector<float> label(num_classes_, 0);
        it.get_node().get_occupied_probs(label);
        labels.push_back(label);
      }
    }
    num_points_ = positions_.size();
    features_.resize(num_points_, 5);
    labels_.resize(num_points_, num_classes_);
    for (int i = 0; i < num_points_; ++i) {
      for (int j = 0; j < 5; ++j) {
        features_(i, j) = features[i][j];
      }
      for (int j = 0; j < num_classes_; ++j) {
        labels_(i, j) = labels[i][j];
      }
    }
  }

  CvoPointCloud::CvoPointCloud(){}
  CvoPointCloud::~CvoPointCloud() {
    
    
  }

  int CvoPointCloud::read_cvo_pointcloud_from_file(const std::string & filename) {
    std::ifstream infile(filename);
    if (infile.is_open()) {
      infile>> num_points_;
      infile>> num_classes_;
      positions_.clear();
      positions_.resize(num_points_);
      features_.resize(num_points_, 5);
      if (num_classes_)
        labels_.resize(num_points_, num_classes_ );
      for (int i = 0; i < num_points_; i++) {
        float u, v;
        infile >> u >>v;
        float idepth;
        infile >> idepth;
        for (int j = 0; j <5 ; j++)
          infile >> features_(i, j);
        features_(i,0) = features_(i,0) / 255.0;
        features_(i,1) = features_(i,1) / 255.0;
        features_(i,2) = features_(i,2) / 255.0;
        features_(i,3) = features_(i,3) / 500.0 + 0.5;
        features_(i,4) = features_(i,4) / 500.0 + 0.5;
        
        for (int j = 0; j < 3; j++)
          infile >> positions_[i](j);
        for (int j = 0; j < num_classes_; j++)
          infile >> labels_(i, j);
      }
      infile.close();

      std::cout<<"Read pointcloud with "<<num_points_<<" points in "<<num_classes_<<" classes. \n The first point is ";
      std::cout<<" xyz: "<<positions_[0].transpose()<<", rgb_dxdy is "<<features_.row(0) <<"\n";
      if (num_classes_)
        std::cout<<" semantics is "<<labels_.row(0)<<std::endl;
      std::cout<<"The last point is ";
      std::cout<<" xyz: "<<positions_[num_points_-1].transpose()<<", rgb_dxdy is "<<features_.row(num_points_-1)<<"\n";
      if (num_classes_)
        std::cout<<" semantics is "<<labels_.row(num_points_-1)<<std::endl;
      return 0;
    } else
      return -1;
    
  }
  
  void CvoPointCloud::transform(const Eigen::Matrix4f& pose) {
    tbb::parallel_for(int(0), num_points_, [&](int j) {
      positions_[j] = (pose.block(0, 0, 3, 3) * positions_[j] + pose.block(0, 3, 3, 1)).eval();
    });
  }

  void CvoPointCloud::write_to_color_pcd(const std::string & name) const {
    pcl::PointCloud<pcl::PointXYZRGB> pc;
    for (int i = 0; i < num_points_; i++) {
      pcl::PointXYZRGB p;
      p.x = positions_[i](0);
      p.y = positions_[i](1);
      p.z = positions_[i](2);
      p.r = static_cast<uint8_t>(std::min(255.0f, features_(i, 0) * 255));
      p.g = static_cast<uint8_t>(std::min(255.0f, features_(i, 1) * 255));
      p.b = static_cast<uint8_t>(std::min(255.0f, features_(i, 2) * 255));
      pc.push_back(p);
    }
    pcl::io::savePCDFileASCII(name ,pc);
  }
  
  void CvoPointCloud::write_to_label_pcd(const std::string & name) const {
    if (num_classes_ < 1)
      return;
    pcl::PointCloud<pcl::PointXYZL> pc;
    for (int i = 0; i < num_points_; i++) {
      pcl::PointXYZL p;
      p.x = positions_[i](0);
      p.y = positions_[i](1);
      p.z = positions_[i](2);
      labels_.row(i).maxCoeff(&(p.label));
      pc.push_back(p);
    }
    pcl::io::savePCDFileASCII(name, pc);
  }

}
