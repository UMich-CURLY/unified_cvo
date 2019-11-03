#include <string>
#include <fstream>
#include <cstdio>
#include <iostream>
#include "utils/CvoPointCloud.hpp"

namespace cvo{

  CvoPointCloud::CvoPointCloud(const cv::Mat & left_image,
                               const cv::Mat & right_image) {
    
    
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
  
}
