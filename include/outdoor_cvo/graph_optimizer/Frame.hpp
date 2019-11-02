#pragma once
#include <iostream>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "utils/data_type.hpp"
#include "utils/CvoPointCloud.hpp"

namespace cvo {

  struct RawImage {
    // assume all data to be float32
    cv::Mat color_image;
    std::vector<float> semantic_image;
  };
  
  class Frame {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Frame(int ind,
          const cv::Mat & left_image,
          const cv::Mat & right_image);
    
    ~Frame();
    
    const int id;
    const int h;
    const int w;

  private:
    
    CvoPointCloud points;

    //std::unique_ptr local_map;
    RawImage raw_image;
    
  };

    
  
}
