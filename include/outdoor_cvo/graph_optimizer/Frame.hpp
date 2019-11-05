#pragma once
#include <iostream>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "utils/data_type.hpp"
#include "utils/CvoPointCloud.hpp"
#include "utils/RawImage.hpp"
#include "utils/Calibration.hpp"
namespace cvo {

  class Frame {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Frame(int ind,
          const cv::Mat & left_image,
          const cv::Mat & right_image,
          const Calibration & calib);
    
    ~Frame();
    
    const int id;
    const int h;
    const int w;

    const CvoPointCloud & points() {return points_;}
    const RawImage & raw_image() { return raw_image_;}

  private:

    //std::unique_ptr local_map;
    RawImage raw_image_;


    CvoPointCloud points_;
  };

    
  
}
