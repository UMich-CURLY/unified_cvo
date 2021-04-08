#pragma once

#include <iostream>
#include <utility>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "libelas/elas.h"
#include "utils/data_type.hpp"
#include "utils/RawImage.hpp"
#include "utils/Calibration.hpp"
namespace cvo {
  namespace StaticStereo {
    
    elas::Elas::parameters elas_init_params();
    
    enum TraceStatus {GOOD=0,
                      OOB,
                      OUTLIER};


    void disparity(const cv::Mat & left_gray,
                   const cv::Mat & right_gray,
                   std::vector<float> & output_left_disparity); 

    TraceStatus pt_depth_from_disparity(const RawImage & left_image,
                                        const std::vector<float> & left_disparity,
                                        const Calibration & calib,
                                        // output
                                        const Vec2i & uv,
                                        Eigen::Ref<Vec3f> result
                                        );


  }
  
}
