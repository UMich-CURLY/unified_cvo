#pragma once

#include <iostream>
#include <utility>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "libelas/elas.h"
#include "utils/data_type.hpp"

namespace cvo {
  namespace StaticStereo {
    
    elas::Elas::parameters elas_init_params();
    
    enum TraceStatus {GOOD=0,
                      OOB,
                      OUTLIER};


    void disparity(const cv::Mat & left_gray,
                   const cv::Mat & right_gray,
                   std::vector<float> & output_left_disparity); 

    TraceStatus pt_depth_from_disparity(const std::vector<float> & left_disparity,
                                        const Mat33f & intrinsic,
                                        const float baseline,
                                        const Vec2f & input,
                                        // output
                                        Eigen::Ref<Vec2f> result
                                        );


  }
  
}
