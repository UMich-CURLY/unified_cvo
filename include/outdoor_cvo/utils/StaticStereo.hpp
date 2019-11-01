#pragma once

#include <iostream>
#include <utility>
#include <opencv2/opencv.hpp>

#include "utils/data_type.hpp"

namespace cvo {
  class StaticStereo {
    enum TraceStatus {GOOD=0,
                      OOB,
                      OUTLIER};

    static TraceStatus trace_stereo(const cv::Mat & left,
                                    const cv::Mat & right,
                                    const Mat33f & intrinsic,
                                    const float baseline,
                                    const pair<float, float> & input,
                                    // output
                                    std::pair<float, float> & result
                                    ) const ;
  };
  
}
