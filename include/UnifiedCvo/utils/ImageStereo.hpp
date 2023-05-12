#pragma once

#include <vector>
#include "RawImage.hpp"

namespace cv {
  class Mat;
}

namespace cvo {
  class ImageStereo : public RawImage {
  public:
    ImageStereo(const cv::Mat & left_image,
                const cv::Mat & right_image);

    ImageStereo(const cv::Mat & left_image,
                const cv::Mat & right_image,
                int num_classes,
                const  std::vector<float> & semantics);

    const std::vector<float> & disparity() const {return disparity_;}
    const cv::Mat & right() const {return right_;}
    
  private:

    std::vector<float> disparity_;
    cv::Mat right_;
  };
}
