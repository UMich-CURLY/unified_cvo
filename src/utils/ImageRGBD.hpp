#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "RawImage.hpp"

namespace cvo {
  class ImageStereo : public RawImage {
  public:
    ImageStereo(const cv::Mat & left_image,
                const cv::Mat & right_image);

    ImageStereo(const cv::Mat & left_image,
                const cv::Mat & right_image,
                int num_classes,
                const std::vector<float> & semantic);



    const std::vector<float> & get_disparity() { return disparity_; }

  private:
    std::vector<float> disparity_;
  };
}
