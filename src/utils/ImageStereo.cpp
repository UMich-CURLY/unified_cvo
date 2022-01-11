#include <vector>
#include <opencv2/opencv.hpp>
#include "utils/StaticStereo.hpp"
#include "utils/RawImage.hpp"
#include "utils/ImageStereo.hpp"

namespace cvo {
  ImageStereo::ImageStereo(const cv::Mat & left_image,
                           const cv::Mat & right_image) : RawImage(left_image) {

    cv::Mat  left_gray, right_gray;
    if (left_image.channels() == 3)
      cv::cvtColor(left_image, left_gray, cv::COLOR_BGR2GRAY);
    else
      left_gray = left_image;

    if (right_image.channels() == 3)
      cv::cvtColor(right_image, right_gray, cv::COLOR_BGR2GRAY);
    else
      right_gray = right_image;
    
    StaticStereo::disparity(left_gray, right_gray, disparity_);

  }

  ImageStereo::ImageStereo(const cv::Mat & left_image,
                           const cv::Mat & right_image,
                           int num_classes,
                           const  std::vector<float> & semantics) :
    RawImage(left_image, num_classes, semantics) {
    
    cv::Mat  left_gray, right_gray;
    if (left_image.channels() == 3)
      cv::cvtColor(left_image, left_gray, cv::COLOR_BGR2GRAY);
    else
      left_gray = left_image;

    if (right_image.channels() == 3)
      cv::cvtColor(right_image, right_gray, cv::COLOR_BGR2GRAY);
    else
      right_gray = right_image;
    
    StaticStereo::disparity(left_gray, right_gray, disparity_);
    
  }


  
}
