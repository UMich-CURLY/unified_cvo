#pragma once

#include <vector>
#include "RawImage.hpp"

namespace cv {
  class Mat;
}

namespace cvo {
  class ImageRGBD : public RawImage {
  public:
    ImageRGBD(const cv::Mat & image,
              const std::vector<uint16_t> & depth_image) : depth_image_(depth_image),
                                                           RawImage(image) {}

    ImageRGBD(const cv::Mat & image,
              const std::vector<uint16_t> & depth_image,
              int num_classes,
              const std::vector<float> & semantics) : depth_image_(depth_image),
                                                      RawImage(image, num_classes, semantics) {}
    
    const std::vector<uint16_t> & depth_image() const  { return depth_image_; }
    
  private:
    std::vector<uint16_t>  depth_image_;
  };
  
}
