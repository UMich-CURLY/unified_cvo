#pragma once

#include <vector>
#include "RawImage.hpp"

namespace cv {
  class Mat;
}

namespace cvo {
  template <typename DepthType>
  class ImageRGBD : public RawImage {
  public:
    ImageRGBD(const cv::Mat & image,
              const std::vector<DepthType> & depth_image) : depth_image_(depth_image),
                                                            RawImage(image) {}

    ImageRGBD(const cv::Mat & image,
              const std::vector<DepthType> & depth_image,
              int num_classes,
              const std::vector<float> & semantics) : depth_image_(depth_image),
                                                      RawImage(image, num_classes, semantics) {}

    ImageRGBD(const cv::Mat & image,
              const std::vector<DepthType> & depth_image,
              int num_classes,
              const std::vector<float> & semantics,
              bool is_adding_semantic_noise) : depth_image_(depth_image),
                                                      RawImage(image, num_classes, semantics) {}
    
    
    const std::vector<DepthType> & depth_image() const  { return depth_image_; }
    
  private:
    std::vector<DepthType>  depth_image_;
  };
  
}
