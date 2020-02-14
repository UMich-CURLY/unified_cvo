#pragma once

#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils/RawImage.hpp"
#include "utils/data_type.hpp"

namespace cvo {

  class RawImage {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    RawImage(const cv::Mat & left_image);
    RawImage(const cv::Mat & left_image, int num_classes, const std::vector<float> & semantic);
    RawImage();

    const std::vector<float> intensity() const { return intensity_; }
    const cv::Mat & color() const { return color_;}
    const std::vector<Vec2f, Eigen::aligned_allocator<Vec2f>> & gradient() const {return gradient_;}
    const std::vector<float> & gradient_square() const {return gradient_square_;}
    const std::vector<float> & semantic_image() const {return semantic_image_;}
    int num_class() const {return num_class_;}
    
  private: 
    // assume all data to be float32
    cv::Mat color_;
    std::vector<float> intensity_;
    std::vector<Vec2f, Eigen::aligned_allocator<Vec2f>> gradient_;
    std::vector<float> gradient_square_;
    int num_class_;
    std::vector<float> semantic_image_;

    // fill in gradient_ and gradient_square_
    void compute_image_gradient();
    
    
  };

}
