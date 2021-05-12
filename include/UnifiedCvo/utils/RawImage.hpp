#pragma once

#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils/RawImage.hpp"
//#include "utils/data_type.hpp"

namespace cvo {

  class RawImage {
  public:
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    RawImage(const cv::Mat & left_image);
    RawImage(const cv::Mat & left_image, int num_classes, const std::vector<float> & semantic);
    RawImage();
    ~RawImage() {}

    const std::vector<float> intensity() const { return intensity_; }
    const cv::Mat & image() const { return image_;}
    const std::vector<float> & gradient() const {return gradient_;}
    const std::vector<float> & gradient_square() const {return gradient_square_;}
    const std::vector<float> & semantic_image() const {return semantic_image_;}
    int rows() const {return rows_;}
    int cols() const {return cols_;}
    int channels() const {return channels_;}
    int num_class() const {return num_class_;}
    
  private: 
    // assume all data to be float32
    cv::Mat image_;
    std::vector<float> intensity_;
    std::vector<float> gradient_; // size: image total pixels x 2
    std::vector<float> gradient_square_;
    int num_class_;
    int rows_;
    int cols_;
    int channels_;
    std::vector<float> semantic_image_;

    // fill in gradient_ and gradient_square_
    void compute_image_gradient();
    
    
  };

}
