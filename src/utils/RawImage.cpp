#include <cmath>
#include <opencv2/photo.hpp>
#include "utils/RawImage.hpp"


namespace cvo {
  RawImage::RawImage(const cv::Mat & image)
    : gradient_(image.total() * 2, 0),
      gradient_square_(image.total(), 0) {

    if (image.total() == 0)
      return;
    
    num_class_ = 0;
    image_ = image.clone();
    rows_ = image.rows;
    cols_ = image.cols;
    channels_ = image.channels();
    if (channels_==3)
      cv::fastNlMeansDenoisingColored(image_,image_,10,10,7,21);
    else if (channels_ == 1)
      cv::fastNlMeansDenoising(image_,image_,10,7,21);
    else {
      std::cerr<<"Image channels should be 1 or 3!\n";
      return;
    }
    //cv::fastNlMeansDenoising (color_, color_);
    intensity_.resize(image_.total());
    cv::Mat gray;    
    if (channels_ == 3) {
      cv::cvtColor(image_, gray, cv::COLOR_BGR2GRAY);
      gray.convertTo(gray, CV_32FC1);
    } else {
      image.convertTo(gray, CV_32FC1);
    }
    memcpy(intensity_.data(), gray.data, sizeof(float) * image_.total());      
    compute_image_gradient();
  }

  RawImage::RawImage(const cv::Mat & image, int num_classes, const std::vector<float> & semantic)
    : RawImage(image)  {
    num_class_ = num_classes;
    semantic_image_.resize(semantic.size());
    memcpy(semantic_image_.data(), semantic.data(), sizeof(float) * semantic.size() );
  }

  RawImage::RawImage(){
    
  }

  void RawImage::compute_image_gradient() {


    // calculate gradient
    // we skip the first row&col and the last row&col

    for(int idx=cols_; idx<cols_*(rows_-1); idx++) {
      if (idx % cols_ == 0 || idx%cols_ == cols_-1) {
        gradient_[idx * 2] = 0;
        gradient_[idx * 2 + 1] = 0;
        gradient_square_[idx] = 0;
        continue;
      }
                
      float dx = 0.5f*( intensity_[idx+1] - intensity_[idx-1] );
      float dy = 0.5f*( intensity_[idx+cols_] - intensity_[idx-cols_] );

      // if it's not finite, set to 0
      if(!std::isfinite(dx)) dx=0;
      if(!std::isfinite(dy)) dy=0;
                
      gradient_[2*idx] = dx;
      gradient_[2*idx+1] = dy;
      gradient_square_[idx] = dx*dx+dy*dy;

    }
    
  }
}
