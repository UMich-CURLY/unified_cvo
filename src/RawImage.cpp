#include <cmath>
#include <opencv2/photo.hpp>
#include "utils/RawImage.hpp"


namespace cvo {
  RawImage::RawImage(const cv::Mat & image)
    : gradient_(image.total()),
      gradient_square_(image.total(), 0) {
    num_class_ = 0;
    color_ = image.clone();
    cv::fastNlMeansDenoisingColored(color_,color_,10,10,7,21);
    //cv::fastNlMeansDenoising (color_, color_);
    cv::imwrite("denoised.png", color_);
    cv::Mat gray;
    cv::cvtColor(color_, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32FC1);
    intensity_.resize(color_.total());
    memcpy(intensity_.data(), gray.data, sizeof(float) * color_.total());
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

    int h = color_.rows;
    int w = color_.cols;

    // calculate gradient
    // we skip the first row&col and the last row&col
    for(int idx=w; idx<w*(h-1); idx++) {
      if (idx % w == 0 || idx%w == w-1) {
        gradient_[idx] << 0 ,0;
        gradient_square_[idx] = 0;
        continue;
      }
                
      float dx = 0.5f*( intensity_[idx+1] - intensity_[idx-1] );
      float dy = 0.5f*( intensity_[idx+w] - intensity_[idx-w] );

      // if it's not finite, set to 0
      if(!std::isfinite(dx)) dx=0;
      if(!std::isfinite(dy)) dy=0;
                
      gradient_[idx] << dx, dy;
      gradient_square_[idx] = dx*dx+dy*dy;

    }
    
  }
}
