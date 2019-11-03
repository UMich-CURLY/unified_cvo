
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "graph_optimizer/Frame.hpp"
#include "utils/StaticStereo.hpp"

namespace cvo {

  Frame::Frame(int ind,
               const cv::Mat &left_image,
               const cv::Mat & right_image)
    : id(ind) ,
      h(left_image.rows) ,
      w(left_image.cols) ,
      points(left_image, right_image) {
    
    raw_image.color_image = left_image.clone();

    cv::Mat left_gray, right_gray;
    cv::cvtColor(left_image, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right_image, right_gray, cv::COLOR_BGR2GRAY);

    std::vector<float> left_disparity;
    StaticStereo::disparity(left_gray, right_gray, left_disparity);
    
    
  }
  Frame::~Frame() {
    
    
  }
}
