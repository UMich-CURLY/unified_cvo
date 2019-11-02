
#include "graph_optimizer/Frame.hpp"

namespace cvo {
  Frame::Frame(int ind,
               const cv::Mat &left_image,
               const cv::Mat & right_image)
    : id(ind) ,
      h(left_image.rows) ,
      w(left_image.cols) ,
      points(left_image, right_image) {
    
    raw_image.color_image = left_image.clone();

  }
  Frame::~Frame() {
    
    
  }
}
