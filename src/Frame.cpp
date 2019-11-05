
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "graph_optimizer/Frame.hpp"
#include "utils/StaticStereo.hpp"
#include "utils/Calibration.hpp"
namespace cvo {

  Frame::Frame(int ind,
               const cv::Mat &left_image,
               const cv::Mat & right_image,
               const Calibration & calib
               )
    : id(ind) ,
      h(left_image.rows),
      w(left_image.cols),
      raw_image_(left_image), 
      points_(raw_image_, right_image, calib) {
    
    
    
  }
  Frame::~Frame() {
    
    
  }
}
