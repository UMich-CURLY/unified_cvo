#include <cassert>
#include <cmath>
#include <utility>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utils/StaticStereo.hpp"

namespace cvo{

  namespace StaticStereo {


    elas::Elas::parameters elas_init_params() {
      elas::Elas::parameters p;
      p.postprocess_only_left = true;
      return p;
    }

    static elas::Elas elas_(elas_init_params());


    
    void disparity(const cv::Mat & left_gray,
                   const cv::Mat & right_gray,
                   std::vector<float> & output_left_disparity)  {

      int32_t width = left_gray.cols;
      int32_t height = left_gray.rows;
      output_left_disparity.resize(width*height);
      std::vector<float> right_disparity(left_gray.total());
      int32_t dims[3] = {width , height, width};
      elas_.process(left_gray.data, right_gray.data,
                    output_left_disparity.data(), right_disparity.data(),
                    dims);

      bool is_visualize = false;
      if (is_visualize) {
        std::vector<uint8_t> vis(output_left_disparity.size());
        float disp_max = 0;
        auto D1_data = output_left_disparity.data();
        for (int32_t i=0; i<width*height; i++) {
          if (D1_data[i]>disp_max) disp_max = D1_data[i];
        }
        // copy float to uchar
        for (int32_t i=0; i<width*height; i++) 
          vis[i] = (uint8_t)std::max(255.0*D1_data[i]/disp_max,0.0);
        // convet to cv mat
        cv::Mat disp_left(height, width, CV_8UC1, vis.data());
        //cv::namedWindow("Disparity left", )
        cv::imshow("Left disparity", disp_left);
        cv::waitKey(100);
      }
    }
  
    TraceStatus pt_depth_from_disparity(const RawImage & left,
                                        //const cv::Mat & right_gray,
                                        const std::vector<float> & disparity,
                                        const Calibration & calib,
                                        const Vec2i & input,
                                        Eigen::Ref<Vec3f> result
                                        )  {
      Vec3f bl;
      bl << calib.baseline(), 0, 0;
      int u = input(0);
      int v = input(1);
      int h = left.color().rows;
      int w = left.color().cols;

      if ( u < 1 || u > w-2 || v < 1 || v > h -2 )
        return TraceStatus::OOB;

      if (disparity[w * v + u] <= 0.05)
        return TraceStatus::OUTLIER;
      
      float depth = std::abs(calib.baseline()) * calib.intrinsic()(0,0) / disparity[w * v + u];

      result << static_cast<float>(u), static_cast<float>(v), 1.0;
      
      result = (calib.intrinsic().inverse() * result * depth).eval();

      return TraceStatus::GOOD;
    
    }
  }
  
  
}
