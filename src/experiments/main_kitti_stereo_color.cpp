
#include <iostream>
#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include "dataset_handler/KittiHandler.hpp"
#include "graph_optimizer/Frame.hpp"
#include "graph_optimizer/PoseGraph.hpp"
#include "utils/Calibration.hpp"
int main(int argc, char ** argv) {
  
  cvo::KittiHandler kitti(argv[1], 0);
  int total_iters = kitti.get_total_number();

  cvo::PoseGraph pose_graph;

  std::string calib_name(argv[2]);
  cvo::Calibration calib(calib_name);
  
  for (int i = 0; i < total_iters; i++) {
    cv::Mat left, right;
    if (kitti.read_next_stereo(left, right ) == 0) {
      std::cout<<"construct new frame "<<i<<"\n"<<std::flush;
      std::shared_ptr<cvo::Frame> new_frame(new cvo::Frame(i, left, right, calib ));
      pose_graph.add_new_frame(new_frame);
    }
  }
  return 0;
}
