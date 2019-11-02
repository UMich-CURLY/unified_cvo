
#include <iostream>
#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include "dataset_handler/KittiHandler.hpp"
#include "graph_optimizer/Frame.hpp"
#include "graph_optimizer/PoseGraph.hpp"

int main(int argc, char ** argv) {
  
  cvo::KittiHandler kitti(argv[1]);
  int total_iters = kitti.get_total_number();

  cvo::PoseGraph pose_graph;
  
  for (int i = 0; i < total_iters; i++) {
    cv::Mat left, right;
    if (kitti.read_next_stereo(left, right) == 0) {
      std::shared_ptr<cvo::Frame> new_frame(new cvo::Frame(i, left, right));
      pose_graph.add_new_frame(new_frame);
      pose_graph.optimize();
    }
  }
  return 0;
}
