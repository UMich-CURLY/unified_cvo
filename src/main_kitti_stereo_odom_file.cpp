#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "dataset_handler/KittiHandler.hpp"
#include "graph_optimizer/Frame.hpp"
#include "graph_optimizer/PoseGraph.hpp"
#include "utils/Calibration.hpp"
int main(int argc, char ** argv) {
  assert(argc > 3);
  cvo::KittiHandler kitti(argv[1]);
  int total_iters = kitti.get_total_number();

  cvo::PoseGraph pose_graph(true, cvo::PoseGraph::FIXED_LAG_SMOOTHER,  7);

  std::string calib_name(argv[2]);
  cvo::Calibration calib(calib_name);

  int num_class = std::stoi(argv[3]);

  std::string odom_factor_log(argv[4]);
  std::ifstream infile(odom_factor_log);
  if (infile.is_open()) {
    std::string line;
    int curr_kf_id = 0;
    while (!infile.eof()) {
      
      
    }

    
    infile.close();
  }
  




  
  for (int i = 0; i < total_iters; i++) {
    cv::Mat left, right;
    std::vector<float> semantics;
    if (kitti.read_next_stereo(left, right, num_class, semantics ) == 0) {
      std::cout<<"\n====================================================\n";
      std::cout<<"[main] construct new frame "<<i<<"\n"<<std::flush;
      std::shared_ptr<cvo::Frame> new_frame(new cvo::Frame(i, left, right, num_class, semantics, calib ));
      pose_graph.add_new_frame(new_frame);
    } else {
      std::cout<<" read image fails\n";
      
    }
  }
  return 0;
}
