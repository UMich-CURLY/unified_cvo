#pragma once
#include <iostream>
#include <list>
#include <vector>
#include <string>
#include <memory>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <opencv2/opencv.hpp>

#include "utils/data_type.hpp"
#include "cvo/Cvo.hpp"
#include "Frame.hpp"

namespace cvo {
  class PoseGraph {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    PoseGraph();
    ~PoseGraph();

    // cvo_align and keyframe 
    void add_new_frame(std::shared_ptr<Frame> new_frame);


    //void write_trajectory(std::string filename);

    // gtsam pose graph optimization
    void pose_graph_optimize();
    
  private:
    std::list<std::shared_ptr<Frame>> all_frames_since_last_keyframe;
    std::list<std::shared_ptr<Frame>> last_two_frames;
    std::list<std::shared_ptr<Frame>> keyframes;
    gtsam::NonlinearFactorGraph factor_graph;
    cvo cvo_align;
    
  };
  
}
