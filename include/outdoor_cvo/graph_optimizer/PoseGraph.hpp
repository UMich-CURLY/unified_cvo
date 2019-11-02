#pragma once
#include <iostream>
#include <list>
#include <string>
#include <memory>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <opencv2/opencv.hpp>

#include "utils/data_type.hpp"
#include "Frame.hpp"

namespace cvo {
  class PoseGraph {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    PoseGraph();
    ~PoseGraph();

    // interfacing with outside
    void add_new_frame(std::shared_ptr<Frame> new_frame);

    //void write_trajectory(std::string filename);

    void optimize();
    
  private:
    std::list<std::shared_ptr<Frame>> frames;
    gtsam::NonlinearFactorGraph factor_graph;
    
    
  };
  
}
