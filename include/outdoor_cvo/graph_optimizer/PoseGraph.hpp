#pragma once
#include <iostream>
#include <list>
#include <queue>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/PriorFactor.h>

#include <Eigen/Dense>
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
    
    Eigen::Affine3f compute_frame_pose_in_graph(std::shared_ptr<Frame> frame);

    void write_trajectory(std::string filename);

    
  private:


    float track_new_frame(std::shared_ptr<Frame> new_frame,
                           // output
                           bool & is_keyframe);


    
    // gtsam pose graph optimization. called by add_new_frame
    void pose_graph_optimize(std::shared_ptr<Frame> new_frame);
    void init_pose_graph(std::shared_ptr<Frame> new_frame);
    void update_optimized_poses_to_frames();
    
    std::vector<std::shared_ptr<Frame>> all_frames_since_last_keyframe_;
    std::queue<std::shared_ptr<Frame>> last_two_frames_;
    std::unordered_map<int, std::shared_ptr<Frame>> id2keyframe_;
    std::vector<std::shared_ptr<Frame>> keyframes_;
    std::vector<RelativePose> tracking_relative_transforms_;
    

    // factor graph
    gtsam::NonlinearFactorGraph factor_graph_;
    std::unique_ptr<gtsam::ISAM2> isam2_;
    gtsam::Values graph_values_;
    std::unordered_map<gtsam::Key, int> key2id_;

    // tracking
    cvo cvo_align_;
    
  };
  
}
