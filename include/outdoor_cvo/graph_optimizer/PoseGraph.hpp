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
#include "graph_optimizer/RelativePose.hpp"
namespace cvo {
  class PoseGraph {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    PoseGraph(bool is_f2f = true,
              bool use_sliding_window = false);
    ~PoseGraph();

    // cvo_align and keyframe 
    void add_new_frame(std::shared_ptr<Frame> new_frame);
    
    Eigen::Affine3f compute_frame_pose_in_graph(std::shared_ptr<Frame> frame);

    void write_trajectory(std::string filename) ;

    
  private:

    RelativePose track_from_last_frame(std::shared_ptr<Frame> new_frame) ;
    RelativePose track_from_last_keyframe(std::shared_ptr<Frame> new_frame) ;
    RelativePose tracking_from_last_keyframe_map(std::shared_ptr<Frame> new_frame) ;
    bool is_tracking_bad(float inner_product) const;
    float track_new_frame(std::shared_ptr<Frame> new_frame,
                           // output
                           bool & is_keyframe);
    bool decide_new_keyframe(std::shared_ptr<Frame> new_frame,
                             const Aff3f & pose_from_last_keyframe,
                             float & inner_product_from_kf);

    // for f2f
    Eigen::Affine3f compute_tracking_pose_from_last_keyframe(const Eigen::Affine3f & ref_to_new,
                                                             std::shared_ptr<Frame> tracking_ref
                                                             // output
                                                             //float & inner_product_from_kf
                                                             ) ;
    
    // gtsam pose graph optimization. called by add_new_frame
    void pose_graph_optimize(std::shared_ptr<Frame> new_frame);
    void init_pose_graph(std::shared_ptr<Frame> new_frame);
    void update_optimized_poses_to_frames();

    // tracking data
    std::vector<std::shared_ptr<Frame>> all_frames_since_last_keyframe_;
    std::queue<std::shared_ptr<Frame>> last_two_frames_;

    // sliding window
    std::unordered_map<int, std::shared_ptr<Frame>> id2keyframe_;
    std::list<std::shared_ptr<Frame>> keyframes_; // sliding window

    // for keyframes:  after marginalization, the pose wrt  the first frame
    // for non-keyframe:  relative to the latest keyframe
    std::vector<RelativePose> tracking_relative_transforms_;
    // recording keyframes ids, and their poses


    

    // factor graph
    gtsam::NonlinearFactorGraph factor_graph_;
    std::unique_ptr<gtsam::ISAM2> isam2_;
    gtsam::Values graph_values_;
    std::unordered_map<gtsam::Key, int> key2id_;
    bool using_sliding_window_;

    // tracking
    cvo cvo_align_;
    bool is_f2f_; // using frame to frame or keyframe to frame to init the pose
    
  };
  
}
