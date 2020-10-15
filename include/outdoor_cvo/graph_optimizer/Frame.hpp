#pragma once
#include <iostream>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "utils/data_type.hpp"
#include "utils/LidarPointType.hpp"
#include "utils/CvoPointCloud.hpp"
#include "utils/RawImage.hpp"
#include "utils/Calibration.hpp"
#include "graph_optimizer/RelativePose.hpp"
#include "mapping/bkioctomap.h"
namespace cvo {

  class Frame {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    Frame(int ind,
          const cv::Mat & rgb_image,
          const cv::Mat & depth_image,
          const Calibration & calib,
          const bool & is_using_rgbd);

    Frame(int ind,
          const cv::Mat & left_image,
          const cv::Mat & right_image,
          const Calibration & calib);

    Frame(int ind,
          const cv::Mat & left_image,
          const cv::Mat & right_image,
          int num_classes,
          const std::vector<float> & left_semantics,
          const Calibration & calib,
          float local_map_res=0.1);
    
    Frame(int ind,
          pcl::PointCloud<pcl::PointXYZI>::Ptr pc,
          const Calibration & calib);

    Frame(int ind,
          pcl::PointCloud<pcl::PointXYZI>::Ptr pc,
          const std::vector<int> & semantics,
          const Calibration & calib);
    
    Frame(int ind,
          pcl::PointCloud<pcl::PointXYZIR>::Ptr pc,
          const Calibration & calib);

    Frame(int ind,
          pcl::PointCloud<pcl::PointXYZIR>::Ptr pc,
          const std::vector<int> & semantics,
          const Calibration & calib);

    ~Frame();

    // public attributes
    const int id;
    const int h;
    const int w;
    const Calibration calib;


    // getters
    const CvoPointCloud & points() const  {return points_;}
    bool is_keyframe() const { return is_keyframe_; }
    const RawImage & raw_image() const { return raw_image_;}
    
    // set graph optimization results
    void set_pose_in_graph(const Eigen::Affine3f & optimized_pose) { pose_in_graph_ = optimized_pose; }
    const Eigen::Affine3f pose_in_graph() const;


    // tracking pose
    const RelativePose & tracking_pose_from_last_keyframe() const { return tracking_pose_from_last_keyframe_; }
    // set tracking result here
    void set_relative_transform_from_ref(int ref_frame_id, const Eigen::Affine3f & ref_to_curr, float inner_prod) {
      tracking_pose_from_last_keyframe_.set_relative_transform(ref_frame_id, ref_to_curr, inner_prod);
    }
    void set_relative_transform_from_ref(const RelativePose & input) {
      tracking_pose_from_last_keyframe_.set_relative_transform(input);
    }

    
    void set_keyframe(bool is_kf) {is_keyframe_ = is_kf;}


    // keyframe map operations
    void construct_map(); // for keyframe
    std::unique_ptr<CvoPointCloud> export_points_from_map() const ; 
    void add_points_to_map_from(const Frame & nonkeyframe); // for non-keyframe
    
    
  private:
    
    // for keyframes. 
    Eigen::Affine3f pose_in_graph_;
    bool is_keyframe_;

    // for all frames
    RelativePose tracking_pose_from_last_keyframe_;

    RawImage raw_image_;

    CvoPointCloud points_;

    // nullptr for non-keyframes.
    std::unique_ptr<semantic_bki::SemanticBKIOctoMap> local_map_;
    float map_resolution_;
    //    bool is_map_centroids_latest_;
    //std::unique_ptr<CvoPointCloud> map_centroids_;
  };

    
  
}
