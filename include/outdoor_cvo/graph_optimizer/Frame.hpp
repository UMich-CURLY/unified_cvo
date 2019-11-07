#pragma once
#include <iostream>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "utils/data_type.hpp"
#include "utils/CvoPointCloud.hpp"
#include "utils/RawImage.hpp"
#include "utils/Calibration.hpp"
#include "mapping/bkioctomap.h"
namespace cvo {

  // the relative pose computed when running cvo
  class RelativePose {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    RelativePose(int curr_id):
      curr_frame_id_(curr_id) { ref_frame_id_ = -1; }
    
    RelativePose(int curr_id, int ref_id, const Eigen::Affine3f & ref_to_curr) :
      curr_frame_id_(curr_id), ref_frame_id_(ref_id), ref_frame_to_curr_frame_(ref_to_curr),
      cvo_inner_product_(0){}

    void set_relative_transform( int ref_id, const Eigen::Affine3f & ref_to_curr, float inner_prod) {
      ref_frame_id_ = ref_id;
      ref_frame_to_curr_frame_ = ref_to_curr;
      cvo_inner_product_ = inner_prod;
    }

    int curr_frame_id() const {return curr_frame_id_;}
    int ref_frame_id() const {return ref_frame_id_;}
    float cvo_inner_product() const {return cvo_inner_product_;}
    const Eigen::Affine3f & ref_frame_to_curr_frame() const {return ref_frame_to_curr_frame_;}
  private:
    
    const int curr_frame_id_;
    int ref_frame_id_;
    Eigen::Affine3f ref_frame_to_curr_frame_;
    float cvo_inner_product_;
  };

  class Frame {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Frame(int ind,
          const cv::Mat & left_image,
          const cv::Mat & right_image,
          const Calibration & calib);

    Frame(int ind,
          const cv::Mat & left_image,
          const cv::Mat & right_image,
          int num_classes,
          const std::vector<float> & left_semantics,
          const Calibration & calib);
    
    
    ~Frame();

    // public attributes
    const int id;
    const int h;
    const int w;
    const Calibration calib;

    // getters
    const CvoPointCloud & points() const  {return points_;}
    const RawImage & raw_image() const { return raw_image_;}
    const Eigen::Affine3f & pose_in_graph() const {return pose_in_graph_;}
    const RelativePose & tracking_relative_transform() const { return tracking_relative_transform_; }

    // set tracking result here
    void set_relative_transform(int ref_frame_id, const Eigen::Affine3f & ref_to_curr, float inner_prod) {
      tracking_relative_transform_.set_relative_transform(ref_frame_id, ref_to_curr, inner_prod);
    }

    // set graph optimization results
    void set_pose_in_graph(const Eigen::Affine3f & optimized_pose) {
      pose_in_graph_ = optimized_pose;
    }

    // keyframe map operations
    void construct_map();
    std::unique_ptr<CvoPointCloud> export_points_from_map() const ; 
    void add_points_to_map_from(const Frame & nonkeyframe);
    
    
  private:

    // the pose obtained from pose graph
    Eigen::Affine3f pose_in_graph_;
    RelativePose tracking_relative_transform_;

    RawImage raw_image_;

    CvoPointCloud points_;

    // nullptr for non-keyframes.
    std::unique_ptr<semantic_bki::SemanticBKIOctoMap> local_map_;
    //    bool is_map_centroids_latest_;
    //std::unique_ptr<CvoPointCloud> map_centroids_;
  };

    
  
}
