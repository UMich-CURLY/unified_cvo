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
    const Eigen::Affine3f & pose_in_world() const {return pose_in_world_;}
    

    // keyframe map operations
    void construct_map();
    std::unique_ptr<CvoPointCloud> export_points_from_map() const ; 
    void add_points_to_map_from(const Frame & nonkeyframe);
    
    
  private:

    Eigen::Affine3f pose_in_world_;

    RawImage raw_image_;

    CvoPointCloud points_;

    // nullptr for non-keyframes.
    std::unique_ptr<semantic_bki::SemanticBKIOctoMap> local_map_;
    //    bool is_map_centroids_latest_;
    //std::unique_ptr<CvoPointCloud> map_centroids_;
  };

    
  
}
