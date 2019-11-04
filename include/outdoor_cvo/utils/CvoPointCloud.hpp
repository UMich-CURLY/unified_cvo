#pragma once
#include <string>
#include <tbb/tbb.h>
#include "utils/data_type.hpp"
#include "mapping/bkioctomap.h"

namespace semantic_bki {
  class SemanticBKIOctoMap;
}

namespace cvo {

  class CvoPointCloud{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    CvoPointCloud(const cv::Mat & left_image,
                  const cv::Mat & right_image);
    CvoPointCloud(const semantic_bki::SemanticBKIOctoMap& map,
                  int num_semantic_class);
    CvoPointCloud();
    CvoPointCloud(const cv::Mat & left_image,
                  const cv::Mat & right_image,
                  int num_semantic_class,
                  const cv::Mat & semantic_left_img);
    ~CvoPointCloud();

    int read_cvo_pointcloud_from_file(const std::string & filename);
    void transform(const Eigen::Matrix4f& pose);
    void write_to_color_pcd(const std::string & name) const;
    void write_to_label_pcd(const std::string & name) const;

    // getters
    int num_points() const {return num_points_;}
    int num_classes() const {return num_classes_;}
    const ArrayVec3f & positions() const {return positions_;}
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> & labels() const { return labels_;}
    const Eigen::Matrix<float, Eigen::Dynamic, 5> & features() const {return features_;}
    
  private:
    int num_points_;
    int num_classes_;
    
    ArrayVec3f positions_;  // points position. x,y,z
    Eigen::Matrix<float, Eigen::Dynamic, 5> features_;   // rgb, gradient in [0,1]
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> labels_; // number of points by number of classes  
    // 0. building 1. sky 2. road
    // 3. vegetation 4. sidewalk 5. car 6. pedestrian
    // 7. cyclist 8. signate 9. fence 10. pole

  };

  // for historical reasons
  typedef CvoPointCloud point_cloud;
}
