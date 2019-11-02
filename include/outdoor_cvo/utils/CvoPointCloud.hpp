#pragma once
#include "utils/data_type.hpp"
namespace cvo {


  struct CvoPointCloud{
    //public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    CvoPointCloud(const cv::Mat & left_image,
                  const cv::Mat & right_image);
    CvoPointCloud(const cv::Mat & left_image,
                  const cv::Mat & right_image,
                  int num_semantic_class,
                  const cv::Mat & semantic_left_img);
    ~CvoPointCloud();
    
    int num_points;
    int num_classes;
    
    ArrayVec3f positions;  // points position. x,y,z
    Eigen::Matrix<float, Eigen::Dynamic, 5> features;   // rgb, gradient
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> labels; // number of points by number of classes  
    // 0. building 1. sky 2. road
    // 3. vegetation 4. sidewalk 5. car 6. pedestrian
    // 7. cyclist 8. signate 9. fence 10. pole

  };

  // for historical reasons
  typedef CvoPointCloud point_cloud;
}
