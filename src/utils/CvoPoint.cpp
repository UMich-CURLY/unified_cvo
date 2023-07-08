#include "utils/CvoPoint.hpp"
//#include "utils/VoxelMap.hpp"
#include "utils/CvoPointCloud.hpp"
#include <Eigen/Dense>

namespace cvo {

  template <>
  CvoPointCloud::CvoPointCloud(const pcl::PointCloud<CvoPoint> & pc,
                                         GeometryType  gtype) {
    num_points_ = pc.size();
    num_classes_ = NUM_CLASSES;
    feature_dimensions_ = FEATURE_DIMENSIONS;

//    positions_.resize(pc.size());
//    features_.resize(num_points_, feature_dimensions_);
//    labels_.resize(num_points_, num_classes_);
//    geometric_types_.resize(num_points_*2);
    for (int i = 0; i < num_points_; i++) {
//      Eigen::Vector3f xyz;
      auto & p = (pc)[i];
      CvoPoint point(p.x, p.y, p.z);
//      xyz << p.x, p.y, p.z;
//      positions_[i] = xyz;

      

      for (int j = 0 ; j < FEATURE_DIMENSIONS; j++) {
//        features_(i, j) = pc[i].features[j];
          point.features[j] = pc[i].features[j];
      }
      //TODO:fix label
//      for (int j = 0 ; j < NUM_CLASSES; j++) {
//        point.label_distribution[j] = pc[i].label_distribution[j];
//      }

      point.geometric_type[0] = p.geometric_type[0];
      point.geometric_type[1] = p.geometric_type[1];
      if (gtype == GeometryType::SURFACE) {
        point.geometric_type[0] = 0;
        point.geometric_type[1] = 1;
      } else {
        point.geometric_type[0] = 1;
        point.geometric_type[1] = 0;
        
      }
      points_.push_back(point);
      
    }
    
  }

  
  template <>
  CvoPointCloud::CvoPointCloud(const pcl::PointCloud<CvoPoint> & pc) {
    num_points_ = pc.size();
    num_classes_ = NUM_CLASSES;
    feature_dimensions_ = FEATURE_DIMENSIONS;

//    positions_.resize(pc.size());
//    features_.resize(num_points_, feature_dimensions_);
//    labels_.resize(num_points_, num_classes_);
//    geometric_types_.resize(num_points_*2);
    for (int i = 0; i < num_points_; i++) {
//      Eigen::Vector3f xyz;
      auto & p = (pc)[i];
      CvoPoint point(p.x, p.y, p.z);
//      xyz << p.x, p.y, p.z;
//      positions_[i] = xyz;

      for (int j = 0 ; j < FEATURE_DIMENSIONS; j++) {
//        features_(i, j) = pc[i].features[j];
        point.features[j] = pc[i].features[j];
      }
      //TODO:fix label
//      for (int j = 0 ; j < NUM_CLASSES; j++) {
//        point.label_distribution[j] = pc[i].label_distribution[j];
//      }

      point.geometric_type[0] = p.geometric_type[0];
      point.geometric_type[1] = p.geometric_type[1];
      
    }
    
  }


  template <>
  void CvoPointCloud::export_to_pcd<CvoPoint>(pcl::PointCloud<CvoPoint> & pc)  const {

    for (int i = 0; i < num_points_; i++) {
      CvoPoint p;
      p.x = points_[i].x;
      p.y = points_[i].y;
      p.z = points_[i].z;
      
      uint8_t b = static_cast<uint8_t>(std::min(255, (int)(points_[i].features[0] * 255) ) );
      uint8_t g = static_cast<uint8_t>(std::min(255, (int)(points_[i].features[1] * 255) ) );
      uint8_t r = static_cast<uint8_t>(std::min(255, (int)(points_[i].features[2] * 255)));

      for (int j = 0; j < this->feature_dimensions_; j++)
        p.features[j] = points_[i].features[j];

      if (this->num_classes() > 0) {
        //labels_.row(i).maxCoeff(p.label);

        //TODO:fix label
        p.label = points_[i].label;
//        for (int j = 0; j < this->num_classes(); j++) {
//          p.label_distribution[j] = points_[i].label_distribution[j];
//        }
      }
      p.r = r;
      p.g = g;
      p.b = b;

      p.geometric_type[0] = points_[i].geometric_type[0];
      p.geometric_type[1] = points_[i].geometric_type[1];
      
      pc.push_back(p);
    }
    
  }  

  

}
