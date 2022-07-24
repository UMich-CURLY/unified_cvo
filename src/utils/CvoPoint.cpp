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

    positions_.resize(pc.size());
    features_.resize(num_points_, feature_dimensions_);
    labels_.resize(num_points_, num_classes_);
    geometric_types_.resize(num_points_*2);
    for (int i = 0; i < num_points_; i++) {
      Eigen::Vector3f xyz;
      auto & p = (pc)[i];
      xyz << p.x, p.y, p.z;
      positions_[i] = xyz;

      

      for (int j = 0 ; j < FEATURE_DIMENSIONS; j++) {
        features_(i, j) = pc[i].features[j];
      }
      for (int j = 0 ; j < NUM_CLASSES; j++) {
        labels_(i, j) = pc[i].label_distribution[j];
      }

      geometric_types_[i*2] = p.geometric_type[0];
      geometric_types_[i*2+1] = p.geometric_type[1];
      if (gtype == GeometryType::SURFACE) {
        geometric_types_[i*2] = 0;
        geometric_types_[i*2+1] = 1;
      } else {
        geometric_types_[i*2] = 1;
        geometric_types_[i*2+1] = 0;
        
      }
      
      
    }
    
  }

  
  template <>
  CvoPointCloud::CvoPointCloud(const pcl::PointCloud<CvoPoint> & pc) {
    num_points_ = pc.size();
    num_classes_ = NUM_CLASSES;
    feature_dimensions_ = FEATURE_DIMENSIONS;

    positions_.resize(pc.size());
    features_.resize(num_points_, feature_dimensions_);
    labels_.resize(num_points_, num_classes_);
    geometric_types_.resize(num_points_*2);
    for (int i = 0; i < num_points_; i++) {
      Eigen::Vector3f xyz;
      auto & p = (pc)[i];
      xyz << p.x, p.y, p.z;
      positions_[i] = xyz;

      

      for (int j = 0 ; j < FEATURE_DIMENSIONS; j++) {
        features_(i, j) = pc[i].features[j];
      }
      for (int j = 0 ; j < NUM_CLASSES; j++) {
        labels_(i, j) = pc[i].label_distribution[j];
      }

      geometric_types_[i*2] = p.geometric_type[0];
      geometric_types_[i*2+1] = p.geometric_type[1];
      
    }
    
  }


  template <>
  void CvoPointCloud::export_to_pcd<CvoPoint>(pcl::PointCloud<CvoPoint> & pc)  const {

    for (int i = 0; i < num_points_; i++) {
      CvoPoint p;
      p.x = positions_[i]( 0);
      p.y = positions_[i]( 1);
      p.z = positions_[i]( 2);
      
      uint8_t b = static_cast<uint8_t>(std::min(255, (int)(features_(i,0) * 255) ) );
      uint8_t g = static_cast<uint8_t>(std::min(255, (int)(features_(i,1) * 255) ) );
      uint8_t r = static_cast<uint8_t>(std::min(255, (int)(features_(i,2) * 255)));

      for (int j = 0; j < this->feature_dimensions_; j++)
        p.features[j] = features_(i,j);

      if (this->num_classes() > 0) {
        //labels_.row(i).maxCoeff(p.label);
        p.label = labels_.row(i).maxCoeff();
        for (int j = 0; j < this->num_classes(); j++)
          p.label_distribution[j] = labels_(i,j);
      }
      p.r = r;
      p.g = g;
      p.b = b;

      p.geometric_type[0] = geometric_types_[i*2];
      p.geometric_type[1] = geometric_types_[i*2+1];
      
      pc.push_back(p);
    }
    
  }  

  

}
