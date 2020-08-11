#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <gtsam/geometry/Pose3.h>
#include "utils/data_type.hpp"
namespace cvo {

  inline
  gtsam::Pose3 affine3f_to_pose3(const Eigen::Affine3f & aff) {
    //auto aff_mat = aff.matrix();
    Mat33 rot = aff.matrix().block(0,0,3,3).cast<double>();
    Eigen::Quaterniond q_eigen (rot);
    q_eigen.normalize();
    gtsam::Quaternion q_gtsam(q_eigen.w(),
                              q_eigen.x(), q_eigen.y(), q_eigen.z());
    gtsam::Vector3 t_gtsam;
    t_gtsam << (double)aff.translation()(0), (double)aff.translation()(1), (double)aff.translation()(2);
    gtsam::Pose3 pose(q_gtsam, t_gtsam);
    return pose;
  }
  
}
