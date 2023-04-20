#pragma once

#include <Eigen/Dense>

namespace cvo {
  
  
  double dist_se3_cpu(const Eigen::Matrix<double, 4,4, Eigen::DontAlign> & m );
  double dist_se3_cpu(const Eigen::Matrix<double, 4,4, Eigen::ColMajor> & m );
}
