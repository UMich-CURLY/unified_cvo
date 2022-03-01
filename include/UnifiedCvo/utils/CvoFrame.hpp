#pragma once
//#include "utils/PointSegmentedDistribution.hpp"
#include <Eigen/Dense>
#include <cstdlib>

namespace cvo {
  class CvoPointCloud;
  
  struct CvoFrame {
    const CvoPointCloud * points; // no ownership
    double pose_vec[12]; // 3x4 row-order matrix. [R t]


    typedef std::shared_ptr<CvoFrame> Ptr;
    CvoFrame(const CvoPointCloud * pts,
             const double poses[12]) : points(pts) {
      memcpy(pose_vec, poses, 12*sizeof(double));
    }
  };

}
