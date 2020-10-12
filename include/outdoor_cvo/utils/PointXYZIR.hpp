#pragma once

// for the newly defined pointtype
#define PCL_NO_PRECOMPILE

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <boost/shared_ptr.hpp>
#include <pcl/impl/point_types.hpp>

#include <Eigen/Core>

namespace pcl {


  struct PointXYZIR
  {
    PCL_ADD_POINT4D;                  // preferred way of adding a XYZ+padding
    float intensity;
    std::uint16_t ring;
    PCL_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned
  } EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment
}
