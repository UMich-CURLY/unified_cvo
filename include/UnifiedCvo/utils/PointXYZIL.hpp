#pragma once

// for the newly defined pointtype
#define PCL_NO_PRECOMPILE

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
//#include <boost/shared_ptr.hpp>
#include <pcl/impl/point_types.hpp>

#include <Eigen/Core>

namespace pcl {
  struct
#ifdef __CUDACC__
  __align__(16)
#else
    alignas(16)
#endif  
  PointXYZIL
  {
    // data
    PCL_ADD_POINT4D;                 
    PCL_ADD_RGB;
    float intensity;
    int   label;
  };
}
POINT_CLOUD_REGISTER_POINT_STRUCT (pcl::PointXYZIL,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, intensity, intensity)
                                   (int, label, label)
                                   )

  

