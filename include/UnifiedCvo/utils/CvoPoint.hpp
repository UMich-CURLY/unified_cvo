#pragma once

#include "utils/PointSegmentedDistribution.hpp"

extern template struct pcl::PointSegmentedDistribution<FEATURE_DIMENSIONS, NUM_CLASSES>;

namespace cvo {
  typedef pcl::PointSegmentedDistribution<FEATURE_DIMENSIONS, NUM_CLASSES> CvoPoint;
}

POINT_CLOUD_REGISTER_POINT_STRUCT (cvo::CvoPoint,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, rgb, rgb)
                                   (float[FEATURE_DIMENSIONS], features, features )
                                   (int, label, label)
                                   (float[NUM_CLASSES], label_distribution, label_distribution)
                                   (float[2], geometric_type, geometric_type)
                                   (float[3], normal, normal)
                                   (float[9], covariance, covariance)
                                   (float[3], cov_eigenvalues, cov_eigenvalues)
                                   )

template <typename PointT> struct CvoPointToPCL {};
template <> struct CvoPointToPCL<pcl::PointSegmentedDistribution<1, NUM_CLASSES>> {
  using type = pcl::PointXYZI;
};
template <> struct CvoPointToPCL<pcl::PointSegmentedDistribution<5, NUM_CLASSES>> {
  using type = pcl::PointXYZRGB;
};
template <> struct CvoPointToPCL<pcl::PointSegmentedDistribution<3, NUM_CLASSES>> {
  using type = pcl::PointXYZRGB;
};
