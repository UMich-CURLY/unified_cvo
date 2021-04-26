#pragma once

#include "PointSegmentedDistribution.hpp"

extern template struct pcl::PointSegmentedDistribution<FEATURE_DIMENSIONS, NUM_CLASSES>;

namespace cvo {
  typedef pcl::PointSegmentedDistribution<FEATURE_DIMENSIONS, NUM_CLASSES> CvoPoint;
}

POINT_CLOUD_REGISTER_POINT_STRUCT (cvo::CvoPoint,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, r, r)
                                   (float, g, g)
                                   (float, b, b)
                                   (float[FEATURE_DIMENSIONS], features, features )
                                   (int, label, label)
                                   (float[NUM_CLASSES], label_distribution, label_distribution)
                                   (float[3], normal, normal)
                                   (float[9], covariance, covariance)
                                   (float[3], cov_eigenvalues, cov_eigenvalues)
                                   )




