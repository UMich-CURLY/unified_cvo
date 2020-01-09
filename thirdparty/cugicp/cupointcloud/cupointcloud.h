#pragma once

#include <memory>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/swap.h>

#include "point_types.h"


#define THREADNUM 128

#define CUDA_STACK 50
#include "cupointcloud.h"

#define cudaSafe(ans) \
  { gpuAssert((ans), __FILE__, __LINE__, true); }



namespace perl_registration  {

template <typename PointT>
class cuPointCloud {
 private:
 public:
  typedef PointT PointType;
  typedef typename thrust::device_vector<PointT> DeviceVectorType;
  typedef typename thrust::host_vector<PointT, Eigen::aligned_allocator<PointT>>
      HostVectorType;
  typedef typename std::shared_ptr<cuPointCloud<PointT>> SharedPtr;
  typedef typename std::shared_ptr<const cuPointCloud<PointT>> SharedConstPtr;

  /* public members */
  DeviceVectorType points;

  /* constant compatibility typedefs */
  typedef PointT value_type;
  typedef PointT &reference;
  typedef const PointT &const_reference;
  typedef typename DeviceVectorType::difference_type difference_type;
  typedef typename DeviceVectorType::size_type size_type;

  /* iterators */
  typedef typename DeviceVectorType::iterator iterator;
  typedef typename DeviceVectorType::const_iterator const_iterator;
  inline iterator begin() { return (points.begin()); }
  inline iterator end() { return (points.end()); }
  inline const_iterator cbegin() { return (points.cbegin()); }
  inline const_iterator cend() { return (points.cend()); }

  /* capacity */
  inline size_type size() { return (points.size()); }
  inline size_type max_size() { return (points.max_size()); }
  inline void reserve(size_type n) { points.reserve(n); }
  inline bool empty() const { return (points.empty()); }

  /* constructors */
  cuPointCloud() {}

  template <typename ConstructorPointType>
  cuPointCloud(pcl::PointCloud<ConstructorPointType> &pc)
      : points(pc.begin(), pc.end()) {}

  cuPointCloud(cuPointCloud &pc) { *this = pc; }

  cuPointCloud(size_t n_, const PointType &value_ = PointType())
      : points(n_, value_) {}

  ~cuPointCloud() {}

  /** \breif Swap point clouds */
  inline void swap(cuPointCloud &rhs) {
    thrust::swap(this->points, rhs.points);
  }
};

}  // namespace perl_registration
