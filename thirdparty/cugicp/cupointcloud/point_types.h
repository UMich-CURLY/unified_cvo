#pragma once

#define PCL_NO_PRECOMPILE

#include <pcl/point_types.h>
#include <Eigen/Core>

namespace perl_registration {

struct cuPointXYZ : public pcl::_PointXYZ {
  __host__ __device__ __forceinline__ cuPointXYZ(const pcl::_PointXYZ &p) {
    x = p.x;
    y = p.y;
    z = p.z;
    data[3] = 1.0f;
  }

  __host__ __device__ __forceinline__ cuPointXYZ() {
    x = y = z = 0.0f;
    data[3] = 1.0f;
  }

  __host__ __device__ __forceinline__ cuPointXYZ(const float &_x,
                                                 const float &_y,
                                                 const float &_z) {
    x = _x;
    y = _y;
    z = _z;
    data[3] = 1.0f;
  }
  __host__ __device__ __forceinline__ Eigen::Map<Eigen::Vector3f>
  cuVector3fMap() {
    return (Eigen::Vector3f::Map(data));
  }

  __host__ __device__ __forceinline__ Eigen::Map<const Eigen::Vector3f>
  cuConstVector3fMap() {
    return (Eigen::Map<const Eigen::Vector3f>(data));
  }

  __host__ __device__ __forceinline__ Eigen::Vector3f toVec() {
    Eigen::Vector3f out(x, y, z);
    return out;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct cuPointXYZI : public pcl::_PointXYZ {
  float intensity;
  __host__ __device__ __forceinline__ cuPointXYZI(const pcl::_PointXYZ &p) {
    x = p.x;
    y = p.y;
    z = p.z;
    data[3] = 1.0f;
    intensity = 0.0f;
  }

  __host__ __device__ __forceinline__ cuPointXYZI() {
    x = y = z = 0.0f;
    data[3] = 1.0f;
    intensity = 0.0f;
  }

  __host__ __device__ __forceinline__ cuPointXYZI(const float &_x,
                                                  const float &_y,
                                                  const float &_z,
                                                  const float &_intensity) {
    x = _x;
    y = _y;
    z = _z;
    data[3] = 1.0f;
    intensity = _intensity;
  }
  __host__ __device__ __forceinline__ Eigen::Map<Eigen::Vector3f>
  cuVector3fMap() {
    return (Eigen::Vector3f::Map(data));
  }

  __host__ __device__ __forceinline__ Eigen::Map<const Eigen::Vector3f>
  cuConstVector3fMap() {
    return (Eigen::Map<const Eigen::Vector3f>(data));
  }

  __host__ __device__ __forceinline__ Eigen::Vector3f toVec() {
    Eigen::Vector3f out(x, y, z);
    return out;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <size_t N>
struct cuPointXYZP : public pcl::_PointXYZ {
  float probability[N];
  __host__ __device__ __forceinline__ cuPointXYZP(const pcl::_PointXYZ &p) {
    x = p.x;
    y = p.y;
    z = p.z;
    data[3] = 1.0f;
    for (size_t i = 0; i < N; i++) {
      probability[i] = 0;
    }
  }

  __host__ __device__ __forceinline__ cuPointXYZP(const cuPointXYZP<N> &p) {
    x = p.x;
    y = p.y;
    z = p.z;
    data[3] = 1.0f;
    for (size_t i = 0; i < N; i++) {
      probability[i] = p.probability[i];
    }
  }

  __host__ __device__ __forceinline__ cuPointXYZP() {
    x = y = z = 0.0f;
    data[3] = 1.0f;
    for (size_t i = 0; i < N; i++) {
      probability[i] = 0;
    }
  }

  __host__ __device__ __forceinline__ cuPointXYZP(const float &_x,
                                                  const float &_y,
                                                  const float &_z) {
    x = _x;
    y = _y;
    z = _z;
    data[3] = 1.0f;
    for (size_t i = 0; i < N; i++) {
      probability[i] = 0;
    }
  }

  __host__ __device__ __forceinline__ Eigen::Map<Eigen::Vector3f>
  cuVector3fMap() {
    return (Eigen::Vector3f::Map(data));
  }

  __host__ __device__ __forceinline__ Eigen::Map<const Eigen::Vector3f>
  cuConstVector3fMap() {
    return (Eigen::Map<const Eigen::Vector3f>(data));
  }

  __host__ __device__ __forceinline__ Eigen::Map<Eigen::Matrix<float, N, 1>>
  cuProbabilityMap() {
    return (Eigen::Map<Eigen::Matrix<float, N, 1>>(probability));
  }

  __host__ __device__ __forceinline__
      Eigen::Map<const Eigen::Matrix<float, N, 1>>
      cuConstProbabilityMap() {
    return (Eigen::Map<const Eigen::Matrix<float, N, 1>>(probability));
  }
};

}  // namespace perl_registration
