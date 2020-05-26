// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software without
//    specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,OR
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

#ifndef LOAM_VECTOR3_H
#define LOAM_VECTOR3_H


#include <pcl/point_types.h>


namespace cvo {

/** \brief Vector4f decorator for convenient handling.
 *
 */
class Vector3 : public Eigen::Vector4f {
public:
  Vector3(float x, float y, float z)
      : Eigen::Vector4f(x, y, z, 0) {}

  Vector3(void)
      : Eigen::Vector4f(0, 0, 0, 0) {}

  template<typename OtherDerived>
  Vector3(const Eigen::MatrixBase <OtherDerived> &other)
      : Eigen::Vector4f(other) {}

  Vector3(const pcl::PointXYZI &p)
      : Eigen::Vector4f(p.x, p.y, p.z, 0) {}

  template<typename OtherDerived>
  Vector3 &operator=(const Eigen::MatrixBase <OtherDerived> &rhs) {
    this->Eigen::Vector4f::operator=(rhs);
    return *this;
  }

  Vector3 &operator=(const pcl::PointXYZ &rhs) {
    x() = rhs.x;
    y() = rhs.y;
    z() = rhs.z;
    return *this;
  }

  Vector3 &operator=(const pcl::PointXYZI &rhs) {
    x() = rhs.x;
    y() = rhs.y;
    z() = rhs.z;
    return *this;
  }

  float x() const { return (*this)(0); }

  float y() const { return (*this)(1); }

  float z() const { return (*this)(2); }

  float &x() { return (*this)(0); }

  float &y() { return (*this)(1); }

  float &z() { return (*this)(2); }

  // easy conversion
  operator pcl::PointXYZI() {
    pcl::PointXYZI dst;
    dst.x = x();
    dst.y = y();
    dst.z = z();
    dst.intensity = 0;
    return dst;
  }
};

} // end namespace loam

#endif //LOAM_VECTOR3_H