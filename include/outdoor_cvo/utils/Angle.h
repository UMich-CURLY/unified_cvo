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

#ifndef LOAM_ANGLE_H
#define LOAM_ANGLE_H

#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>


namespace cvo {


/** \brief Class for holding an angle.
 *
 * This class provides buffered access to sine and cosine values to the represented angular value.
 */
class Angle {
public:
  Angle()
      : _radian(0.0),
        _cos(1.0),
        _sin(0.0) {}

  Angle(float radValue)
      : _radian(radValue),
        _cos(std::cos(radValue)),
        _sin(std::sin(radValue)) {}

  Angle(const Angle &other)
      : _radian(other._radian),
        _cos(other._cos),
        _sin(other._sin) {}

  void operator=(const Angle &rhs) {
    _radian = (rhs._radian);
    _cos = (rhs._cos);
    _sin = (rhs._sin);
  }

  void operator+=(const float &radValue) { *this = (_radian + radValue); }

  void operator+=(const Angle &other) { *this = (_radian + other._radian); }

  void operator-=(const float &radValue) { *this = (_radian - radValue); }

  void operator-=(const Angle &other) { *this = (_radian - other._radian); }

  Angle operator-() const {
    Angle out;
    out._radian = -_radian;
    out._cos = _cos;
    out._sin = -(_sin);
    return out;
  }

  float rad() const { return _radian; }

  float deg() const { return float(_radian * 180 / M_PI); }

  float cos() const { return _cos; }

  float sin() const { return _sin; }

private:
  float _radian;    ///< angle value in radian
  float _cos;       ///< cosine of the angle
  float _sin;       ///< sine of the angle
};

} // end namespace loam

#endif //LOAM_ANGLE_H