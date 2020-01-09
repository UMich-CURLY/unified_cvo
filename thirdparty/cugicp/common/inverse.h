#pragma once

namespace perl_registration {

__host__ __device__ __forceinline__ Eigen::Matrix3f Inverse(
    const Eigen::Matrix3f& m) {
  float det = m(0, 0) * (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2)) -
              m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
              m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));

  float invdet = 1 / det;

  Eigen::Matrix3f minv;  // inverse of matrix m
  minv(0, 0) = (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2)) * invdet;
  minv(0, 1) = (m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2)) * invdet;
  minv(0, 2) = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) * invdet;
  minv(1, 0) = (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) * invdet;
  minv(1, 1) = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) * invdet;
  minv(1, 2) = (m(1, 0) * m(0, 2) - m(0, 0) * m(1, 2)) * invdet;
  minv(2, 0) = (m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1)) * invdet;
  minv(2, 1) = (m(2, 0) * m(0, 1) - m(0, 0) * m(2, 1)) * invdet;
  minv(2, 2) = (m(0, 0) * m(1, 1) - m(1, 0) * m(0, 1)) * invdet;

  return minv;
}

}  // namespace perl_registration
