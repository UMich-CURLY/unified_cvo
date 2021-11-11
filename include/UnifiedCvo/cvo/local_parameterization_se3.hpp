#pragma once

#include <ceres/local_parameterization.h>
#include "LieGroup.h"
#include <Eigen/Dense>

namespace cvo {

  typedef Eigen::Matrix<double, 3, 3, Eigen::RowMajor>  Mat3d_row ;
  typedef Eigen::Matrix<double, 4, 4, Eigen::RowMajor>  Mat4d_row ;
  typedef Eigen::Matrix<double, 3, 4, Eigen::RowMajor>  Mat34d_row ;

class LocalParameterizationSE3 : public ceres::LocalParameterization {
 public:
  virtual ~LocalParameterizationSE3() {}

  // SE3 plus operation for Ceres
  //
  //  T * exp(x)
  //
  virtual bool Plus(double const* T_raw, double const* delta_raw,
                    double* T_plus_delta_raw) const {

    
    Eigen::Map<Mat34d_row const> const T(T_raw);
    Mat4d_row T_full = Mat4d_row::Identity();
    T_full.block<3,4>(0,0) = T;
    
    Eigen::Map<Eigen::Matrix<double, 6, 1> const> const delta(delta_raw);
    Eigen::Matrix<double, 6, 1> delta_norm = delta;
    //delta_norm.head(3) = delta.head(3).normalized();
    //delta_norm.tail(3) = delta.tail(3).normalized();
    
    //std::cout<<"delta is "<<delta_norm.transpose()<<std::endl;    
    Mat34d_row exp_delta = Exp_SE3<double, Eigen::RowMajor>(delta_norm, false);
    Mat4d_row exp_delta_full = Mat4d_row::Identity();
    exp_delta_full.block<3,4>(0,0) = exp_delta;
    
    Eigen::Map<Mat34d_row> T_plus_delta(T_plus_delta_raw);
    Mat4d_row T_plus_delta_full = T_full * exp_delta_full;
    T_plus_delta = T_plus_delta_full.block<3,4>(0,0);
    return true;
  }

  // Jacobian of SE3 plus operation for Ceres
  //
  // Dx T * exp(x)  with  x=0
  //
  virtual bool ComputeJacobian(double const* T_raw,
                               double* jacobian_raw) const {
    Eigen::Map<Mat34d_row const> T(T_raw);
    Eigen::Map<Eigen::Matrix<double, 12, 6, Eigen::RowMajor> > jacobian(jacobian_raw);
    jacobian = Eigen::Matrix<double, 12, 6, Eigen::RowMajor>::Zero();
    //jacobian.block<3,3>(9,0) = T.block<3,3>(0,0);
    Eigen::Vector3d d1 = T.block<3,1>(0,0);
    Eigen::Vector3d d2 = T.block<3,1>(0,1);
    Eigen::Vector3d d3 = T.block<3,1>(0,2);
            
    jacobian(0,4) = -d3(0);
    jacobian(0,5) = d2(0);
    jacobian(4,4) = -d3(1);
    jacobian(4,5) = d2(1);
    jacobian(8,4) = -d3(2);
    jacobian(8,5) = d2(2);
    jacobian(1,3) = d3(0);
    jacobian(1,5) = -d1(0);
    jacobian(5,3) = d3(1);
    jacobian(5,5) = -d1(1);
    jacobian(9,3) = d3(2);
    jacobian(9,5) = -d1(2);
    jacobian(2,3) = -d2(0);
    jacobian(2,4) = d1(0);
    jacobian(6,3) = -d2(1);
    jacobian(6,4) = d1(1);
    jacobian(10,3)=-d2(2);
    jacobian(10,4)=d1(2);
    
    jacobian.block<1,3>(3,0) = T.block<1,3>(0,0);
    jacobian.block<1,3>(7,0) = T.block<1,3>(1,0);
    jacobian.block<1,3>(11,0) = T.block<1,3>(2,0);
    
    
    return true;
  }

  /*
  virtual bool MultiplyByJacobian(const double * T_raw, const int num_rows, const double *global_matrix, double *local_matrix) const {
    Eigen::Map<Mat34d_row const> T(T_raw);

    Eigen::Map<Eigen::Matrix<double, 12, 6, Eigen::RowMajor> > local_jacobian(local_matrix);
    Eigen::Map<Eigen::Matrix<double, 3, 12, Eigen::RowMajor> > global_jacobian(global_matrix);

    
    
    }*/

  virtual int GlobalSize() const { return 12; }

  virtual int LocalSize() const { return 6; }
};

}  // namespace Sophus

