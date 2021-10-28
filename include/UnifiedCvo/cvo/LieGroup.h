#ifndef __LIEGROUP_H__
#define __LIEGROUP_H__

#include <Eigen/Dense>
#include <iostream>


namespace cvo {

  extern const float TOLERANCE;

  //Eigen::Matrix<float, 3, 3> skew(const Eigen::Vector3f& v);

  template <typename T, int RC_MAJOR>
  Eigen::Matrix<T, 3, 3, RC_MAJOR> skew(const Eigen::Matrix<T, 3, 1, Eigen::ColMajor>& v);

  Eigen::Vector3f unskew(const Eigen::Matrix3f& M);
  Eigen::Matrix4f hat2(const Eigen::VectorXf& x);    // hat: R^6 -> se(3)
  Eigen::VectorXf wedge(const Eigen::Matrix4f& X); 

  template <typename T, int RC_MAJOR>
  Eigen::Matrix<T, 3, 3, RC_MAJOR> LeftJacobian_SO3(const Eigen::Matrix<T, 3, 1, Eigen::ColMajor>& w);

  //Eigen::Matrix3f LeftJacobian_SO3(const Eigen::Vector3f& w);
  Eigen::MatrixXf LeftJacobianInverse_SO3(const Eigen::Vector3f& w);
  Eigen::MatrixXf LeftJacobian_SE3(Eigen::VectorXf& v);
  Eigen::MatrixXf RightJacobian_SE3(Eigen::VectorXf& v);
  Eigen::MatrixXf RightJacobianInverse_SE3(Eigen::VectorXf& v);
  Eigen::Vector3f Log_SO3(const Eigen::Matrix3f& M);
  Eigen::VectorXf Log_SE3(const Eigen::MatrixXf& X);

  template <typename T, int RC_MAJOR>
  Eigen::Matrix<T, 3, 3, RC_MAJOR> Exp_SO3(const Eigen::Matrix<T, 3, 1>& w);

  //Eigen::Matrix3f Exp_SO3(const Eigen::Vector3f& w);
  Eigen::MatrixXf Exp_SE3(const Eigen::VectorXf& v);

  template <typename T, int RC_MAJOR>
  Eigen::Matrix<T, 3, 4, RC_MAJOR> Exp_SE3(const Eigen::Matrix<T, 6, 1>& v, bool is_wu=true);

  Eigen::MatrixXf Exp_SEK3(const Eigen::VectorXf& v, float dt);

  Eigen::Matrix<float,3,4> Exp_SEK3(const Eigen::Matrix<float, 6,1>& v, float dt);

  Eigen::MatrixXf Adjoint_SEK3(const Eigen::MatrixXf& X);
  Eigen::VectorXcf poly_solver(const Eigen::VectorXf& coef);
  Eigen::VectorXcd poly_solver(const Eigen::VectorXd& coef);
  double dist_se3(const Eigen::Matrix3f& R, const Eigen::Vector3f& T);
  double dist_se3(const Eigen::Matrix3d& R, const Eigen::Vector3d& T);
  Eigen::Vector3cd poly_solver_order3(const Eigen::Matrix<double, 4, 1, Eigen::DontAlign>& coef);
  Eigen::Vector3cf poly_solver_order3(const Eigen::Vector4f& coef);



  // spetializations
  extern template
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> skew<double, Eigen::RowMajor>(const Eigen::Matrix<double, 3, 1, Eigen::ColMajor>& v);
  extern template
  Eigen::Matrix<float, 3, 3, Eigen::ColMajor> skew<float, Eigen::ColMajor>(const Eigen::Matrix<float, 3, 1, Eigen::ColMajor>& v);
  
  extern template 
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> LeftJacobian_SO3<double, Eigen::RowMajor>(const Eigen::Matrix<double, 3, 1>& w);
  extern template
  Eigen::Matrix<float, 3, 3, Eigen::ColMajor> LeftJacobian_SO3<float, Eigen::ColMajor>(const Eigen::Matrix<float, 3, 1>& w);

  extern template 
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Exp_SO3<double, Eigen::RowMajor>(const Eigen::Matrix<double, 3, 1>& w);
  extern template 
  Eigen::Matrix<float, 3, 3, Eigen::ColMajor> Exp_SO3<float, Eigen::ColMajor>(const Eigen::Matrix<float, 3, 1>& w);

 

  
}
#endif
