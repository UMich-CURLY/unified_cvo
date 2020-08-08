/* ----------------------------------------------------------------------------
 * Copyright 2018, Ross Hartley, Tzu-yuan LIn
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   LieGroup.h
 *  @author Ross Hartley, Tzu-yuan Lin
 *  @brief  Header file for various Lie Group functions 
 *  @date   September 18, 2018
 **/
#ifndef __LIEGROUP_H__
#define __LIEGROUP_H__

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>


extern const float TOLERANCE;

Eigen::Matrix3f skew(const Eigen::Vector3f& v);
Eigen::Vector3f unskew(const Eigen::Matrix3f& M);
Eigen::Matrix4f hat2(const Eigen::VectorXf& x);    // hat: R^6 -> se(3)
Eigen::VectorXf wedge(const Eigen::Matrix4f& X); 
Eigen::Matrix3f LeftJacobian_SO3(const Eigen::Vector3f& w);
Eigen::MatrixXf LeftJacobianInverse_SO3(const Eigen::Vector3f& w);
Eigen::MatrixXf LeftJacobian_SE3(Eigen::VectorXf& v);
Eigen::MatrixXf RightJacobian_SE3(Eigen::VectorXf& v);
Eigen::MatrixXf RightJacobianInverse_SE3(Eigen::VectorXf& v);
Eigen::Vector3f Log_SO3(const Eigen::Matrix3f& M);
Eigen::VectorXf Log_SE3(const Eigen::MatrixXf& X);
Eigen::Matrix3f Exp_SO3(const Eigen::Vector3f& w);
Eigen::MatrixXf Exp_SE3(const Eigen::VectorXf& v);
Eigen::MatrixXf Exp_SEK3(const Eigen::VectorXf& v, float dt);

Eigen::Matrix<float,3,4> Exp_SEK3(const Eigen::Matrix<float, 6,1>& v, float dt);

Eigen::MatrixXf Adjoint_SEK3(const Eigen::MatrixXf& X);
Eigen::VectorXcf poly_solver(const Eigen::VectorXf& coef);
double dist_se3(const Eigen::Matrix3f& R, const Eigen::Vector3f& T);
double dist_se3(const Eigen::Matrix3d& R, const Eigen::Vector3d& T);
Eigen::Vector3cd poly_solver_order3(const Eigen::Vector4d& coef);
Eigen::Vector3cf poly_solver_order3(const Eigen::Vector4f& coef);
#endif
