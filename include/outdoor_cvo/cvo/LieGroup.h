/* ----------------------------------------------------------------------------
 * Copyright 2018, Ross Hartley
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   LieGroup.h
 *  @author Ross Hartley
 *  @brief  Header file for various Lie Group functions 
 *  @date   September 25, 2018
 **/

#ifndef LIEGROUP_H
#define LIEGROUP_H 
#include <Eigen/Dense>
#include <iostream>


extern const float TOLERANCE;

Eigen::Matrix3f skew(const Eigen::Vector3f& v);
Eigen::Matrix3f Exp_SO3(const Eigen::Vector3f& w);
Eigen::MatrixXf Exp_SEK3(const Eigen::VectorXf& v, float dt);
Eigen::MatrixXf Adjoint_SEK3(const Eigen::MatrixXf& X);

#endif 
