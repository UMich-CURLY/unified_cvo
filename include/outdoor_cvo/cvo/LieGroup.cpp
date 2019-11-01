/* ----------------------------------------------------------------------------
 * Copyright 2018, Ross Hartley <m.ross.hartley@gmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   LieGroup.cpp
 *  @author Ross Hartley
 *  @brief  Source file for various Lie Group functions 
 *  @date   September 25, 2018
 **/

#include "LieGroup.h"

using namespace std;

const float TOLERANCE = 1e-10;

Eigen::Matrix3f skew(const Eigen::Vector3f& v) {
    // Convert vector to skew-symmetric matrix
    Eigen::Matrix3f M = Eigen::Matrix3f::Zero();
    M << 0, -v[2], v[1],
         v[2], 0, -v[0], 
        -v[1], v[0], 0;
        return M;
}

Eigen::Matrix3f Exp_SO3(const Eigen::Vector3f& w) {
    // Computes the vectorized exponential map for SO(3)
    Eigen::Matrix3f A = skew(w);
    float theta = w.norm();
    if (theta < TOLERANCE) {
        return Eigen::Matrix3f::Identity();
    } 
    Eigen::Matrix3f R =  Eigen::Matrix3f::Identity() + (sin(theta)/theta)*A + ((1-cos(theta))/(theta*theta))*A*A;
    return R;
}

Eigen::MatrixXf Exp_SEK3(const Eigen::VectorXf& v, float dt) {
    // Computes the vectorized exponential map for SE_K(3)
    int K = (v.size()-3)/3;
    Eigen::MatrixXf X = Eigen::MatrixXf::Identity(3+K,3+K);
    Eigen::Matrix3f R;
    Eigen::Matrix3f Jl;
    Eigen::Vector3f w = v.head(3);
    float theta = w.norm();
    Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
    if (theta < TOLERANCE) {
        R = I;
        Jl = I;
    } else {
        Eigen::Matrix3f A = skew(w);
        float theta2 = theta*theta;
        float stheta = sin(dt*theta);
        float ctheta = cos(dt*theta);
        float oneMinusCosTheta2 = (1-ctheta)/(theta2);
        Eigen::Matrix3f A2 = A*A;
        R =  I + (stheta/theta)*A + oneMinusCosTheta2*A2;
        Jl = dt*I + oneMinusCosTheta2*A + ((dt*theta-stheta)/(theta2*theta))*A2;
    }
    X.block<3,3>(0,0) = R;
    for (int i=0; i<K; ++i) {
        X.block<3,1>(0,3+i) = Jl * v.segment<3>(3+3*i);
    }
    return X;
}

Eigen::MatrixXf Adjoint_SEK3(const Eigen::MatrixXf& X) {
    // Compute Adjoint(X) for X in SE_K(3)
    int K = X.cols()-3;
    Eigen::MatrixXf Adj = Eigen::MatrixXf::Zero(3+3*K, 3+3*K);
    Eigen::Matrix3f R = X.block<3,3>(0,0);
    Adj.block<3,3>(0,0) = R;
    for (int i=0; i<K; ++i) {
        Adj.block<3,3>(3+3*i,3+3*i) = R;
        Adj.block<3,3>(3+3*i,0) = skew(X.block<3,1>(0,3+i))*R;
    }
    return Adj;
}
