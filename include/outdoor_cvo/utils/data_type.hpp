/* ----------------------------------------------------------------------------
 * Copyright 2019, Tzu-yuan Lin <tzuyuan@umich.edu>, Maani Ghaffari <maanigj@umich.edu>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   data_type.h
 *  @author Tzu-yuan Lin, Maani Ghaffari 
 *  @brief  Data type definition
 *  @date   August 4, 2019
 **/
#pragma once

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Dense>
#include<Eigen/StdVector>
#include <opencv2/opencv.hpp>
#include <tbb/concurrent_vector.h>
#include <sophus/se3.hpp>
//#include "util/settings.h"
//#define PYR_LEVELS 3

namespace cvo{


#define SSEE(val,idx) (*(((float*)&val)+idx))


#define MAX_RES_PER_POINT 8
#define NUM_THREADS 6


 
  template<typename T>
  inline double todouble(T x) {
    return static_cast<double>(x);
  }


  typedef Sophus::SE3d SE3;
  //typedef Sophus::Sim3d Sim3;
  typedef Sophus::SO3d SO3;



#define CPARS 4
  typedef Eigen::Affine3f Aff3f;
  typedef Eigen::Affine3d Aff3d;
  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_row;
  typedef Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> VecXf_row;
  typedef Eigen::Matrix<float, 1, 5, Eigen::RowMajor> Vec5f_row;
  typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatXX;
  typedef Eigen::Matrix<double,CPARS,CPARS> MatCC;
#define MatToDynamic(x) MatXX(x)



  typedef Eigen::Matrix<double,CPARS,10> MatC10;
  typedef Eigen::Matrix<double,10,10> Mat1010;
  typedef Eigen::Matrix<double,13,13> Mat1313;

  typedef Eigen::Matrix<double,8,10> Mat810;
  typedef Eigen::Matrix<double,8,3> Mat83;
  typedef Eigen::Matrix<double,6,6> Mat66;
  typedef Eigen::Matrix<double,5,3> Mat53;
  typedef Eigen::Matrix<double,4,3> Mat43;
  typedef Eigen::Matrix<double,4,2> Mat42;
  typedef Eigen::Matrix<double,3,3> Mat33;
  typedef Eigen::Matrix<double,2,2> Mat22;
  typedef Eigen::Matrix<double,8,CPARS> Mat8C;
  typedef Eigen::Matrix<double,CPARS,8> MatC8;
  typedef Eigen::Matrix<float,8,CPARS> Mat8Cf;
  typedef Eigen::Matrix<float,CPARS,8> MatC8f;

  typedef Eigen::Matrix<double,8,8> Mat88;
  typedef Eigen::Matrix<double,7,7> Mat77;

  typedef Eigen::Matrix<double,CPARS,1> VecC;
  typedef Eigen::Matrix<float,CPARS,1> VecCf;
  typedef Eigen::Matrix<double,13,1> Vec13;
  typedef Eigen::Matrix<double,10,1> Vec10;
  typedef Eigen::Matrix<double,9,1> Vec9;
  typedef Eigen::Matrix<double,8,1> Vec8;
  typedef Eigen::Matrix<double,7,1> Vec7;
  typedef Eigen::Matrix<double,6,1> Vec6;
  typedef Eigen::Matrix<double,5,1> Vec5;
  typedef Eigen::Matrix<double,4,1> Vec4;
  typedef Eigen::Matrix<double,3,1> Vec3;
  typedef Eigen::Matrix<double,2,1> Vec2;
  typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecX;

  typedef Eigen::Matrix<float,3,3> Mat33f;
  typedef Eigen::Matrix<float,10,3> Mat103f;
  typedef Eigen::Matrix<float,2,2> Mat22f;
  typedef Eigen::Matrix<float,3,1> Vec3f;
  typedef Eigen::Matrix<float,2,1> Vec2f;
  typedef Eigen::Matrix<float,6,1> Vec6f;

  typedef Eigen::Vector2i Vec2i;
  typedef Eigen::Vector3i Vec3i;


  typedef Eigen::Matrix<double,4,9> Mat49;
  typedef Eigen::Matrix<double,8,9> Mat89;

  typedef Eigen::Matrix<double,9,4> Mat94;
  typedef Eigen::Matrix<double,9,8> Mat98;

  typedef Eigen::Matrix<double,8,1> Mat81;
  typedef Eigen::Matrix<double,1,8> Mat18;
  typedef Eigen::Matrix<double,9,1> Mat91;
  typedef Eigen::Matrix<double,1,9> Mat19;


  typedef Eigen::Matrix<double,8,4> Mat84;
  typedef Eigen::Matrix<double,4,8> Mat48;
  typedef Eigen::Matrix<double,4,4> Mat44;


  typedef Eigen::Matrix<float,MAX_RES_PER_POINT,1> VecNRf;
  typedef Eigen::Matrix<float,12,1> Vec12f;
  typedef Eigen::Matrix<float,1,8> Mat18f;
  typedef Eigen::Matrix<float,6,6> Mat66f;
  typedef Eigen::Matrix<float,8,8> Mat88f;
  typedef Eigen::Matrix<float,8,4> Mat84f;
  typedef Eigen::Matrix<float,8,1> Vec8f;
  typedef Eigen::Matrix<float,10,1> Vec10f;
  typedef Eigen::Matrix<float,6,6> Mat66f;
  typedef Eigen::Matrix<float,4,1> Vec4f;
  typedef Eigen::Matrix<float,5,1> Vec5f;
  typedef Eigen::Matrix<float,4,4> Mat44f;
  typedef Eigen::Matrix<float,12,12> Mat1212f;
  typedef Eigen::Matrix<float,12,1> Vec12f;
  typedef Eigen::Matrix<float,13,13> Mat1313f;
  typedef Eigen::Matrix<float,10,10> Mat1010f;
  typedef Eigen::Matrix<float,13,1> Vec13f;
  typedef Eigen::Matrix<float,9,9> Mat99f;
  typedef Eigen::Matrix<float,9,1> Vec9f;
  typedef	Eigen::Matrix<float,3,4> Mat34f;

  typedef Eigen::Matrix<float,4,2> Mat42f;
  typedef Eigen::Matrix<float,6,2> Mat62f;
  typedef Eigen::Matrix<float,1,2> Mat12f;

  typedef Eigen::Matrix<float,Eigen::Dynamic,1> VecXf;
  typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> MatXXf;


  typedef Eigen::Matrix<double,8+CPARS+1,8+CPARS+1> MatPCPC;
  typedef Eigen::Matrix<float,8+CPARS+1,8+CPARS+1> MatPCPCf;
  typedef Eigen::Matrix<double,8+CPARS+1,1> VecPC;
  typedef Eigen::Matrix<float,8+CPARS+1,1> VecPCf;

  typedef Eigen::Matrix<float,14,14> Mat1414f;
  typedef Eigen::Matrix<float,14,1> Vec14f;
  typedef Eigen::Matrix<double,14,14> Mat1414;
  typedef Eigen::Matrix<double,14,1> Vec14;

  typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> ArrayVec3f;
  typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> ArrayVec3d;
  typedef ArrayVec3f cloud_t;
  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_row;
  /*  
  struct frame{

    int frame_id;

    int h;        // height of the image without downsampling
    int w;        // width of the image without downsampling

    cv::Mat image;
    cv::Mat intensity;
    cv::Mat depth;
    cv::Mat semantic_img;
    MatrixXf_row semantic_labels;


    Eigen::Vector3f* dI;    // flattened image gradient, (w*h,3). 0: magnitude, 1: dx, 2: dy
    Eigen::Vector3f* dI_pyr[PYR_LEVELS];  // pyramid for dI. dI_pyr[0] = dI
    float* abs_squared_grad[PYR_LEVELS];  // pyramid for absolute squared gradient (dx^2+dy^2)

  };

  struct point_cloud{

    int num_points;
    int num_classes;
    //typedef std::vector<Eigen::Vector3f> cloud_t;
    cloud_t positions;  // points position. x,y,z
    //Eigen::Matrix<float, Eigen::Dynamic, 5> features;   // features are rgb dx dy
    Eigen::Matrix<float, Eigen::Dynamic, 5> features;   // rgb
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> labels; // number of points by number of classes  
    // 0. building 1. sky 2. road
    // 3. vegetation 4. sidewalk 5. car 6. pedestrian
    // 7. cyclist 8. signate 9. fence 10. pole

  };
  */


}


