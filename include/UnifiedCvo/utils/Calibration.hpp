#pragma once

#include <fstream>
#include <string>
#include <cstdio>
#include <cassert>
#include <Eigen/Dense>

namespace cvo {
  class Calibration {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    enum PointCloudType {
      STEREO,
      RGBD
    };

    /* 
     * data_type 0: stereo, 1: rgbd
    */
    Calibration(std::string &file, PointCloudType data_type=STEREO) {
      std::ifstream infile(file);
      intrinsic_.setIdentity();
      baseline_ = 0;
      scaling_factor_ = 0;

      if(data_type == STEREO){
        scaling_factor_ = 1;
        if (infile.is_open()) {
          float fx, fy, cx , cy;
          infile >> fx >> fy >> cx >> cy >> baseline_;
          intrinsic_(0,0) = fx;
          intrinsic_(1,1) = fy;
          intrinsic_(0,2) = cx;
          intrinsic_(1,2) = cy;
          std::cout<<"calib files for stereo read!"<<std::endl;
          std::cout<<"intrinsic: \n"<<intrinsic_<<"\n baseline: "<<baseline_<<std::endl;
          if (!infile.eof()) {
            infile >> cols >> rows;            
          }
          infile.close();

        } else {
          std::cerr<<" calibration file "<<file<<" not found!\n";
          //assert(0);
        }
      }
      else if(data_type == RGBD){
        if (infile.is_open()) {
          float fx, fy, cx , cy;
          infile >> fx >> fy >> cx >> cy >> scaling_factor_;
          intrinsic_(0,0) = fx;
          intrinsic_(1,1) = fy;
          intrinsic_(0,2) = cx;
          intrinsic_(1,2) = cy;
          std::cout<<"calib files for RGBD read!"<<std::endl;
          std::cout<<"intrinsic: \n"<<intrinsic_<<"\n scaling_factor_: "<<scaling_factor_<<std::endl;
          if (!infile.eof()) {
            infile >> cols >> rows;            
          }
          infile.close();

        } else {
          std::cerr<<" calibration file "<<file<<" not found!\n";
          //assert(0);
        }
      }
    }

    Calibration(){}


    const Eigen::Matrix3f & intrinsic()const {return intrinsic_;}
    float baseline()const { return  baseline_; }
    float scaling_factor()const {return scaling_factor_; }
    const Eigen::Matrix<float, 3, 4> & cam_frame_to_lidar_frame() {return cam_frame_to_lidar_frame_;}
    int image_cols() const {return cols;}
    int image_rows() const {return rows;}
  private:
    Eigen::Matrix3f intrinsic_;
    float baseline_; // for stereo
    float scaling_factor_; // for rgbd
    Eigen::Matrix<float, 3, 4> cam_frame_to_lidar_frame_;
    int cols, rows;
  };
  
}
