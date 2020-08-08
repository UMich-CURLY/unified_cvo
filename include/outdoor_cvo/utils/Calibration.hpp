#pragma once

#include <fstream>
#include <string>
#include <cstdio>
#include <cassert>
#include <Eigen/Dense>

namespace cvo {
  class Calibration {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    Calibration(std::string &file) {
      std::ifstream infile(file);
      intrinsic_.setIdentity();
      baseline_ = 0;
      if (infile.is_open()) {
        float fx, fy, cx , cy;
        infile >> fx >> fy >> cx >> cy >> baseline_;
        intrinsic_(0,0) = fx;
        intrinsic_(1,1) = fy;
        intrinsic_(0,2) = cx;
        intrinsic_(1,2) = cy;
        std::cout<<"intrinsic: \n"<<intrinsic_<<"\n baseline: "<<baseline_<<std::endl;
        if (!infile.eof()) {
          
          
        }
        infile.close();

      } else {
        std::cerr<<" calibration file "<<file<<" not found!\n";
        //assert(0);
        
      }


    }

    Calibration(){}

    void read_file(std::string &file) {
      std::ifstream infile(file);
      intrinsic_.setIdentity();
      baseline_ = 0;
      if (infile.is_open()) {
        float fx, fy, cx , cy;
        infile >> fx >> fy >> cx >> cy >> baseline_;
        intrinsic_(0,0) = fx;
        intrinsic_(1,1) = fy;
        intrinsic_(0,2) = cx;
        intrinsic_(1,2) = cy;
        std::cout<<"intrinsic: \n"<<intrinsic_<<"\n baseline: "<<baseline_<<std::endl;
        if (!infile.eof()) {
          
          
        }
        infile.close();

      } else {
        std::cerr<<" calibration file "<<file<<" not found!\n";
        assert(0);
        
      }


    }
    const Mat33f & intrinsic()const {return intrinsic_;}
    float baseline()const { return  baseline_; }
    const Mat34f & cam_frame_to_lidar_frame() {return cam_frame_to_lidar_frame_;}
  private:
    Mat33f intrinsic_;
    float baseline_;
    Mat34f cam_frame_to_lidar_frame_;
  };
  
}
