#pragma once
//#include "utils/PointSegmentedDistribution.hpp"
//#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <cstdlib>
#include "CvoFrame.hpp"
#include "utils/CvoPoint.hpp"
#include "cvo/CvoGPU_impl.cuh"
#include <thrust/device_vector.h>

namespace cvo {
  /*
  struct CvoFrameGPU : public CvoFrame {
  public:
    CvoFrameGPU(const CvoPointCloud * pts,
                const double poses[12]);

    ~CvoFrameGPU();

    void transform_pointcloud();

    const CvoPoint * points_transformed_gpu() const;

    //void set_points_transformed_gpu();
    const float * pose_vec_gpu(); // 12, row major
    void set_pose_vec_gpu(float * new_pose_vec_cpu); // 12, row major

  private:
    CvoFrameGPUImpl * impl;
    
    thrust::device_vector<CvoPoint> points_init_gpu_;
    thrust::device_vector<CvoPoint> points_transformed_gpu_;
    float * pose_vec_gpu_; // 12, row major

  };
  */

  class CvoFrameGPU_Impl {
  public:
    CvoFrameGPU_Impl(const CvoPointCloud * pts,
                     const double poses[12]);
    ~CvoFrameGPU_Impl();
    
    const CvoPoint * points_transformed_gpu() const;
    void transform_pointcloud_from_input_pose(const double * pose_vec_cpu);
    
    const float * pose_vec_gpu() const; // 12, row major
    // void set_pose_vec_gpu(float * new_pose_vec_cpu); // 12, row major


  private:
    
    thrust::device_vector<CvoPoint> points_init_gpu_;
    thrust::device_vector<CvoPoint> points_transformed_gpu_;
    float * pose_vec_gpu_; // 12, row major
    
  };
}
