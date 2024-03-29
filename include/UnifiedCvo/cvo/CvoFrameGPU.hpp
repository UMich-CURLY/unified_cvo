#pragma once
//#include "utils/PointSegmentedDistribution.hpp"
//#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <cstdlib>
#include "CvoFrame.hpp"
#include "utils/CvoPoint.hpp"
//#include "cvo/CvoGPU_impl.cuh"
#include <memory>
namespace cvo {

  class CvoFrameGPU_Impl;

  struct CvoFrameGPU : public CvoFrame  {
  public:
    CvoFrameGPU(const CvoPointCloud * pts,
                const double poses[12]);

    ~CvoFrameGPU();

    void transform_pointcloud();

    // access
    const CvoPoint * points_transformed_gpu() const;
    //void set_points_transformed_gpu();
    const float * pose_vec_gpu() const; // 12, row major
    //void set_pose_vec_gpu(float * new_pose_vec_cpu); // 12, row major

  private:
    
    ///CvoPoint * points_init_gpu_;
    //CvoPoint * points_transformed_gpu_;
    //float * pose_vec_gpu_; // 12, row major
    std::unique_ptr<CvoFrameGPU_Impl> impl;

  };
}
