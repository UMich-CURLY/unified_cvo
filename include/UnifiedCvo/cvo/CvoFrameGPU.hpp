#pragma once
//#include "utils/PointSegmentedDistribution.hpp"
//#include <Eigen/Dense>
//#include <pcl/point_types.h>
#include <cstdlib>
#include "CvoFrame.hpp"
#include "utils/CvoPoint.hpp"

//#include "cvo/CvoGPU_impl.cuh"
#include <memory>
namespace cvo {

  //class CvoFrameGPU_Impl;
  class CvoPointCloudGPU;
  class CuKdTree;
  class CvoPointCloud;
  
  struct CvoFrameGPU : public CvoFrame  {
  public:
    CvoFrameGPU(const CvoPointCloud * pts, // points in the sensor frame
                const double poses[12],   // global poses
                bool is_using_kdtree=false);

    ~CvoFrameGPU();

    void transform_pointcloud();

    // access
    std::shared_ptr<CvoPointCloudGPU> points_init_gpu();
    std::shared_ptr<CvoPointCloudGPU> points_transformed_gpu();
    size_t size() const;
    const CuKdTree & kdtree() const;
    
    //void set_points_transformed_gpu();
    const float * pose_vec_gpu() const; // 12, row major
    Eigen::Matrix4f pose_cpu() const;
    //void set_pose_vec_gpu(float * new_pose_vec_cpu); // 12, row major

  private:
    

    //std::unique_ptr<CvoFrameGPU_Impl> impl;
    std::shared_ptr<CvoPointCloudGPU> points_init_gpu_;
    std::shared_ptr<CvoPointCloudGPU> points_transformed_gpu_; // place holder

    //perl_registration::cuKdTree<CvoPoint>::SharedPtr kdtree_points;
    //
    std::shared_ptr<CuKdTree> kdtree_;

    
    float * pose_vec_gpu_; // 12, row major

  };
}
