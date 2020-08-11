#pragma once
#include "SparseKernelMat.cuh"
#include "CvoParams.hpp"
#include "utils/PointSegmentedDistribution.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cukdtree/cukdtree.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <Eigen/Dense>

namespace Eigen {
  typedef Matrix<float,1,3> Vector3f_row;                 
}



namespace cvo {

  const int CVO_POINT_NEIGHBORS = 400;  
  typedef pcl::PointSegmentedDistribution<FEATURE_DIMENSIONS,NUM_CLASSES> CvoPoint;
  typedef perl_registration::cuPointCloud<CvoPoint> CvoPointCloudGPU;
}



namespace cvo {
  // all the GPU memory are allocated when construction CvoState, and deleted in CvoState's destructor
  struct CvoState {
    CvoState(std::shared_ptr<CvoPointCloudGPU> source_points, // ownership: user
             std::shared_ptr<CvoPointCloudGPU> target_points, // ownership: user
             const CvoParams & cvo_params);    
    ~CvoState();

    void reset_state_at_new_iter();
        
    // shared between cpu && gpu
    double dl;
    float step;
    int num_fixed;
    int num_moving;
    float ell;          // kernel characteristic length-scale
    float ell_max;
    
    // GPU raw memory
    SparseKernelMat * A;
    SparseKernelMat *Axx;
    SparseKernelMat *Ayy;
    SparseKernelMat A_host, Axx_host, Ayy_host;
    Eigen::Matrix3f * R_gpu;
    Eigen::Vector3f * T_gpu;

    // thrust device memory. pre-allocated here. 
    std::shared_ptr<CvoPointCloudGPU> cloud_x_gpu; 
    std::shared_ptr<CvoPointCloudGPU> cloud_y_gpu;
    std::shared_ptr<CvoPointCloudGPU> cloud_y_gpu_init;
    //perl_registration::cuKdTree<CvoPoint>::SharedPtr kdtree_fixed_points;
    //perl_registration::cuKdTree<CvoPoint>::SharedPtr kdtree_moving_points;
    thrust::device_vector<double> partial_dl_gradient;
    thrust::device_vector<double> partial_dl_Ayy;
    //thrust::device_vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> omega_gpu;
    thrust::device_vector<Eigen::Vector3d > omega_gpu;
    //thrust::device_vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> v_gpu;
    thrust::device_vector<Eigen::Vector3d> v_gpu;
    //thrust::device_vector<float> cross_xy, diff_yx, diff_xx, diff_yy, sum_diff_yx_2, sum_diff_xx_2, sum_diff_yy_2;
    
    thrust::device_vector<Eigen::Vector3f_row> xiz;
    thrust::device_vector<Eigen::Vector3f_row>  xi2z;
    thrust::device_vector<Eigen::Vector3f_row>  xi3z;
    thrust::device_vector<Eigen::Vector3f_row>  xi4z;
    thrust::device_vector<float> normxiz2;
    thrust::device_vector<float> xiz_dot_xi2z;
    thrust::device_vector<float> epsil_const;
    thrust::device_vector<double> B, C,D,E;
    Eigen::Vector3f *omega;
    Eigen::Vector3f *v;

    bool is_ell_adaptive;

  };

}
