#pragma once
#include "SparseKernelMat.cuh"
#include "CvoParams.hpp"
#include "utils/PointSegmentedDistribution.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cukdtree/cukdtree.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <Eigen/Dense>

#define KDTREE_K_SIZE 500

namespace Eigen {
                 typedef Matrix<float,1,3> Vector3f_row;                 
}


namespace cvo {
  
  typedef pcl::PointSegmentedDistribution<FEATURE_DIMENSIONS,NUM_CLASSES> CvoPoint;
  //typedef thrust::device_vector<CvoPoint> CvoPointCloudGPU;
  typedef perl_registration::cuPointCloud<CvoPoint> CvoPointCloudGPU ;
  
  // all the GPU memory are allocated when construction CvoState, and deleted in CvoState's destructor
  // reducing memory allocation time 
  struct CvoState {


    CvoState(std::shared_ptr<CvoPointCloudGPU> source_points, // ownership: caller
             std::shared_ptr<CvoPointCloudGPU> target_points, // ownership: caller
             const CvoParams & cvo_params,
             bool is_adaptive=true
             ) ;
    
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
    perl_registration::cuKdTree<CvoPoint>::SharedPtr kdtree_fixed_points;
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


  inline
  CvoState::CvoState(std::shared_ptr<CvoPointCloudGPU> source_points,
                     std::shared_ptr< CvoPointCloudGPU> target_points,
                     const CvoParams & cvo_params,
                     bool is_adaptive):
    dl (cvo_params.dl),
    step (0),
    num_fixed (source_points->size()),
    num_moving (target_points->size()),
    ell (cvo_params.ell_init),
    ell_max (cvo_params.ell_max),
    partial_dl_gradient(is_adaptive?num_fixed:1),
    partial_dl_Ayy(is_adaptive?num_moving:1),
    omega_gpu(num_fixed),
    v_gpu(num_fixed),
    //omega_gpu(is_adaptive?num_fixed :num_moving),
    //v_gpu(is_adaptive?num_fixed :num_moving),
    /*
    cross_xy(3 * num_moving),
    diff_yx(),
    diff_xx(),
    diff_yy(),
    sum_diff_yx_2(),
    sum_diff_xx_2(),
    sum_diff_yy_2(),
    */
    xiz(num_moving),
    xi2z(num_moving),
    xi3z(num_moving),
    xi4z(num_moving),
    normxiz2(num_moving),
    xiz_dot_xi2z(num_moving),
    epsil_const(num_moving),
    B(num_fixed),
    C(num_fixed),
    D(num_fixed),
    E(num_fixed),

    //B(is_adaptive?num_fixed:num_moving),
    //C(is_adaptive?num_fixed:num_moving),
    //D(is_adaptive?num_fixed:num_moving),
    //E(is_adaptive?num_fixed:num_moving),
    is_ell_adaptive(is_adaptive)
  {

    // gpu raw
    //int A_rows = is_ell_adaptive?  source_points->size() : target_points->size();
    int A_rows = source_points->size() ;
    int Ayy_rows = target_points->size();
    int Axx_rows = source_points->size();
        
    //int A_cols = target_points->size();
    int A_cols = KDTREE_K_SIZE;
    int Axx_cols = KDTREE_K_SIZE;

    A = init_SparseKernelMat_gpu(A_rows, A_cols, A_host);
    if(is_ell_adaptive) {
      Axx = init_SparseKernelMat_gpu(Axx_rows, Axx_cols, Axx_host);
      Ayy = init_SparseKernelMat_gpu(Ayy_rows, A_cols, Ayy_host);
    }
    cudaMalloc((void**)&R_gpu, sizeof(Eigen::Matrix3f));
    cudaMalloc((void**)&T_gpu, sizeof(Eigen::Vector3f));
    cudaMalloc((void**)&omega, sizeof(Eigen::Vector3f));
    cudaMalloc((void**)&v, sizeof(Eigen::Vector3f));

    //  thrust device memory
    cloud_x_gpu = source_points;
    cloud_y_gpu_init = target_points;
    cloud_y_gpu.reset(new CvoPointCloudGPU(num_moving ) );
    /*
    if (!is_ell_adaptive) {
      printf("Build kdtree...\n");
      kdtree_fixed_points.reset(new perl_registration::cuKdTree<CvoPoint>);
      kdtree_fixed_points->SetInputCloud(source_points);
      printf("finish building kdtree on fixed_points\n");
      }*/

    std::cout<<"partial_dl_gradient size is "<<partial_dl_gradient.size()<<std::endl;

  }

  inline
  CvoState::~CvoState() {
    
    
    cudaFree(R_gpu);
    cudaFree(T_gpu);
    cudaFree(omega);
    cudaFree(v);

    delete_SparseKernelMat_gpu(A, &A_host);
    if (is_ell_adaptive) {
      delete_SparseKernelMat_gpu(Axx, &Axx_host);
      delete_SparseKernelMat_gpu(Ayy, &Ayy_host);
    }
    
  }

  inline void CvoState::reset_state_at_new_iter () {

    cudaMemset( (void*)A_host.mat, 0, sizeof(float) * A_host.rows * A_host.cols );
    cudaMemset( (void*)A_host.ind_row2col , -1 , sizeof(int )* A_host.rows * A_host.cols  );

    if (is_ell_adaptive) {
      cudaMemset( (void*)Axx_host.mat, 0, sizeof(float) * Axx_host.rows * Axx_host.cols  );
      cudaMemset( (void*)Axx_host.ind_row2col , -1 , sizeof(int )* Axx_host.rows * Axx_host.cols);
      cudaMemset( (void*)Ayy_host.mat, 0, sizeof(float) * Ayy_host.rows * Ayy_host.cols  );
      cudaMemset( (void*)Ayy_host.ind_row2col , -1 , sizeof(int )* Ayy_host.rows * Ayy_host.cols );
    }
  }
  
}
