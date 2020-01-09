#pragma once
#include "SparseKernelMat.cuh"
#include "CvoParams.cuh"
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
  
  typedef pcl::PointSegmentedDistribution<FEATURE_DIMENSIONS,NUM_CLASSES> CvoPoint;
  //typedef thrust::device_vector<CvoPoint> CvoPointCloudGPU;
  typedef perl_registration::cuPointCloud<CvoPoint> CvoPointCloudGPU ;
  
  // all the GPU memory are allocated when construction CvoState, and deleted in CvoState's destructor
  // reducing memory allocation time 
  struct CvoState {


    CvoState(std::shared_ptr<CvoPointCloudGPU> source_points, // ownership: caller
             std::shared_ptr<CvoPointCloudGPU> target_points, // ownership: caller
             const CvoParams & cvo_params
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
    Eigen::Matrix3f * R_gpu;
    Eigen::Vector3f * T_gpu;

    // thrust device memory. pre-allocated here. 
    std::shared_ptr<CvoPointCloudGPU> cloud_x_gpu; 
    std::shared_ptr<CvoPointCloudGPU> cloud_y_gpu;
    std::shared_ptr<CvoPointCloudGPU> cloud_y_gpu_init;
    perl_registration::cuKdTree<CvoPoint>::SharedPtr kdtree_fixed_points;
    thrust::device_vector<double> partial_dl_gradient;
    thrust::device_vector<double> partial_dl_Ayy;
    thrust::device_vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> omega_gpu;
    thrust::device_vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> v_gpu;
    thrust::device_vector<Eigen::Vector3f_row, Eigen::aligned_allocator<Eigen::Vector3f_row>> xiz;
    thrust::device_vector<Eigen::Vector3f_row, Eigen::aligned_allocator<Eigen::Vector3f_row>>  xi2z;
    thrust::device_vector<Eigen::Vector3f_row, Eigen::aligned_allocator<Eigen::Vector3f_row>>  xi3z;
    thrust::device_vector<Eigen::Vector3f_row, Eigen::aligned_allocator<Eigen::Vector3f_row>>  xi4z;
    thrust::device_vector<float> normxiz2;
    thrust::device_vector<float> xiz_dot_xi2z;
    thrust::device_vector<float> epsil_const;
    thrust::device_vector<double> B, C,D,E;
    Eigen::Vector3f *omega;
    Eigen::Vector3f *v;


  };


  inline
  CvoState::CvoState(std::shared_ptr<CvoPointCloudGPU> source_points,
                     std::shared_ptr< CvoPointCloudGPU> target_points,
                     const CvoParams & cvo_params):
    dl (cvo_params.dl),
    step (0),
    num_fixed (source_points->size()),
    num_moving (target_points->size()),
    ell (cvo_params.ell_init),
    ell_max (cvo_params.ell_max),
    partial_dl_gradient(num_fixed),
    partial_dl_Ayy(num_moving),
    omega_gpu(num_fixed),
    v_gpu(num_fixed),
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
    E(num_fixed)
  {

    // gpu raw
    int A_rows = source_points->size();
    int Ayy_rows = target_points->size();
    int A_cols = KDTREE_K_SIZE;
    A = init_SparseKernelMat_gpu(A_rows, A_cols);
    Axx = init_SparseKernelMat_gpu(A_rows, A_cols);
    Ayy = init_SparseKernelMat_gpu(Ayy_rows, A_cols);
    cudaMalloc((void**)&R_gpu, sizeof(Eigen::Matrix3f));
    cudaMalloc((void**)&T_gpu, sizeof(Eigen::Vector3f));
    cudaMalloc((void**)&omega, sizeof(Eigen::Vector3f));
    cudaMalloc((void**)&v, sizeof(Eigen::Vector3f));

    //  thrust device memory
    cloud_x_gpu = source_points;
    cloud_y_gpu_init = target_points;
    cloud_y_gpu.reset(new CvoPointCloudGPU(num_moving ) );
    kdtree_fixed_points.reset(new perl_registration::cuKdTree<CvoPoint>);
    
  }

  inline
  CvoState::~CvoState() {
    
    
    cudaFree(R_gpu);
    cudaFree(T_gpu);
    cudaFree(omega);
    cudaFree(v);

    delete_SparseKernelMat_gpu(A);
    delete_SparseKernelMat_gpu(Axx);
    delete_SparseKernelMat_gpu(Ayy);
    
  }

  inline void CvoState::reset_state_at_new_iter () {
    cudaMemset( (void*)A->mat, 0, sizeof(float) * num_fixed * KDTREE_K_SIZE );
    cudaMemset( (void*)A->ind_row2col , 0 , sizeof(int )* num_fixed  * KDTREE_K_SIZE );
    cudaMemset( (void*)Axx->mat, 0, sizeof(float) * num_fixed * KDTREE_K_SIZE  );
    cudaMemset( (void*)Axx->ind_row2col , 0 , sizeof(int )* num_fixed * KDTREE_K_SIZE );
    cudaMemset( (void*)Ayy->mat, 0, sizeof(float) * num_moving* KDTREE_K_SIZE  );
    cudaMemset( (void*)Ayy->ind_row2col , 0 , sizeof(int )* num_moving * KDTREE_K_SIZE );
    
    
  }
  
}
