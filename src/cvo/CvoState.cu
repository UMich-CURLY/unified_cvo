#include "cvo/CvoState.cuh"
#include "cvo/gpu_utils.cuh"
#include "cukdtree/cukdtree.h"
#include <chrono>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>


// explicit template instantiation
template struct pcl::PointSegmentedDistribution<FEATURE_DIMENSIONS, NUM_CLASSES>;

namespace cvo {


  /*   class member functions  */
  
  CvoState::CvoState(std::shared_ptr<CvoPointCloudGPU> source_points,
                     std::shared_ptr< CvoPointCloudGPU> target_points,
                     const CvoParams & cvo_params
                     ):
    dl (cvo_params.dl),
    step (0),
    num_fixed (source_points->size()),
    num_moving (target_points->size()),
    ell (cvo_params.ell_init),
    ell_max (cvo_params.ell_max),
    partial_dl_gradient(cvo_params.is_ell_adaptive?num_fixed:1, 0),
    partial_dl_Ayy(cvo_params.is_ell_adaptive?num_moving:1, 0),
    omega_gpu(num_fixed, Eigen::Vector3d::Zero()),
    v_gpu(num_fixed,Eigen::Vector3d::Zero()),
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
    xiz(num_moving, Eigen::Vector3f_row::Zero()),
    xi2z(num_moving, Eigen::Vector3f_row::Zero()),
    xi3z(num_moving, Eigen::Vector3f_row::Zero()),
    xi4z(num_moving, Eigen::Vector3f_row::Zero()),
    normxiz2(num_moving, 0),
    xiz_dot_xi2z(num_moving, 0),
    epsil_const(num_moving, 0),
    B(num_fixed, 0),
    C(num_fixed, 0),
    D(num_fixed, 0),
    E(num_fixed, 0),

    //B(is_adaptive?num_fixed:num_moving),
    //C(is_adaptive?num_fixed:num_moving),
    //D(is_adaptive?num_fixed:num_moving),
    //E(is_adaptive?num_fixed:num_moving),
    is_ell_adaptive(cvo_params.is_ell_adaptive)
  {
    std::cout<<"start construct CvoState\n";
    // gpu raw
    //int A_rows = is_ell_adaptive?  source_points->size() : target_points->size();
    int A_rows = source_points->size() ;
    int Ayy_rows = target_points->size();
    int Axx_rows = source_points->size();
        
    int A_cols = CVO_POINT_NEIGHBORS;
    int Axx_cols = CVO_POINT_NEIGHBORS;

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

  }

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

  void CvoState::reset_state_at_new_iter () {

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
