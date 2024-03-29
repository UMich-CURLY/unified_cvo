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
    
    is_ell_adaptive(cvo_params.is_ell_adaptive),
    least_square_LHS(num_fixed, Eigen::Matrix<float, 6,6>::Zero()),
    least_square_RHS(num_fixed, Eigen::Matrix<float, 6,1>::Zero())
  {
    //std::cout<<"start construct CvoState\n";
    // gpu raw
    //int A_rows = is_ell_adaptive?  source_points->size() : target_points->size();
    int A_rows = source_points->size() ;
    int Ayy_rows = target_points->size();
    int Axx_rows = source_points->size();

    int A_cols, Axx_cols;
    if (cvo_params.is_full_ip_matrix)
      A_cols = Axx_cols = target_points->size();
    else 
      A_cols = Axx_cols = cvo_params.nearest_neighbors_max;
    
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

    if (cvo_params.is_using_kdtree) {
      kdtree_moving_points.reset(new perl_registration::cuKdTree<CvoPoint> );
      kdtree_moving_points->SetInputCloud(cloud_y_gpu_init);
      cloud_x_gpu_transformed_kdtree.reset(new CvoPointCloudGPU(num_fixed));
      kdtree_inds_results.resize(cvo_params.is_using_kdtree * num_fixed);
    }
    cudaDeviceSynchronize();    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { 
      fprintf(stderr, "Failed to run CvoState Init %s .\n", cudaGetErrorString(err)); 
      exit(EXIT_FAILURE); 
    }
    
    

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

    clear_SparseKernelMat(&A_host);
    
    if (is_ell_adaptive) {
      cudaMemset( (void*)Axx_host.mat, 0, sizeof(float) * Axx_host.rows * Axx_host.cols  );
      cudaMemset( (void*)Axx_host.ind_row2col , -1 , sizeof(int )* Axx_host.rows * Axx_host.cols);
      cudaMemset( (void*)Ayy_host.mat, 0, sizeof(float) * Ayy_host.rows * Ayy_host.cols  );
      cudaMemset( (void*)Ayy_host.ind_row2col , -1 , sizeof(int )* Ayy_host.rows * Ayy_host.cols );
    }

    if (kdtree_moving_points) {
      cudaMemset(thrust::raw_pointer_cast(kdtree_inds_results.data()), -1, sizeof(int) * kdtree_inds_results.size()   );
    }
  }

  void CvoState::reset_state_at_new_iter (int num_neighbors) {

    clear_SparseKernelMat(&A_host, num_neighbors);
    
    if (is_ell_adaptive) {
      cudaMemset( (void*)Axx_host.mat, 0, sizeof(float) * Axx_host.rows * num_neighbors  );
      cudaMemset( (void*)Axx_host.ind_row2col , -1 , sizeof(int )* Axx_host.rows * num_neighbors);
      cudaMemset( (void*)Ayy_host.mat, 0, sizeof(float) * Ayy_host.rows * num_neighbors );
      cudaMemset( (void*)Ayy_host.ind_row2col , -1 , sizeof(int )* Ayy_host.rows * num_neighbors );
    }

    if (kdtree_moving_points) {
      cudaMemset(thrust::raw_pointer_cast(kdtree_inds_results.data()), -1, sizeof(int) * kdtree_inds_results.size()   );
    }
  }

  
}
