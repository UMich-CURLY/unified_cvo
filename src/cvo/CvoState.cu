#include "cvo/CvoState.cuh"
#include <chrono>

#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>


namespace cvo {

  /*   helper functions  */

  __global__
  void init_covariance(CvoPoint * points, // mutate
                       int num_points,
                       int * neighbors,
                       int num_neighbors_each_point
                       ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > num_points - 1) {
      return;
    }

    CvoPoint & curr_p = points[i];
    Eigen::Vector3f curr_p_vec;
    curr_p_vec << curr_p.x, curr_p.y, curr_p.z;
    int * indices = neighbors + i * num_neighbors_each_point
    
    Eigen::Vector3f mean(0, 0, 0);
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();

    for (int j = 0; j < num_neighbors_each_point; j++) {
      
      mean = mean + points.data[k_nns.data[i]].toVec();
    }
  mean = mean * (1.0f / (float)(k));

  for (int i = pos * k; i < (pos + 1) * k; i++) {
    Eigen::Vector3f temp = points.data[k_nns.data[i]].toVec() - mean;
    Eigen::Matrix3f temp_m = temp * temp.transpose();
    covariance = covariance + temp_m;
  }

  /* PCA */
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(3);
  es.computeDirect(covariance);
  // Eigen values are sorted
  Eigen::Matrix3f eigen_value_replacement = Eigen::Matrix3f::Zero();
  eigen_value_replacement(0, 0) = 1e-3;
  eigen_value_replacement(1, 1) = 1.0;
  eigen_value_replacement(2, 2) = 1.0;
  covariances.data[pos] = es.eigenvectors() * eigen_value_replacement *
                          es.eigenvectors().transpose();
  covariance = covariances.data[pos];
  }

  
  void fill_in_pointcloud_covariance(perl_registration::cuKdTree<CvoPoint>::SharedPtr kdtree,
                                     std::shared_ptr<CvoPointCloudGPU> pointcloud_gpu ) {
    auto start chrono::system_clock::now();
    thrust::device_vector<int> indices;
    const int num_wanted_points = 20;
    kdtree->NearestKSearch(pointcloud_gpu, num_wanted_points, indices );
    ind_device = thrust::raw_pointer_cast(indices.data() );
    auto end = chrono::system_clock::now();
    chrono::duration<double> t_kdtree_search = end-start;
    std::cout<<"t kdtree search in se_kernel is "<<t_kdtree_search.count()<<std::endl;



    
  }



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
    is_ell_adaptive(cvo_params.is_ell_adaptive)
  {

    // gpu raw
    //int A_rows = is_ell_adaptive?  source_points->size() : target_points->size();
    int A_rows = source_points->size() ;
    int Ayy_rows = target_points->size();
    int Axx_rows = source_points->size();
        
    //int A_cols = target_points->size();
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
    
    if (cvo_params.is_dense_kernel) {
      printf("Build kdtree...\n");
      kdtree_fixed_points.reset(new perl_registration::cuKdTree<CvoPoint>);
      kdtree_fixed_points->SetInputCloud(source_points);
      kdtree_moving_points.reset(new perl_registration::cuKdTree<CvoPoint>);
      kdtree_moving_points->SetInputCloud(target_points);
      
      printf("finish building kdtree on fixed_points and target_points\n");
    }

    std::cout<<"partial_dl_gradient size is "<<partial_dl_gradient.size()<<std::endl;

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
