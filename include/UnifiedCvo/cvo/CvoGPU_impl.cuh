#pragma once
#include "cvo/Association.hpp"
#include "cvo/SparseKernelMat.hpp"
#include "cvo/CvoState.cuh"

#include "cvo/LieGroup.h"

#include "cvo/CvoParams.hpp"
#include "utils/PointSegmentedDistribution.hpp"
#include "cupointcloud/point_types.h"
#include "cupointcloud/cupointcloud.h"
#include "cukdtree/cukdtree.cuh"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

/*
#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void gpu_error_check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        std::exit(EXIT_FAILURE);
    }
}
#endif

#ifndef CHECK_LAST_CUDA_ERROR
#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void gpu_last_error_check(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        std::exit(EXIT_FAILURE);
    }
}
#endif
*/
#define GpuErrorCheck(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
  
}

namespace cvo {
  //const int CUDA_BLOCK_SIZE = 1024 ;//128;
  CvoPointCloudGPU::SharedPtr CvoPointCloud_to_gpu(const CvoPointCloud& cvo_cloud );
  CvoPointCloudGPU::SharedPtr pcl_PointCloud_to_gpu(const pcl::PointCloud<CvoPoint> & cvo_cloud );
  void CvoPointCloud_to_gpu(const CvoPointCloud& cvo_cloud, thrust::device_vector<CvoPoint> & output );
  void pcl_PointCloud_to_gpu(const pcl::PointCloud<CvoPoint> & cvo_cloud, thrust::device_vector<CvoPoint> & output );



  __global__
  void fill_in_A_mat_gpu(const CvoParams * cvo_params,
                         const CvoPoint * points_a,
                         int a_size,
                         const CvoPoint * points_b,
                         int b_size,
                         int num_neighbors,
                         float ell,
                         // output
                         SparseKernelMat * A_mat // the kernel matrix!
                         );

  

  float compute_ranged_lengthscale(float curr_dist_square, float curr_ell, float min_ell, float max_ell );

  void update_tf(const Mat33f & R, const Vec3f & T,
                 // outputs
                 CvoState * cvo_state,
                 Eigen::Ref<Mat44f > transform
                 );
   

  /**
   * @brief isotropic (same length-scale for all dimensions) squared-exponential kernel
   * @param l: kernel characteristic length-scale, aka cvo.ell
   * @param s2: signal variance, square of cvo.sigma
   * @return k: n-by-m kernel matrix 
   */
   
  void se_kernel(const CvoParams * params_gpu,
                 std::shared_ptr<CvoPointCloudGPU> points_fixed,
                 std::shared_ptr<CvoPointCloudGPU> points_moving,
                 float ell,
                 perl_registration::cuKdTree<CvoPoint>::SharedPtr kdtree,
                 // output
                 SparseKernelMat * A_mat, SparseKernelMat * A_mat_gpu
                 );

  void dense_covariance_kernel(const CvoParams * params_gpu,
                               std::shared_ptr<CvoPointCloudGPU> points_fixed,
                               std::shared_ptr<CvoPointCloudGPU> points_moving,
                               perl_registration::cuKdTree<CvoPoint>::SharedPtr kdtree,
                               // output
                               SparseKernelMat * A_mat, SparseKernelMat * A_mat_gpu);

  /**
   * @brief computes the Lie algebra transformation elements
   *        twist = [omega; v] will be updated in this function
   */
  void compute_flow(CvoState * cvo_state, const CvoParams * params_gpu );

  void compute_step_size(CvoState * cvo_state, const CvoParams * params_cpu);


  /**
   * @brief transform cloud_y for current update
   */
  void transform_pointcloud_thrust(std::shared_ptr<CvoPointCloudGPU> init_cloud,
                                   std::shared_ptr<CvoPointCloudGPU> transformed_cloud,
                                   Mat33f * R_gpu, Vec3f * T_gpu,
                                   bool update_normal_and_cov
                                   );

  void transform_pointcloud_thrust(thrust::device_vector<CvoPoint> & init_cloud,
                                   thrust::device_vector<CvoPoint> & transformed_cloud,
                                   float * T_12_row_gpu,
                                   bool update_normal_and_cov
                                   );
  


  

  void gpu_association_to_cpu(const SparseKernelMat & association_gpu,
                              Association & association_cpu,
                              int num_source,
                              int num_target,
                              int num_neighbors=-1
                              );
}
