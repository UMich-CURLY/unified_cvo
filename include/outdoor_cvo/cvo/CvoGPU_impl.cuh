#pragma once




#include "cvo/SparseKernelMat.cuh"
#include "cvo/CvoState.cuh"
#include "cvo/KDTreeVectorOfVectorsAdaptor.h"
#include "cvo/LieGroup.h"
#include "cvo/nanoflann.hpp"
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

//#define IS_USING_KDTREE

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
  const int CUDA_BLOCK_SIZE = 256;
  typedef Eigen::Matrix<float, KDTREE_K_SIZE, 1> VecKDf;
  typedef Eigen::Matrix<float, 1, KDTREE_K_SIZE> VecKDf_row;
  typedef Eigen::Matrix<double, 1, KDTREE_K_SIZE> VecKDd_row;
  typedef Eigen::Matrix<float, KDTREE_K_SIZE, 3> MatKD3f;
  typedef Eigen::Matrix<double, KDTREE_K_SIZE, 3> MatKD3d;

  CvoPointCloudGPU::SharedPtr CvoPointCloud_to_gpu(const CvoPointCloud& cvo_cloud );

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
                                   Mat33f * R_gpu, Vec3f * T_gpu
                                   ) ;
}
