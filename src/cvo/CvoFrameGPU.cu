#include "cvo/CvoFrameGPU.cuh"
#include "cvo/CvoFrameGPU.hpp"
#include "cvo/CvoGPU_impl.cuh"

namespace cvo {

  CvoFrameGPU_Impl::CvoFrameGPU_Impl(const CvoPointCloud * pts,
                                     const double poses[12]) :
    points_transformed_gpu_(pts->size()){
    CvoPointCloud_to_gpu(*pts, points_init_gpu_);
    cudaMalloc((void**)&pose_vec_gpu_, sizeof(float)*12);

    float pose_float[12];
    #pragma omp parallel for
    for (int i = 0; i < 12; i++)
      pose_float[i] = static_cast<float>(poses[i]);
    
    cudaMemcpy((void*)pose_vec_gpu_, (void*)pose_float, sizeof(float)*12, cudaMemcpyHostToDevice);
    
  }

  CvoFrameGPU::CvoFrameGPU(const CvoPointCloud * pts,
                           const double poses[12]) :
    CvoFrame(pts, poses),
    impl(new CvoFrameGPU_Impl(pts, poses))
  {
    //    cudaMalloc((void**)&points_init_gpu_, sizeof(CvoPoint)*pts->size() );
    //cudaMalloc((void**)&points_transformed_gpu_, sizeof(CvoPoint)*pts->size() );
    
  }
  
  CvoFrameGPU_Impl::~CvoFrameGPU_Impl() {
    cudaFree((void*)pose_vec_gpu_);
    //cudaFree((void*)points_init_gpu_);
    //cudaFree((void*)points_transformed_gpu_);
  }

  CvoFrameGPU::~CvoFrameGPU() {
    
  }
  

  
  void CvoFrameGPU_Impl::transform_pointcloud_from_input_pose(const double * pose_vec_cpu ) {

    float pose_float_cpu[12];
    
    #pragma omp parallel for
    for (int i = 0; i < 12; i++) {
      pose_float_cpu[i] = static_cast<float>(pose_vec_cpu[i]);
    }
    
    cudaMemcpy((void*)pose_vec_gpu_, (void*)pose_float_cpu, sizeof(float)*12, cudaMemcpyHostToDevice);

    transform_pointcloud_thrust(points_init_gpu_,
                                points_transformed_gpu_,
                                pose_vec_gpu_,
                                false
                                );
    
  }

  /*
  const CvoPoint * CvoFrameGPU::points_transformed_gpu() {
    return points_transformed_gpu_;
  }

  const CvoPoint * CvoFrameGPU::points_init_gpu() {
    return points_init_gpu_;
  }
  */
  const CvoPoint * CvoFrameGPU_Impl::points_transformed_gpu() const {
    return thrust::raw_pointer_cast(points_transformed_gpu_.data());
  }

  //const void CvoFrameGPU_Impl::set_pose_vec_gpu(float * pose_new) {
  // }

  void CvoFrameGPU::transform_pointcloud() {
    impl->transform_pointcloud_from_input_pose(pose_vec);
  }

  const float * CvoFrameGPU::pose_vec_gpu() const {
    return impl->pose_vec_gpu();
  }
 

  const float * CvoFrameGPU_Impl::pose_vec_gpu() const {
    return pose_vec_gpu_;
  }
 
  

  const CvoPoint * CvoFrameGPU::points_transformed_gpu() const {
    return impl->points_transformed_gpu();
  }

  //void CvoFrameGPU::set_pose_vec_gpu(float* new_pose) {
  //  impl->set_pose_vec_gpu(new_pose);
  //} 

  //const thrust::device_vector<CvoPoint>& CvoFrameGPU::points_init_gpu() {
  //  return points_init_gpu_;
  //}
  
}
