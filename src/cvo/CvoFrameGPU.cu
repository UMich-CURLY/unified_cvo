// #include "cvo/CvoFrameGPU.cuh"
#include "cvo/CudaTypes.cuh"
#include "cvo/CvoFrameGPU.hpp"
#include "cvo/CvoGPU_impl.cuh"

#include <memory>

namespace cvo {

  /*
  CvoFrameGPU_Impl::CvoFrameGPU_Impl(const CvoPointCloud * pts,
                                     const double poses[12],
                                     bool is_using_kdtree) :
    points_transformed_gpu_(pts->size()){
    points_init_gpu_ = CvoPointCloud_to_gpu(*pts);
    point_transformed_gpu_.reset(new CvoPointCloudGPU(pts->size()));
      
    cudaMalloc((void**)&pose_vec_gpu_, sizeof(float)*12);
    float pose_float[12];
    #pragma omp parallel for
    for (int i = 0; i < 12; i++)
      pose_float[i] = static_cast<float>(poses[i]);
    cudaMemcpy((void*)pose_vec_gpu_, (void*)pose_float, sizeof(float)*12, cudaMemcpyHostToDevice);

    if (is_using_kdtree) {
      //kdtree.reset(new perl_registration::cuKdTree<CvoPoint> );
      kdtree.SetInputCloud(points_init_gpu_);
      //kdtree_inds_results.resize(cvo_params.is_using_kdtree * num_fixed);
    }
    
    
    }*/

  CvoFrameGPU::CvoFrameGPU(const CvoPointCloud * pts,
                           const double poses[12],
                           bool is_using_kdtree) :
    CvoFrame(pts, poses, is_using_kdtree)
  {
    //    cudaMalloc((void**)&points_init_gpu_, sizeof(CvoPoint)*pts->size() );
    //cudaMalloc((void**)&points_transformed_gpu_, sizeof(CvoPoint)*pts->size() );
    points_init_gpu_ = CvoPointCloud_to_gpu(*pts);
    points_transformed_gpu_ = std::make_shared<CvoPointCloudGPU>((pts->size()));
      
    cudaMalloc((void**)&pose_vec_gpu_, sizeof(float)*12);
    float pose_float[12];
    #pragma omp parallel for
    for (int i = 0; i < 12; i++)
      pose_float[i] = static_cast<float>(poses[i]);
    cudaMemcpy((void*)pose_vec_gpu_, (void*)pose_float, sizeof(float)*12, cudaMemcpyHostToDevice);

    if (is_using_kdtree) {
      //kdtree.reset(new perl_registration::cuKdTree<CvoPoint> );
      kdtree_ = std::make_shared<CuKdTree>();
      kdtree_->SetInputCloud(points_init_gpu_);
      //kdtree_inds_results.resize(cvo_params.is_using_kdtree * num_fixed);
    }
    
    
  }
  
  CvoFrameGPU::~CvoFrameGPU() {
    cudaFree((void*)pose_vec_gpu_);    
  }

  CuKdTree & CvoFrameGPU::kdtree() const {
    return *kdtree_;
  }

  /*
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
  */
  /*
  const CvoPoint * CvoFrameGPU::points_transformed_gpu() {
    return points_transformed_gpu_;
  }

  const CvoPoint * CvoFrameGPU::points_init_gpu() {
    return points_init_gpu_;
  }
  */
  std::shared_ptr<CvoPointCloudGPU> CvoFrameGPU::points_transformed_gpu() {
    //return thrust::raw_pointer_cast(points_transformed_gpu_.data());
    return points_transformed_gpu_;
  }
  std::shared_ptr<CvoPointCloudGPU> CvoFrameGPU::points_init_gpu() {
    //return thrust::raw_pointer_cast(points_transformed_gpu_.data());
    return points_init_gpu_;
  }


  /*
  const CvoPoint * CvoFrameGPU_Impl::points_transformed_gpu() const {
    return thrust::raw_pointer_cast(points_transformed_gpu_.data());
    }*/
  

  //const void CvoFrameGPU_Impl::set_pose_vec_gpu(float * pose_new) {
  // }

  void CvoFrameGPU::transform_pointcloud() {
  //  impl->transform_pointcloud_from_input_pose(pose_vec);

    float pose_float_cpu[12];
    
    #pragma omp parallel for
    for (int i = 0; i < 12; i++) {
      pose_float_cpu[i] = static_cast<float>(pose_vec[i]);
    }
    
    cudaMemcpy((void*)pose_vec_gpu_, (void*)pose_float_cpu, sizeof(float)*12, cudaMemcpyHostToDevice);

    transform_pointcloud_thrust(points_init_gpu_,
                                points_transformed_gpu_,
                                pose_vec_gpu_,
                                false
                                );
    
  }

  const float * CvoFrameGPU::pose_vec_gpu() const {
    return pose_vec_gpu_;
  }

  size_t CvoFrameGPU::size() const {
    return points_init_gpu_->size();
  }
 

  /*const float * CvoFrameGPU_Impl::pose_vec_gpu() const {
    return pose_vec_gpu_;
  }
  
  

  const CvoPoint * CvoFrameGPU::points_transformed_gpu() const {
    return impl->points_transformed_gpu();
  }
  */
  //void CvoFrameGPU::set_pose_vec_gpu(float* new_pose) {
  //  impl->set_pose_vec_gpu(new_pose);
  //} 

  //const thrust::device_vector<CvoPoint>& CvoFrameGPU::points_init_gpu() {
  //  return points_init_gpu_;
  //}
  
}
