#include "cvo/IRLS_State_GPU.hpp"
#include "cvo/IRLS_Cost_CPU.hpp"
#include "cvo/CvoFrameGPU.hpp"
#include "cvo/CvoFrame.hpp"
#include "cvo/CvoGPU_impl.cuh"
#include "cvo/CudaTypes.cuh"
#include "utils/CvoPointCloud.hpp"
#include "cvo/local_parameterization_se3.hpp"
#include "cupointcloud/cupointcloud.h"
#include "utils/data_type.hpp"
#include "cvo/SparseKernelMat.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>


namespace cvo {

  BinaryStateGPU::BinaryStateGPU(std::shared_ptr<CvoFrameGPU> pc1,
                                 std::shared_ptr<CvoFrameGPU> pc2,
                                 const CvoParams * params_cpu,                                 
                                 const CvoParams * params_gpu,
                                 unsigned int num_neighbor,
                                 float init_ell
                                 ) : frame1_(pc1), frame2_(pc2),
                                     params_cpu_(params_cpu),
                                     params_gpu_(params_gpu),
                                     num_neighbors_(num_neighbor),
                                     ell_(init_ell),
                                     init_num_neighbors_(num_neighbor){

    init_internal_SparseKernelMat_cpu(pc1->points->size(),  num_neighbor, &A_result_cpu_);
    A_device_ = init_SparseKernelMat_gpu(pc1->points->size(), num_neighbor, A_host_);
    clear_SparseKernelMat(&A_host_, num_neighbors_);
    //std::cout<<"Construct BinaryStateGPU: ell is "<<ell_<<", init_num_neighbors_ is "<<init_num_neighbors_<<"\n";

    if (params_cpu_->is_using_kdtree) {
      points_transformed_buffer_gpu_ = std::make_shared<CvoPointCloudGPU>(pc1->size());
      cudaMalloc((int**)&cukdtree_inds_results_gpu_, sizeof(int)*init_num_neighbors_*pc1->size());
    }

    iter_ = 0;

  }

  BinaryStateGPU::~BinaryStateGPU() {
    delete_internal_SparseKernelMat_cpu(&A_result_cpu_);
    delete_SparseKernelMat_gpu(A_device_, &A_host_);
    if (params_cpu_->is_using_kdtree) {
      cudaFree(cukdtree_inds_results_gpu_);
    }
  }

  int BinaryStateGPU::update_inner_product() {

    unsigned int last_num_neibors = max_neighbors(&A_host_);
    if (last_num_neibors > 0)
      num_neighbors_ = std::min(init_num_neighbors_, (unsigned int)(last_num_neibors*1.1));
    std::cout<< "Current num_neighbors_ is "<<num_neighbors_<<"\n";
    

    clear_SparseKernelMat(&A_host_, num_neighbors_);

    if (params_cpu_->is_using_kdtree) {

      thrust::device_vector<CvoPoint> & f1_points = frame1_->points_init_gpu()->points;
      thrust::device_vector<CvoPoint> & f2_points = frame2_->points_init_gpu()->points;
      
      //thrust::device_vector<int> cukdtree_inds_results;
      Eigen::Matrix4f T_f2_to_f1 = (frame2_->pose_cpu().inverse() * frame1_->pose_cpu()).cast<float>();
      // std::cout<<"T_f2_to_f1 is "<<T_f2_to_f1<<"\n";
      
      //thrust::device_ptr<int> inds_ptr_gpu = thrust::device_pointer_cast(cukdtree_inds_results_gpu_);
      //thrust::device_vector<int> inds_device_vec;
      //(inds_ptr_gpu, inds_ptr_gpu + num_neighbors_ * frame1_->size());
      cudaMemset(cukdtree_inds_results_gpu_, -1, num_neighbors_ * frame1_->size()  * sizeof(int));

      //std::cout<<"find_nearby_source_points_cukdtree\n";
      find_nearby_source_points_cukdtree(//const CvoParams *cvo_params,
                                         frame1_->points_init_gpu(),
                                         frame2_->kdtree(),
                                         T_f2_to_f1,
                                         num_neighbors_,
                                         // output
                                         points_transformed_buffer_gpu_,
                                         cukdtree_inds_results_gpu_
                                         //inds_device_vec
                                         );

      /*
      thrust::device_ptr<int> inds_ptr_gpu_before = thrust::device_pointer_cast(cukdtree_inds_results_gpu_);
      thrust::device_vector<int> inds_device_vec_before(inds_ptr_gpu_before, inds_ptr_gpu_before + num_neighbors_ * frame1_->size());
      std::cout<<"First few indices before: ";
      for (int k = 0; k < num_neighbors_; k++) std::cout<<inds_device_vec_before[k]<<", ";
      std::cout<<"\n";
      */

      fill_in_A_mat_cukdtree<<< (f1_points.size() / CUDA_BLOCK_SIZE)+1, CUDA_BLOCK_SIZE  >>>
        (params_gpu_,
         thrust::raw_pointer_cast(points_transformed_buffer_gpu_->points.data()),         
         f1_points.size(),
         thrust::raw_pointer_cast(f2_points.data()),
         f2_points.size(),
         //thrust::raw_pointer_cast(inds_device_vec.data()),
         cukdtree_inds_results_gpu_,
         num_neighbors_, ell_,
         A_device_);

      thrust::device_ptr<int> inds_ptr_gpu = thrust::device_pointer_cast(A_host_.ind_row2col);
      thrust::device_vector<int> inds_device_vec(inds_ptr_gpu, inds_ptr_gpu + num_neighbors_ * frame1_->size());
      thrust::device_ptr<float> A_ptr_gpu = thrust::device_pointer_cast(A_host_.mat);
      thrust::device_vector<float> A_device_vec(A_ptr_gpu, A_ptr_gpu + num_neighbors_ * frame1_->size());
      std::cout<<"First few indices after A mat: ";
      for (int k = 0; k < num_neighbors_; k++) std::cout<<"("<<inds_device_vec[k]<<": "<<A_device_vec[k]<<"), ";
      std::cout<<"\n";


    } else {

      fill_in_A_mat_gpu<<< (frame1_->points->size() / CUDA_BLOCK_SIZE)+1, CUDA_BLOCK_SIZE  >>>(
                                                                                               params_gpu_,
                                                                                               thrust::raw_pointer_cast(frame1_->points_transformed_gpu()->points.data()),
                                                                                               //frame1_->points_transformed_gpu(),
                                                                                               frame1_->points->size(),
                                                                                               thrust::raw_pointer_cast(frame2_->points_transformed_gpu()->points.data()),
                                                                                               //frame2_->points_transformed_gpu(),
                                                                                               frame2_->points->size(),
                                                                                               num_neighbors_,
                                                                                               (float)ell_,
                                                                                               A_device_
                                                                                               );
      /*
      thrust::device_ptr<int> inds_ptr_gpu = thrust::device_pointer_cast(A_host_.ind_row2col);
      thrust::device_vector<int> inds_device_vec(inds_ptr_gpu, inds_ptr_gpu + num_neighbors_ * frame1_->size());
      std::cout<<"First few indices: ";
      for (int k = 0; k < num_neighbors_; k++) std::cout<<inds_device_vec[k]<<", ";
      std::cout<<"\n";
      */
    }
    compute_nonzeros(&A_host_);
    std::cout<<"Nonzeros is "<<A_host_.nonzero_sum<<"\n";

    copy_internal_SparseKernelMat_gpu_to_cpu(&A_host_, &A_result_cpu_,
                                             num_neighbors_);

    iter_++;
    return A_result_cpu_.nonzero_sum;;
    //if (ip_mat_.nonZeros() < 100) {
    //  std::cout<<"too sparse inner product mat "<<ip_mat_.nonZeros()<<std::endl;
    //  return -1;
    //} else
    //  return 0;
    
  }

  CvoFrame * BinaryStateGPU::frame1() {
    return frame1_.get();
  }
  CvoFrame * BinaryStateGPU::frame2() {
    return frame2_.get();
  }

  void BinaryStateGPU::export_association(Association & output) {
    gpu_association_to_cpu(A_host_,
                           output,
                           frame1()->points->size(),
                           frame2()->points->size(),
                           num_neighbors_);
  }

  
}

