#include "cvo/IRLS_State_GPU.hpp"
#include "cvo/IRLS_Cost_CPU.hpp"
#include "cvo/CvoFrameGPU.hpp"
#include "cvo/CvoFrame.hpp"
#include "cvo/CvoGPU_impl.cuh"
#include "utils/CvoPointCloud.hpp"
#include "cvo/local_parameterization_se3.hpp"
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
      points_transformed_buffer_gpu_.reset(new CvoPointCloudGPU);
      cudaMalloc((int**)&cukdtree_inds_results_gpu_, sizeof(int)*init_num_neighbors_*pc2->size());
    
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
    //std::cout<< "Current num_neighbors_ is "<<num_neighbors_<<"\n";
    

    clear_SparseKernelMat(&A_host_, num_neighbors_);

    if (params_cpu_->is_using_kdtree) {
      //thrust::device_vector<int> cukdtree_inds_results;
      Eigen::Matrix4f T_f2_to_f1 = frame2_->pose_cpu().inverse() * frame1_->pose_cpu();

      find_nearby_source_points_cukdtree(//const CvoParams *cvo_params,
                                         frame1_->points_init_gpu(),
                                         frame2_->kdtree(),
                                         T_f2_to_f1,
                                         num_neighbors_,
                                         // output
                                         points_transformed_buffer_gpu_,
                                         cukdtree_inds_results_gpu_
                                         );


      thrust::device_vector<CvoPoint> & f1_points = frame1_->points_init_gpu()->points;
      thrust::device_vector<CvoPoint> & f2_points = frame2_->points_init_gpu()->points;

      fill_in_A_mat_cukdtree<<< (f1_points.size() / CUDA_BLOCK_SIZE)+1, CUDA_BLOCK_SIZE  >>>
        (params_gpu_,
         points_transformed_buffer_gpu_,         
         f1_points.size(),
         thrust::raw_pointer_cast(f2_points.data()),
         f2_points.size(),
         cukdtree_inds_results_gpu_,
         num_neighbors_, ell,
         A_device_);

    } else {

      fill_in_A_mat_gpu<<< (frame1_->points->size() / CUDA_BLOCK_SIZE)+1, CUDA_BLOCK_SIZE  >>>(
                                                                                               params_gpu_,
                                                                                               //thrust::raw_pointer_cast(frame1_->points_transformed_gpu().data()),
                                                                                               frame1_->points_transformed_gpu(),
                                                                                               frame1_->points->size(),
                                                                                               //thrust::raw_pointer_cast(frame2_->points_transformed_gpu().data()),
                                                                                               frame2_->points_transformed_gpu(),
                                                                                               frame2_->points->size(),
                                                                                               num_neighbors_,
                                                                                               (float)ell_,
                                                                                               A_device_
                                                                                               );
    }
    compute_nonzeros(&A_host_);

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

