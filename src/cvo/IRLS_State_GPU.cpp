#include "cvo/IRLS_State_GPU.hpp"
#include "cvo/IRLS_Cost_CPU.hpp"
#include "cvo/CvoFrame.hpp"
#include "cvo/CvoFrameGPU.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoParams.hpp"
#include <ceres/ceres.h>
namespace cvo {
  
  unsigned int BinaryStateGPU::add_residual_to_problem(ceres::Problem & problem) {

    auto & pc1 = frame1_->points->positions();
    auto & pc2 = frame2_->points->positions();

    std::cout<<"Nonzeros is "<<A_result_cpu_.nonzero_sum<<std::endl;
    unsigned int num_residuals = 0;
    for (int r=0; r< A_result_cpu_.rows; ++r)
    {
      for (int c=0; c<std::min(params_cpu_->nearest_neighbors_max, num_neighbors_); c++){

        int idx1 = r;
        int idx2 = A_result_cpu_.ind_row2col[r*num_neighbors_+c];
        if (idx2 == -1)
          break;
        
        double label_ip = A_result_cpu_.mat[r*num_neighbors_+c];

        //if (r == 0)
        //  std::cout<<"add_residual_to_problem: pc1[0] with pc2["<<idx2<<"] and label_ip is "<<label_ip<<std::endl;
        
        ceres::CostFunction * cost_per_point = nullptr;

        if (params_cpu_->multiframe_is_optimizing_ell) {
          cost_per_point = new ceres::AutoDiffCostFunction<PairwisePoseEllAutoDiffFunctor, 1, 12, 12, 1>(new PairwisePoseEllAutoDiffFunctor(pc1[idx1],
                                                                                                                                     pc2[idx2],
                                                                                                                                     label_ip,
                                                                                                                                     params_cpu_->sigma));
          problem.AddResidualBlock(cost_per_point, nullptr , frame1_->pose_vec, frame2_->pose_vec, &ell_);                    
        }
        else {
          cost_per_point = new PairwiseAnalyticalDiffFunctor(pc1[idx1], pc2[idx2], label_ip ,  ell_);
          problem.AddResidualBlock(cost_per_point, nullptr , frame1_->pose_vec, frame2_->pose_vec);          
        }
        num_residuals++;
        //ceres::CostFunction * cost_per_point =
        //  new PairwiseAnalyticalDiffFunctor(pc1[idx1], pc2[idx2], label_ip / ell_ * 1000 / A_result_cpu_.nonzero_sum ,  ell_);
        
        /* ceres::CostFunction* cost_per_point
          = new ceres::AutoDiffCostFunction<PairwiseAutoDiffFunctor, 1, 12, 12>(new PairwiseAutoDiffFunctor(pc1[idx1],
                                                                                                            pc2[idx2],
                                                                                                            color_ip,
                                                                                                            ell_,
                                                                                                            params_->sigma));  
        */
        
        //ceres::LossFunctionWrapper* loss_function(new ceres::HuberLoss(1.0), ceres::TAKE_OWNERSHIP);

      }
    }

    if (params_cpu_->multiframe_is_optimizing_ell) {
      for (int r=0; r< A_f1_cpu_.rows; ++r)
      {
        for (int c=0; c<std::min(params_cpu_->nearest_neighbors_max, num_neighbors_f1_); c++){

          int idx1 = r;
          int idx2 = A_f1_cpu_.ind_row2col[r*num_neighbors_f1_+c];
          if (idx2 == -1)
            break;
        
          double label_ip = A_f1_cpu_.mat[r*num_neighbors_f1_+c];
          ceres::CostFunction * cost_per_point = nullptr;
          cost_per_point = new ceres::AutoDiffCostFunction<SelfPoseEllAutoDiffFunctor, 1, 1>(new SelfPoseEllAutoDiffFunctor(pc1[idx1],
                                                                                                                                        pc1[idx2],
                                                                                                                                        label_ip,
                                                                                                                                        params_cpu_->sigma));
          problem.AddResidualBlock(cost_per_point, nullptr , &ell_);
          num_residuals++;

        }
      }
      for (int r=0; r< A_f2_cpu_.rows; ++r)
      {
        for (int c=0; c<std::min(params_cpu_->nearest_neighbors_max,num_neighbors_f2_); c++){

          int idx1 = r;
          int idx2 = A_f2_cpu_.ind_row2col[r*num_neighbors_f2_+c];
          if (idx2 == -1)
            break;
        
          double label_ip = A_f2_cpu_.mat[r*num_neighbors_f2_+c];
          ceres::CostFunction * cost_per_point = nullptr;          
          cost_per_point = new ceres::AutoDiffCostFunction<SelfPoseEllAutoDiffFunctor, 1, 1>(new SelfPoseEllAutoDiffFunctor(pc2[idx1],
                                                                                                                                        pc2[idx2],
                                                                                                                                        label_ip,
                                                                                                                                        params_cpu_->sigma));
          problem.AddResidualBlock(cost_per_point, nullptr , &ell_);
          num_residuals++;

        }
      }
      
      
    }
    return num_residuals;

  }

  
  void BinaryStateGPU::update_ell() {
    if (ell_ > params_cpu_->multiframe_ell_min) {
      if (!params_cpu_->multiframe_is_optimizing_ell)  {      
        ell_ = ell_ * params_cpu_->multiframe_ell_decay_rate;
      }
    } else  {
      ell_ = params_cpu_->multiframe_ell_min;
    }
  }
  

}
