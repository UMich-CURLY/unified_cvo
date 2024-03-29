#include "cvo/IRLS_State_GPU.hpp"
#include "cvo/IRLS_Cost_CPU.hpp"
#include "cvo/CvoFrame.hpp"
#include "cvo/CvoFrameGPU.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoParams.hpp"
#include <ceres/ceres.h>
namespace cvo {
  
  void BinaryStateGPU::add_residual_to_problem(ceres::Problem & problem) {

    auto & pc1 = frame1()->points->positions();
    auto & pc2 = frame2()->points->positions();

    std::cout<<"Nonzeros is "<<A_result_cpu_.nonzero_sum<<std::endl;
    
    for (int r=0; r< A_result_cpu_.rows; ++r)
    {
      for (int c=0; c<num_neighbors_; c++){

        int idx1 = r;
        int idx2 = A_result_cpu_.ind_row2col[r*num_neighbors_+c];
        if (idx2 == -1)
          break;
        
        double label_ip = A_result_cpu_.mat[r*num_neighbors_+c];

        //if (r == 0)
        //  std::cout<<"add_residual_to_problem: pc1[0] with pc2["<<idx2<<"] and label_ip is "<<label_ip<<std::endl;
        
        ceres::CostFunction * cost_per_point =
          new PairwiseAnalyticalDiffFunctor(pc1[idx1], pc2[idx2], label_ip ,  ell_);
        
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
        problem.AddResidualBlock(cost_per_point, nullptr , frame1()->pose_vec, frame2()->pose_vec);
      }
    }


  }

  
  void BinaryStateGPU::update_ell() {
    if (ell_ > params_cpu_->multiframe_ell_min)
      ell_ = ell_ * params_cpu_->multiframe_ell_decay_rate;     
  }
  

}
