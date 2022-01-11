#pragma once


#include "cvo/local_parameterization_se3.hpp"
#include "utils/PointSegmentedDistribution.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/IRLS_State.hpp"
#include "utils/CvoFrame.hpp"
#include "cvo/CvoParams.hpp"
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <list>

namespace cvo {

  /*
  class CvoBinaryCost : public  ceres::CostFunction {
  public:

    CvoBinaryCost(std::shared_ptr<BinaryState> binary_state) : binary_state_(binary_state) {}
    ~CvoBinaryCost() {}

    virtual bool Evaluate(double const* const* pose_vec_raw_two,
                          double* residuals,
                          double** jacobians) const  {
      Eigen::Map<
        Eigen::Matrix<double, 3, 4, Eigen::RowMajor> const> T1(pose_vec_raw_two[0]);
      Eigen::Map<
        Eigen::Matrix<double, 3, 4, Eigen::RowMajor> const> T2((pose_vec_raw_two[1]));

      binary_state_->update_inner_product(T1, T2);
      if (residuals)
        binary_state_->update_residuals(residuals);

      if (jacobians && jacobians[0]) {
        binary_state_->update_jacobians(jacobians);
      }
      return true;
    }



  private:
    std::shared_ptr<BinaryState> binary_state_; 
  };
  */

  class CvoBatchIRLS {
  public:

    using Mat34d_row = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;
    
    CvoBatchIRLS(const std::vector<CvoFrame::Ptr> & frames,
                 const std::vector<bool> & pivot_flags,
                 const std::list<BinaryState::Ptr> & states,
                 const CvoParams * params
                 );

    void solve();

    void remove_frame(CvoFrame::Ptr to_remove);
    void add_frame(CvoFrame::Ptr to_remove);


  private:
    const std::list<BinaryState::Ptr> * states_;
    //std::unique_ptr<ceres::Problem> problem_;
    
    const std::vector<CvoFrame::Ptr> * frames_;
    //CvoFrame * const pivot_;
    const std::vector<bool> * pivot_flags_; 
    const CvoParams * params_;


  };

  
  

}
