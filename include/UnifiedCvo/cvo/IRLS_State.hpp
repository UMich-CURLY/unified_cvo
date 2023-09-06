#pragma once
#include "utils/data_type.hpp"
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <memory>
#include "cvo/Association.hpp"
//#include <ceres/local_parameterization.h>

namespace cvo {
  class CvoFrame;

  class BinaryState {
  public:
    using Mat34d_row = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;
    using Mat4d_row = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;

    typedef std::shared_ptr<BinaryState> Ptr;
    
    virtual int update_inner_product() = 0;

    virtual void add_residual_to_problem(ceres::Problem & problem) = 0;
                                         // ceres::LocalParameterization * parameterization);

    virtual void update_ell () = 0;

    virtual void export_association(Association & output_assocation) = 0;

    virtual double get_ell() const = 0;
    virtual const CvoFrame * get_frame1() const = 0;
    virtual const CvoFrame * get_frame2() const = 0;
    
    
  };



  
}
