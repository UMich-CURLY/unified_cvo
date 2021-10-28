#include "cvo/IRLS_State.hpp"
#include "cvo/KDTreeVectorOfVectorsAdaptor.h"
#include "cvo/CvoParams.hpp"
#include "utils/PointSegmentedDistribution.hpp"
#include "utils/CvoFrame.hpp"
#include "utils/CvoPointCloud.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace cvo {

  class BinaryStateCPU : public BinaryState {
    
  public:

    typedef KDTreeVectorOfVectorsAdaptor<std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> >, float>  Kdtree;
    typedef std::shared_ptr<BinaryStateCPU> Ptr;
    
    BinaryStateCPU(CvoFrame::Ptr pc1,
                   CvoFrame::Ptr pc2,
                   const CvoParams * params
                   );

    // update 
    virtual void update_inner_product();

    void update_ell();

    const Eigen::SparseMatrix<double, Eigen::RowMajor> & get_inner_product() {return ip_mat_;}

    void add_residual_to_problem(ceres::Problem & problem);
                                 //ceres::LocalParameterization * parameterization);

    /*
    virtual void update_residuals(double * residuals) {
      if (residuals) {
        residuals[0] = residuals_;
      }
    }

    virtual void update_jacobians(double ** jacobians);
    */

    CvoFrame::Ptr frame1;
    CvoFrame::Ptr frame2;     

  private:
    //pcl::PointCloud<pcl::PointSegmentedDistribution<>>::Ptr pc1_;
    //pcl::PointCloud<pcl::PointSegmentedDistribution<>>::Ptr pc2_;
    
    // in frame 1
    //pcl::PointCloud<pcl::PointSegmentedDistribution<>> pc2_curr_;
    //std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > pc1_curr_;
    //std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > pc2_curr_;

    std::shared_ptr<Kdtree> pc1_kdtree_;
    Eigen::SparseMatrix<double, Eigen::RowMajor>  ip_mat_;
    
    double ell_;
    int iter_;

    const CvoParams * params_;
    
  };


  
}
