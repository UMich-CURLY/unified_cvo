#include "CvoFrameGPU.hpp"
#include "cvo/IRLS_State.hpp"
//#include "cvo/KDTreeVectorOfVectorsAdaptor.h"
//#include "cvo/CvoParams.hpp"
//#include "utils/PointSegmentedDistribution.hpp"
//#include "utils/CvoFrameGPU.hpp"
#include "utils/CvoPoint.hpp"
#include "CudaTypes.hpp"
//#include "utils/CvoPointCloud.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "cvo/SparseKernelMat.hpp"
//#include <ceres/ceres.h>

namespace ceres {
  class Problem;
}

namespace cvo {
  class CvoFrame;
  //using CvoPointCloudGPU = perl_registration::cuPointCloud<cvo::CvoPoint>;
  //using CuKdTree = perl_registration::cukdtree<cvo::CvoPoint>;
  class CvoParams;
  
  class BinaryStateGPU : public BinaryState {
    
  public:

    //typedef KDTreeVectorOfVectorsAdaptor<std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> >, float>  Kdtree;
    typedef std::shared_ptr<BinaryStateGPU> Ptr;
    
    //BinaryStateGPU(CvoFrame::Ptr pc1,
    //               CvoFrame::Ptr pc2,
    //               const CvoParams * params
    //               );
    BinaryStateGPU(std::shared_ptr<CvoFrameGPU> pc1,
                   std::shared_ptr<CvoFrameGPU> pc2,
                   const CvoParams * params_cpu,
                   const CvoParams * params_gpu,
                   unsigned int num_neighbor,
                   float init_ell
                   );
    ~BinaryStateGPU();

    // update 
    virtual int update_inner_product();

    void update_ell();

    //const Eigen::SparseMatrix<double, Eigen::RowMajor> & get_inner_product() {return ip_mat_;}

    void add_residual_to_problem(ceres::Problem & problem);
                                 //ceres::LocalParameterization * parameterization);

    void export_association(Association & output_association);    

    /*
    virtual void update_residuals(double * residuals) {
      if (residuals) {
        residuals[0] = residuals_;
      }
    }

    virtual void update_jacobians(double ** jacobians);
    */

  private:
    //pcl::PointCloud<pcl::PointSegmentedDistribution<>>::Ptr pc1_;
    //pcl::PointCloud<pcl::PointSegmentedDistribution<>>::Ptr pc2_;
    
    // in frame 1
    //pcl::PointCloud<pcl::PointSegmentedDistribution<>> pc2_curr_;
    //std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > pc1_curr_;
    //std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > pc2_curr_;

    std::shared_ptr<CvoFrameGPU> frame1_;
    std::shared_ptr<CvoFrameGPU> frame2_;

    CvoFrame * frame1();
    CvoFrame * frame2();

    int * cukdtree_inds_results_gpu_;
    std::shared_ptr<CvoPointCloudGPU> points_transformed_buffer_gpu_;

    
    unsigned int num_neighbors_;
    SparseKernelMat  A_host_;
    //std::shared_ptr<Kdtree> pc1_kdtree_;
    //Eigen::SparseMatrix<double, Eigen::RowMajor>  ip_mat_;
    SparseKernelMat * A_device_;

    SparseKernelMat A_result_cpu_;
    
    
    float ell_;
    int iter_;
    const unsigned int init_num_neighbors_;

    const CvoParams * params_gpu_;
    const CvoParams * params_cpu_;

    
    
  };


  
}
