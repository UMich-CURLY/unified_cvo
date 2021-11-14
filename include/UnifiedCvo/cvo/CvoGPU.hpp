

#pragma once
#include "utils/data_type.hpp"
#include "cvo/CvoParams.hpp"

#include "utils/CvoPointCloud.hpp"
#include "utils/CvoPoint.hpp"
#include "utils/CvoFrame.hpp"

#include <vector>
#include <string.h>
#include <iostream>
#include <memory>
#include <utility>
#include <future>
#include <thread>

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/Cholesky> 
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/StdVector>


namespace cvo{
  
  class CvoGPU{

  private:
    // all the parameters, allocated on gpu
    CvoParams * params_gpu;
    CvoParams params;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // constructor and destructor
    CvoGPU(const std::string & f);
    ~CvoGPU();
    CvoParams & get_params() {return params;}
    void write_params(CvoParams * p_cpu);
    
    int align(// inputs
              const CvoPointCloud& source_points,
              const CvoPointCloud& target_points,
              const Eigen::Matrix4f & T_target_frame_to_source_frame,
              // outputs
              Eigen::Ref<Eigen::Matrix4f> transform,
              double *registration_seconds=nullptr ) const;

    int align(// inputs
              const pcl::PointCloud<CvoPoint>& source_points,
              const pcl::PointCloud<CvoPoint>& target_points,
              const Eigen::Matrix4f & T_target_frame_to_source_frame,
              // outputs
              Eigen::Ref<Eigen::Matrix4f> transform,
              double *registration_seconds=nullptr ) const;

    int align(// inputs
              std::vector<CvoFrame::Ptr> & frames,  // point clouds, poses, the outputs are within
              const std::list<std::pair<CvoFrame::Ptr, CvoFrame::Ptr>> & edges,
              // outputs
              double *registration_seconds=nullptr
              ) const;

    

    float function_angle(const CvoPointCloud& source_points,
                         const CvoPointCloud& target_points,
                         const Eigen::Matrix4f & T_target_frame_to_source_frame,
                         bool is_approximate=true,
                         bool is_gpu=true) const;
    float function_angle(const pcl::PointCloud<CvoPoint>& source_points,
                         const pcl::PointCloud<CvoPoint>& target_points,
                         const Eigen::Matrix4f & T_target_frame_to_source_frame,
                         bool is_approximate=true) const;

    
    float inner_product_gpu(const CvoPointCloud& source_points,
                            const CvoPointCloud& target_points,
                            const Eigen::Matrix4f & T_target_frame_to_source_frame
                            ) const;
    float inner_product_gpu(const pcl::PointCloud<CvoPoint>& source_points_pcl,
                            const pcl::PointCloud<CvoPoint>& target_points_pcl,
                            const Eigen::Matrix4f & init_guess_transform
                            ) const;
    float inner_product_cpu(const CvoPointCloud& source_points,
                            const CvoPointCloud& target_points,
                            const Eigen::Matrix4f & T_target_frame_to_source_frame
                            ) const;

  };

  void CvoPointCloud_to_pcl(const CvoPointCloud & cvo_pcd,
                            pcl::PointCloud<CvoPoint> & out_pcl);
  
}


