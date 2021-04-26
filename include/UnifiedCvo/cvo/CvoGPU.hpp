

#pragma once
#include "utils/data_type.hpp"
#include "cvo/CvoParams.hpp"

#include "utils/CvoPointCloud.hpp"
#include "utils/CvoPoint.hpp"
//#include "utils/PointSegmentedDistribution.hpp"

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



#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int



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
    
    /**
     * @brief align two rgbd pointcloud
     *        the function will iterate MAX_ITER times unless break conditions are met
     *        return 0 if sucess. return -1 if fails
     */
    int align(// inputs
              const CvoPointCloud& source_points,
              const CvoPointCloud& target_points,
              const Eigen::Matrix4f & init_guess_transform,
              // outputs
              Eigen::Ref<Eigen::Matrix4f> transform,
              double *registration_seconds=nullptr ) const;

    int align(// inputs
              const pcl::PointCloud<CvoPoint>& source_points,
              const pcl::PointCloud<CvoPoint>& target_points,
              const Eigen::Matrix4f & init_guess_transform,
              // outputs
              Eigen::Ref<Eigen::Matrix4f> transform,
              double *registration_seconds=nullptr ) const;

    

    // callable after each align
    float inner_product(const CvoPointCloud& source_points,
                        const CvoPointCloud& target_points,
                        const Eigen::Matrix4f & source_frame_to_target_frame) const;

    float indicator_value(const CvoPointCloud& source_points,
                      const CvoPointCloud& target_points,
                      const Eigen::Matrix4f & init_guess_transform,
                      Eigen::Ref<Eigen::Matrix4f> transform) const;

    float function_angle(const CvoPointCloud& source_points,
                        const CvoPointCloud& target_points,
                        const Eigen::Matrix4f & source_frame_to_target_frame) const;

    float real_inner_product(const CvoPointCloud& source_points,
                             const CvoPointCloud& target_points,
                             const Eigen::Matrix4f & source_frame_to_target_frame) const;

    float label_only_inner_product(const CvoPointCloud & source_points,
                                   const CvoPointCloud & target_points) const;

  };

  void CvoPointCloud_to_pcl(const CvoPointCloud & cvo_pcd,
                            pcl::PointCloud<CvoPoint> & out_pcl);
  
}


