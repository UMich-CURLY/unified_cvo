#pragma once
#include "utils/data_type.hpp"
#include "cvo/CvoParams.hpp"
#include "cvo/Association.hpp"
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
    /**
     * CvoGPU constructor
     * @brief Constructor.
     * @param f the filename of the yaml config file 
     */   
    CvoGPU(const std::string & f);
    ~CvoGPU();
    CvoParams & get_params() {return params;}
    void write_params(const CvoParams * p_cpu);

    
    /**
     * @brief Aligning two point clouds in the CvoPointCloud format
     * @param source_points The first point cloud
     * @param target_points The second point cloud
     * @param T_target_frame_to_source_frame The transformation from the second frame to the 
     *                                       first frame.
     * @param transform The resulting transformation from the first frame to the second frame
     * @param association The resulting data correspondence
     * @param registration_seconds The running time in seconds
     * @return 0 if sucessful, otherwise failure
     */
    int align(// inputs
              const CvoPointCloud& source_points,
              const CvoPointCloud& target_points,
              const Eigen::Matrix4f & T_target_frame_to_source_frame,
              // outputs
              Eigen::Ref<Eigen::Matrix4f> transform,
              Association * association=nullptr,
              double *registration_seconds=nullptr ) const;

    /**
     * @brief Aligning two point clouds in the pcl::PointCloud format
     * @param source_points The first point cloud
     * @param target_points The second point cloud
     * @param T_target_frame_to_source_frame The transformation from the second frame to the 
     *                                       first frame.
     * @param transform The resulting transformation from the first frame to the second frame
     * @param association The resulting data correspondence
     * @param registration_seconds The running time in seconds
     * @return 0 if sucessful, otherwise failure
     */
    int align(// inputs
              const pcl::PointCloud<CvoPoint>& source_points,
              const pcl::PointCloud<CvoPoint>& target_points,
              const Eigen::Matrix4f & T_target_frame_to_source_frame,
              // outputs
              Eigen::Ref<Eigen::Matrix4f> transform,
              Association * association=nullptr,              
              double *registration_seconds=nullptr ) const;


    /**
     * @brief Aligning multiple point clouds
     * @param frames The vector of CvoFrame, each containing a CvoPointCloud and an initial pose. 
     *               The registration results will be recorded in the pose_vec attribute of CvoFrame. 
     * @param edges The list of connection between different frames. Each edge connects two.
     * @param registration_seconds The running time in seconds
     * @return 0 if sucessful, otherwise failure
     */
    int align(// inputs
              std::vector<CvoFrame::Ptr> & frames,  // point clouds, poses, the outputs are within
              const std::list<std::pair<CvoFrame::Ptr, CvoFrame::Ptr>> & edges,
              // outputs
              double *registration_seconds=nullptr
              ) const;

    
    /**
     * @brief the function_angle measures the overlap of the two point clouds, via 
     *
     *                       cos(theta) = <f(X), f(Y)> / ||f(X)|| / ||f(Y)||
     *
     * @param source_points The first point cloud
     * @param target_points The second point cloud
     * @param T_target_frame_to_source_frame The transformation from the second frame to the 
     *                                       first frame.
     * @param ell The lengthscale used for the evaluation
     * @param is_approximate If true, it uses <f(X), f(Y)>/|X|/|Y|. othewise, it uses 
     *                       <f(X), f(Y)> / ||f(X)|| / ||f(Y)||
     * @param is_gpu If true, running this on gpu, otherwise on cpu
     * @return the float number of this cos calcuation
     */
    float function_angle(const CvoPointCloud& source_points,
                         const CvoPointCloud& target_points,
                         const Eigen::Matrix4f & T_target_frame_to_source_frame,
                         float ell,
                         bool is_approximate=true,
                         bool is_gpu=true) const;

    /**
     * @brief the function_angle measures the overlap of the two point clouds, via 
     *
     *                       cos(theta) = <f(X), f(Y)> / ||f(X)|| / ||f(Y)||
     *
     * @param source_points The first point cloud
     * @param target_points The second point cloud
     * @param T_target_frame_to_source_frame The transformation from the second frame to the 
     *                                       first frame.
     * @param ell The lengthscale used for the evaluation
     * @param is_approximate If true, it uses <f(X), f(Y)>/|X|/|Y|. othewise, it uses 
     *                       <f(X), f(Y)> / ||f(X)|| / ||f(Y)||
     * @return the float number of this cos calcuation
     */
    float function_angle(const pcl::PointCloud<CvoPoint>& source_points,
                         const pcl::PointCloud<CvoPoint>& target_points,
                         const Eigen::Matrix4f & T_target_frame_to_source_frame,
                         float ell,
                         bool is_approximate=true) const;


    /**
     * @brief the inner product <f(X), f(Y)> will return all the pairs of points {(x_i, y_j)}
     *        that are close and have similar appearance.
     * @param source_points The first point cloud
     * @param target_points The second point cloud
     * @param T_target_frame_to_source_frame The transformation from the second frame to the 
     *                                       first frame.
     * @param lengthscale The lengthscale used for the evaluation
     * @param association This is the resulting association, containing the inliers of the 
     *                    registration as well as all pairs of associated points 
     */
    void compute_association_gpu(const CvoPointCloud& source_points,
                                 const CvoPointCloud& target_points,
                                 const Eigen::Matrix4f & T_target_frame_to_source_frame,
                                 float lengthscale,
                                 // output
                                 Association & association
                                 ) const;

    /**
     * @brief the inner product <f(X), f(Y)> will return all the pairs of points {(x_i, y_j)}
     *        that are close and have similar appearance.
     * @param source_points The first point cloud
     * @param target_points The second point cloud
     * @param T_target_frame_to_source_frame The transformation from the second frame to the 
     *                                       first frame.
     * @param non_isotropic_kernel The geometric kernel for geometric association
     * @param association This is the resulting association, containing the inliers of the 
     *                    registration as well as all pairs of associated points 
     */
    void compute_association_gpu(const CvoPointCloud& source_points,
                                 const CvoPointCloud& target_points,
                                 const Eigen::Matrix4f & T_target_frame_to_source_frame,
                                 const Eigen::Matrix3f & non_isotropic_kernel,
                                 // output
                                 Association & association
                                 ) const;
    

    
    /// The below are helper functions
    float inner_product_gpu(const CvoPointCloud& source_points,
                            const CvoPointCloud& target_points,
                            const Eigen::Matrix4f & T_target_frame_to_source_frame,
                            float ell
                            ) const;
    float inner_product_gpu(const pcl::PointCloud<CvoPoint>& source_points_pcl,
                            const pcl::PointCloud<CvoPoint>& target_points_pcl,
                            const Eigen::Matrix4f & init_guess_transform,
                            float ell
                            ) const;
    float inner_product_cpu(const CvoPointCloud& source_points,
                            const CvoPointCloud& target_points,
                            const Eigen::Matrix4f & T_target_frame_to_source_frame,
                            float ell
                            ) const;


  };

  void CvoPointCloud_to_pcl(const CvoPointCloud & cvo_pcd,
                            pcl::PointCloud<CvoPoint> & out_pcl);
  
}


