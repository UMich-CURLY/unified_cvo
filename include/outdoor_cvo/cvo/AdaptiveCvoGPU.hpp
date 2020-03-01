

#pragma once
#include "utils/data_type.hpp"
#include "cvo/CvoParams.hpp"

//#include "cupointcloud/cupointcloud.h"
//#include "cukdtree/cukdtree.h"
#include "utils/CvoPointCloud.hpp"
#include "utils/PointSegmentedDistribution.hpp"
// #include "pcd_generator.hpp"


#include <vector>
#include <string.h>
#include <iostream>
#include <memory>
#include <utility>
#include <future>
#include <thread>

// #include <pcl/filters/filter.h>
// #include <pcl/io/pcd_io.h>
//#include <pcl/common/transforms.h>
//#include <pcl/visualization/cloud_viewer.h>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/Cholesky> 
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/StdVector>
//#include <opencv2/core/mat.hpp>

// #include <omp.h>



//#define IS_USING_SEMANTICS
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE
//using namespacstd move unique_ptr returne std;
//using namespace nanoflann;

namespace cvo{

  
  //namespace cukdtree = perl_registration;

  
  typedef Eigen::Triplet<float> Trip_t;
  typedef Eigen::Vector3f Vector3f;

  typedef Eigen::Vector3d Vector3d;


   
  
  class AdaptiveCvoGPU{

  private:
    // all the parameters, allocated on gpu
    CvoParams * params_gpu;
    CvoParams params;

    //    Eigen::Matrix3d R;   // orientation
    //Eigen::Vector3d T;   // translation
    //Eigen::Vector3d omega;  // so(3) part of twist
    //Eigen::Vector3d v;      // R^3 part of twist

    // variables for cloud manipulations




  public:
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    // public variables
    bool init;          // initialization indicator
    //int iter;           // final iteration for display
    //Eigen::Affine3f transform;  // transformation matrix
    //Eigen::Affine3f prev_transform;
    //Eigen::Affine3f accum_tf;       // accumulated transformation matrix for trajectory
    //Eigen::Affine3f accum_tf_vis;

    //bool debug_print;
    FILE * relative_transform_file;
    FILE * init_guess_file;

    //float get_stop_criteria() {return eps_2;}
    //void set_stop_criteria(float eps_2_new) {eps_2 = eps_2_new;}
    

  private:
    // private functions
        
    /**
     * @brief a polynomial root finder
     * @param coef: coefficeints of the polynomial in descending order
     *              ex. [2,1,3] for 2x^2+x+3
     * @return roots: roots of the polynomial in complex number. ex. a+bi
     */
    //Eigen::VectorXcf poly_solver(const Eigen::VectorXf& coef);

    /**
     * @brief calculate the se3 distance for giving R and T
     * @return d: se3 distance of given R and T
     */
    //float dist_se3(const Eigen::Matrix3f& R, const Eigen::Vector3f& T) const;

    /**
     * @brief update transformation matrix
     *
    void update_tf(const Mat33f & R, const Vec3f & T,
                   // outputs
                   CvoState * cvo_state,
                   Eigen::Ref<Mat44f > transform
                   ) const;
    */

    /**
     * @brief isotropic (same length-scale for all dimensions) squared-exponential kernel
     * @param l: kernel characteristic length-scale, aka cvo.ell
     * @param s2: signal variance, square of cvo.sigma
     * @return k: n-by-m kernel matrix 
     */
    /*
    void se_kernel(//SquareExpParams * se_params_gpu,
                   std::shared_ptr<CvoPointCloudGPU> points_fixed,
                   std::shared_ptr<CvoPointCloudGPU> points_moving,
                   float ell,
                   perl_registration::cuKdTree<CvoPoint>::SharedPtr kdtree,
                   // output
                   SparseKernelMat * A_mat
                   ) const;
    */

    /**
     * @brief computes the Lie algebra transformation elements
     *        twist = [omega; v] will be updated in this function
     */
    //void compute_flow(CvoState * cvo_state ) const;

    //void compute_step_size(CvoState * cvo_state) const;


    /**
     * @brief transform cloud_y for current update
     */
    //void transform_pointcloud_thrust(std::shared_ptr<CvoPointCloudGPU> init_cloud,
    //                                 std::shared_ptr<CvoPointCloudGPU> transformed_cloud,
    //                                 Mat33f * R_gpu, Vec3f * T_gpu
    //                                 ) const ;

  public:
    // public funcitons

    // constructor and destructor
    AdaptiveCvoGPU(const std::string & f);
    ~AdaptiveCvoGPU();
    CvoParams & get_params() {return params;}
    void write_params(CvoParams * param);
    
    /**
     * @brief align two rgbd pointcloud
     *        the function will iterate MAX_ITER times unless break conditions are met
     *        return 0 if sucess. return -1 if fails
     */
    int align(const CvoPointCloud& source_points,
              const CvoPointCloud& target_points,
              const Eigen::Matrix4f & init_guess_transform,
              Eigen::Ref<Eigen::Matrix4f> transform,
              double *registration_seconds=nullptr ) const;

    // callable after each align
    float inner_product(const CvoPointCloud& source_points,
                        const CvoPointCloud& target_points,
                        const Eigen::Matrix4f & source_frame_to_target_frame) const;


    //void run_cvo(const int dataset_seq,const cv::Mat& RGB_img,const cv::Mat& dep_img, MatrixXf_row semantic_label);
  };

  
}

