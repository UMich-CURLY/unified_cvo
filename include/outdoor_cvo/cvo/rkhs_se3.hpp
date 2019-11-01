/* ----------------------------------------------------------------------------
 * Copyright 2019, Tzu-yuan Lin <tzuyuan@umich.edu>, Maani Ghaffari <maanigj@umich.edu>,
                   Ray Zhang <rzh@umich.edu>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   rkhs_se3.hpp
 *  @author Tzu-yuan Lin, Maani Ghaffari 
 *  @brief  Header file for contineuous visual odometry rkhs_se3 registration
 *  @date   August 15, 2019
 **/

#ifndef RKHS_SE3_H
#define RKHS_SE3_H


// #include "data_type.h"
#include "LieGroup.h"
#include "pcd_generator.hpp"
#include "util/nanoflann.h"
#include "util/Pnt.h"
#include "util/settings.h"
#include "FullSystem/HessianBlocks.h"
#include "KDTreeVectorOfVectorsAdaptor.h"


#include <vector>
#include <string.h>
#include <iostream>
#include <memory>
#include <utility>
#include <future>
#include <thread>

// #include <pcl/filters/filter.h>
//#include <pcl/point_types.h>
//#include <pcl/point_cloud.h>
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
#include <opencv2/core/mat.hpp>
#include <boost/timer/timer.hpp>
// #include <omp.h>
#include <tbb/tbb.h>


//using namespace std;
using namespace nanoflann;

namespace cvo{
  class rkhs_se3{

  private:
    // private variables
    std::unique_ptr<frame> ptr_fixed_fr;
    std::unique_ptr<frame> ptr_moving_fr;

    std::unique_ptr<point_cloud> ptr_fixed_pcd;
    std::unique_ptr<point_cloud> ptr_moving_pcd;

    int num_fixed;              // target point cloud counts
    int num_moving;             // source point cloud counts
    cloud_t *cloud_x;    // target points represented as a matrix (num_fixed,3)
    cloud_t *cloud_y;    // source points represented as a matrix (num_moving,3)

    float ell;          // kernel characteristic length-scale
    float ell_init;
   float ell_min;
    float ell_max;
    double dl;           // changes for ell in each iteration
    double dl_step;
    float min_dl_step;
    float max_dl_step;
    float sigma;        // kernel signal variance (set as std)      
    float sp_thres;     // kernel sparsification threshold       
    float c;            // so(3) inner product scale     
    float d;            // R^3 inner product scale
    float color_scale;  // color space inner product scale
    float c_ell;        // kernel characteristic length-scale for color kernel
    float c_sigma;      // kernel signal variance for color kernel
    float s_ell;        // length-scale for semantic labels
    float s_sigma;      // signal variance for semantic labels
    int MAX_ITER;       // maximum number of iteration
    float eps;          // the program stops if norm(omega)+norm(v) < eps
    float eps_2;        // threshold for se3 distance
    float min_step;     // minimum step size for integration
    float step;         // integration step size

    Eigen::Matrix3f R;   // orientation
    Eigen::Vector3f T;   // translation
    Eigen::SparseMatrix<float,Eigen::RowMajor> A;      // coefficient matrix, represented in sparse
    Eigen::SparseMatrix<float,Eigen::RowMajor> Axx;      // coefficient matrix, represented in sparse
    Eigen::SparseMatrix<float,Eigen::RowMajor> Ayy;      // coefficient matrix, represented in sparse
    Eigen::Vector3f omega;  // so(3) part of twist
    Eigen::Vector3f v;      // R^3 part of twist

    // variables for cloud manipulations
    typedef Eigen::Triplet<float> Trip_t;
    tbb::concurrent_vector<Trip_t> A_trip_concur;

    //pcl::visualization::PCLVisualizer::Ptr viewer;
    //int frame_id = 0;
    //    int pcd_id = 0;

  public:
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    // public variables
    bool init;          // initialization indicator
    int iter;           // final iteration for display
    Eigen::Affine3f transform;  // transformation matrix
    Eigen::Affine3f prev_transform;
    Eigen::Affine3f accum_tf;       // accumulated transformation matrix for trajectory
    Eigen::Affine3f accum_tf_vis;

    bool debug_print;
    FILE * relative_transform_file;
    FILE * init_guess_file;
        
  private:
    // private functions
        
    /**
     * @brief a polynomial root finder
     * @param coef: coefficeints of the polynomial in descending order
     *              ex. [2,1,3] for 2x^2+x+3
     * @return roots: roots of the polynomial in complex number. ex. a+bi
     */
    inline Eigen::VectorXcf poly_solver(const Eigen::VectorXf& coef);

    /**
     * @brief calculate the se3 distance for giving R and T
     * @return d: se3 distance of given R and T
     */
    inline float dist_se3(const Eigen::Matrix3f& R, const Eigen::Vector3f& T);

    /**
     * @brief update transformation matrix
     */
    inline void update_tf();


    /**
     * @brief compute color inner product of ith row in fixed and jth row in moving
     * @param i: index of desired row in fixed
     * @param j: indxe of desired row in moving
     * @return CI: the inner product
     */
    inline float color_inner_product(const int i, const int j);
        
    /**
     * @brief compute color kernel
     */
    inline float color_kernel(const int i, const int j);

    /**
     * @brief isotropic (same length-scale for all dimensions) squared-exponential kernel
     * @param l: kernel characteristic length-scale, aka rkhs_se3.ell
     * @prarm s2: signal variance, square of rkhs_se3.sigma
     * @return k: n-by-m kernel matrix 
     */
    //void se_kernel(const float l, const float s2);
    void se_kernel(point_cloud* cloud_a, point_cloud* cloud_b, \
                   cloud_t* cloud_a_pos, cloud_t* cloud_b_pos,          \
                   Eigen::SparseMatrix<float,Eigen::RowMajor>& A_temp,
                   tbb::concurrent_vector<Trip_t> & A_trip_concur_) const;

    /**
     * @brief computes the Lie algebra transformation elements
     *        twist = [omega; v] will be updated in this function
     */
    void compute_flow();
        

    /**
     * @brief compute the integration step size
     *        step will be updated in  this function
     */
    void compute_step_size();

    /**
     * @brief transform cloud_y for current update
     */
    void transform_pcd();

  public:
    // public funcitons

    // constructor and destructor
    rkhs_se3();
    ~rkhs_se3();

    /**
     * @brief initialize new point cloud and extract pcd as matrices
     */
    // void set_pcd(const int dataset_seq,const cv::Mat& RGB_img,const cv::Mat& dep_img, MatrixXf_row semantic_label);


    /*  
        @brief: set pcd from vector of xyz and rgb image directly

    */
    template <class PointType>
    void set_pcd(int w, int h,
                 const dso::FrameHessian * img_source,
                 const std::vector<PointType> & source_points,
                 const dso::FrameHessian * img_target,
                 const std::vector<PointType> & target_points,
                 const Eigen::Affine3f & init_guess,
                 bool is_using_init_guess);

    
  template <class PointType>
  void set_pcd(const vector<PointType> & source_points,
               const vector<PointType> & target_points,
               Eigen::Affine3f & init_guess_transform,
               bool is_using_init_guess);


    
    /**
     * @brief align two rgbd pointcloud
     *        the function will iterate MAX_ITER times unless break conditions are met
     */
    void align();
    // callable after each align
    float inner_product() const ;

    // just compute the inner product
    template <typename PointType>
    float inner_product(const std::vector<PointType> & source_points,
                        const std::vector<PointType> & target_points,
                        const Eigen::Affine3f & source_frame_to_target_frame);
    template <typename PointType>
    void  loop_fill_pcd (const std::vector<PointType> & dso_pts,
                         point_cloud & output_cvo_pcd )  ;
    //void visualize_pcd();


    Eigen::Affine3f get_transform() {return transform;}
    Eigen::Affine3f get_prev_transform() {return prev_transform;}
    Eigen::Affine3f get_accum_transform() {return accum_tf;}


    
    //void run_cvo(const int dataset_seq,const cv::Mat& RGB_img,const cv::Mat& dep_img, MatrixXf_row semantic_label);
  };

  
}
#endif  // RKHS_SE3_H
