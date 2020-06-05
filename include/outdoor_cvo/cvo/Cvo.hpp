/* ----------------------------------------------------------------------------
 * Copyright 2019, Tzu-yuan Lin <tzuyuan@umich.edu>, Maani Ghaffari <maanigj@umich.edu>,
                   Ray Zhang <rzh@umich.edu>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   cvo.hpp
 *  @author Tzu-yuan Lin, Maani Ghaffari 
 *  @brief  Header file for contineuous visual odometry registration
 *  @date   November 03, 2019
 **/

#ifndef CVO_H
#define CVO_H


// #include "data_type.h"
#include "utils/CvoPointCloud.hpp"
// #include "pcd_generator.hpp"


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
#include <pcl/common/transforms.h>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/Cholesky> 
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/StdVector>

// #include <omp.h>

//#include <tbb/tbb.h>
//#define IS_USING_SEMANTICS


//using namespace std;


namespace cvo{
  class cvo{

  private:
    // private variables
    const CvoPointCloud* ptr_fixed_pcd;
    const CvoPointCloud* ptr_moving_pcd;

    int num_fixed;              // target point cloud counts
    int num_moving;             // source point cloud counts
    ArrayVec3f *cloud_x;    // target points represented as a matrix (num_fixed,3)
    ArrayVec3f *cloud_y;    // source points represented as a matrix (num_moving,3)

    
    float ell_init;
    float ell_min;

    float ell_max_fixed;
    float ell_reduced_1;
    float ell_reduced_2;
    float ell_reduced_3;
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
    float n_ell;
    float n_sigma;
    float s_ell;        // length-scale for semantic labels
    float s_sigma;      // signal variance for semantic labels
    int MAX_ITER;       // maximum number of iteration
    float eps;          // the program stops if norm(omega)+norm(v) < eps
    float eps_2;        // threshold for se3 distance
    float min_step;     // minimum step size for integration
    float step;         // integration step size
    float prev_dist;  
    float ell_init_ratio;
    
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

    // variables for indicator
    std::queue<float> indicator_start_queue;
    std::queue<float> indicator_end_queue;
    float indicator_start_sum;
    float indicator_end_sum;
    float last_indicator;
    bool decrease;
    bool last_decrease;
    bool increase;
    bool skip_iteration;
    float indicator;

    std::ofstream ell_file;
    std::ofstream dist_change_file;
    std::ofstream transform_file;
    std::ofstream inner_product_file;
    std::ofstream effective_points_file;


  public:
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    float ell;          // kernel characteristic length-scale
    float ell_max;
    // public variables
    bool init;          // initialization indicator
    int iter;           // final iteration for display
    Eigen::Affine3f transform;  // transformation matrix
    Eigen::Affine3f prev_transform;
    Eigen::Affine3f prev_iter_transform;
    Eigen::Affine3f accum_tf;       // accumulated transformation matrix for trajectory
    Eigen::Affine3f accum_tf_vis;

    bool debug_print;
    FILE * relative_transform_file;
    FILE * init_guess_file;

    float get_stop_criteria() {return eps_2;}
    void set_stop_criteria(float eps_2_new) {eps_2 = eps_2_new;}
    

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
    inline float dist_se3(const Eigen::Matrix4f& tf);
    /**
     * @brief update transformation matrix
     */
    inline void update_tf();



    /**
     * @brief isotropic (same length-scale for all dimensions) squared-exponential kernel
     * @param l: kernel characteristic length-scale, aka cvo.ell
     * @param s2: signal variance, square of cvo.sigma
     * @return k: n-by-m kernel matrix 
     */
    //void se_kernel(const float l, const float s2);
    void se_kernel(const CvoPointCloud* cloud_a, const CvoPointCloud* cloud_b, \
                   cloud_t* cloud_a_pos, cloud_t* cloud_b_pos,          \
                   Eigen::SparseMatrix<float,Eigen::RowMajor>& A_temp,
                   tbb::concurrent_vector<Trip_t> & A_trip_concur_) const;
    void se_kernel_init_ell(const CvoPointCloud* cloud_a, const CvoPointCloud* cloud_b, \
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

    /**
     * 
     **/
    // void convert_to_pcl_cloud(const CvoPointCloud& cvo_cloud, pcl::PointCloud<PointSegmentedDistribution> pcl_cloud);

    /**
     * @brief compute indicator and change lenthscale if needed
     */
    void compute_indicator();

  public:
    // public funcitons

    // constructor and destructor
    cvo();
    cvo(const std::string & name);
    ~cvo();

    /**
     * @brief initialize new point cloud and extract pcd as matrices
     */
    // void set_pcd(const int dataset_seq,const cv::Mat& RGB_img,const cv::Mat& dep_img, MatrixXf_row semantic_label);


    /*  
        @brief: set pcd from vector of xyz and rgb image directly
    */
    void set_pcd(const CvoPointCloud& source_points,
                const CvoPointCloud& target_points,
                const Eigen::Affine3f & init_guess_transform,
                bool is_using_init_guess);
    
    /**
     * @brief align two rgbd pointcloud
     *        the function will iterate MAX_ITER times unless break conditions are met
     *        return 0 if sucess. return -1 if fails
     */
    int align();
    int align_one_iter(int cur_iter);
    void pcd_destructor();

    // callable after each align
    float inner_product() const ;
    float inner_product_normalized() const ;
    int number_of_non_zeros_in_A() const;
    // just compute the inner product
    float inner_product(const CvoPointCloud& source_points,
                        const CvoPointCloud& target_points,
                        const Eigen::Affine3f & source_frame_to_target_frame) const;



    Eigen::Affine3f get_transform() {return transform;}
    Eigen::Affine3f get_prev_transform() {return prev_transform;}
    Eigen::Affine3f get_prev_iter_transform() {return prev_iter_transform;}
    Eigen::Affine3f get_accum_transform() {return accum_tf;}
    const CvoPointCloud* get_fixed_pcd() {return ptr_fixed_pcd;};
    const CvoPointCloud* get_moving_pcd() {return ptr_moving_pcd;};
    Eigen::SparseMatrix<float,Eigen::RowMajor> get_A_matrix() {return A;};
    
    //void run_cvo(const int dataset_seq,const cv::Mat& RGB_img,const cv::Mat& dep_img, MatrixXf_row semantic_label);
  };

  
}
#endif  // cvo_H
