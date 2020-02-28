/* ----------------------------------------------------------------------------
 * Copyright 2019, Tzu-yuan Lin <tzuyuan@umich.edu>, Maani Ghaffari <maanigj@umich.edu>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   acvo.cpp
 *  @author Tzu-yuan Lin, Maani Ghaffari 
 *  @brief  Source file for adaptive contineuous visual odometry registration
 *  @date   Feburary 3, 2020
 **/

#include "cvo/ACvo.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <ctime>
#include <functional>
#include <cassert>
using namespace std;

namespace cvo{

  static bool is_logging = false;

  acvo::acvo(const std::string & param_file):
    // initialize parameters
    init(false),           // initialization indicator
    ell_init(0.15*7),             // kernel characteristic length-scale
    ell(0.1*7),
    ell_min(0.0391*7),
    ell_max(0.15*7),
    ell_max_fixed(0.15*7),
    dl(0),
    dl_step(0.3),
    min_dl_step(0.05),
    max_dl_step(1),

    //ell(0.15*7),             // kernel characteristic length-scale
    sigma(0.1),            // kernel signal variance (set as std)      
    sp_thres(1e-3),        // kernel sparsification threshold      8.315e-3    
    c(7.0),                // so(3) inner product scale     
    d(7.0),                // R^3 inner product scale
    //    color_scale(1.0e-5),   // color space inner product scale
    //c_ell(200),             // kernel characteristic length-scale for color kernel
    c_ell(0.5),
    c_sigma(1),
    s_ell(0.1),
    s_sigma(1),
    MAX_ITER(2000),        // maximum number of iteration
    min_step(2*1.0e-1),    // minimum integration step
    eps(5*1.0e-5),         // threshold for stopping the function
    eps_2(1.0e-5),         // threshold for se3 distance
    R(Eigen::Matrix3f::Identity(3,3)), // initialize rotation matrix to I
    T(Eigen::Vector3f::Zero()),        // initialize translation matrix to zeros
    transform(Eigen::Affine3f::Identity()),    // initialize transformation to I
    prev_transform(Eigen::Affine3f::Identity()),
    accum_tf(Eigen::Affine3f::Identity()),
    accum_tf_vis(Eigen::Affine3f::Identity()),
    debug_print(true)
  {
    FILE* ptr = fopen(param_file.c_str(), "r" ); 
    if (ptr!=NULL) 
    {
      std::cout<<"reading cvo params from file\n";
      fscanf(ptr, "%f\n%f\n%f\n%f\n%f\n%lf\n%lf\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%d\n%f\n%f\n%f\n",
             &ell_init
             ,& ell
             ,& ell_min
	     ,& ell_max
             ,& ell_max_fixed
             ,& dl
             ,& dl_step
             ,& min_dl_step
             ,& max_dl_step
             ,& sigma
             ,& sp_thres
             , &c
             ,& d
             ,& c_ell
             ,& c_sigma
             , &s_ell
             ,& s_sigma
             ,& MAX_ITER
             ,& min_step
             ,& eps
             ,& eps_2);
      fclose(ptr);
    }
    if (is_logging) {
      relative_transform_file = fopen("cvo_relative_transforms.txt", "w");
      init_guess_file = fopen("cvo_init_guess.txt", "w");
      assert (relative_transform_file && init_guess_file);
    }
  }

  acvo::acvo():
    // initialize parameters
    init(false),           // initialization indicator
    ell_init(0.15*7),             // kernel characteristic length-scale
    ell(0.1*7),
    ell_min(0.0391*7),
    ell_max(0.15*7),
    ell_max_fixed(0.55),
    dl(0),
    dl_step(0.3),
    min_dl_step(0.05),
    max_dl_step(1),

    //ell(0.15*7),             // kernel characteristic length-scale
    sigma(0.1),            // kernel signal variance (set as std)      
    sp_thres(1e-3),        // kernel sparsification threshold      8.315e-3    
    c(7.0),                // so(3) inner product scale     
    d(7.0),                // R^3 inner product scale
    //    color_scale(1.0e-5),   // color space inner product scale
    //c_ell(200),             // kernel characteristic length-scale for color kernel
    c_ell(0.5),
    c_sigma(1),
    s_ell(0.1),
    s_sigma(1),
    MAX_ITER(2000),        // maximum number of iteration
    min_step(2*1.0e-1),    // minimum integration step
    eps(5*1.0e-5),         // threshold for stopping the function
    eps_2(1.0e-5),         // threshold for se3 distance
    R(Eigen::Matrix3f::Identity(3,3)), // initialize rotation matrix to I
    T(Eigen::Vector3f::Zero()),        // initialize translation matrix to zeros
    transform(Eigen::Affine3f::Identity()),    // initialize transformation to I
    prev_transform(Eigen::Affine3f::Identity()),
    accum_tf(Eigen::Affine3f::Identity()),
    accum_tf_vis(Eigen::Affine3f::Identity()),
    debug_print(false)
  {
    FILE* ptr = fopen("cvo_params.txt", "r" ); 
    if (ptr!=NULL) 
    {
      std::cout<<"reading cvo params from file\n";
      fscanf(ptr, "%f\n%f\n%f\n%f\n%f\n%lf\n%lf\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%d\n%f\n%f\n%f\n",
             &ell_init
             ,& ell
             ,& ell_min
             ,& ell_max
             ,& ell_max_fixed
             ,& dl
             ,& dl_step
             ,& min_dl_step
             ,& max_dl_step
             ,& sigma
             ,& sp_thres
             , &c
             ,& d
             ,& c_ell
             ,& c_sigma
             , &s_ell
             ,& s_sigma
             ,& MAX_ITER
             ,& min_step
             ,& eps
             ,& eps_2);
      fclose(ptr);
    }
    if (is_logging) {
      relative_transform_file = fopen("cvo_relative_transforms.txt", "w");
      init_guess_file = fopen("cvo_init_guess.txt", "w");
      assert (relative_transform_file && init_guess_file);
    }
    std::cout<<"Using ACVO"<<std::endl;
  }

  acvo::~acvo() {
    if (is_logging){
      if (relative_transform_file)
        fclose(relative_transform_file);
      if (init_guess_file)
        fclose(init_guess_file);
    }
  }

  inline Eigen::VectorXcf acvo::poly_solver(const Eigen::VectorXf& coef){
    // extract order
    int order = coef.size()-1;
    Eigen::VectorXcf roots;
    
    // create M = diag(ones(n-1,1),-1)
    Eigen::MatrixXf M = Eigen::MatrixXf::Zero(order,order);
    M.bottomLeftCorner(order-1,order-1) = Eigen::MatrixXf::Identity(order-1,order-1);
    
    // M(1,:) = -p(2:n+1)./p(1)
    M.row(0) = -(coef/coef(0)).segment(1,order).transpose();

    // eigen(M) and get the answer
    roots = M.eigenvalues();

    return roots;
  }

  inline float acvo::dist_se3(const Eigen::Matrix3f& R, const Eigen::Vector3f& T){
    // create transformation matrix
    Eigen::Matrix4f temp_transform = Eigen::Matrix4f::Identity();
    temp_transform.block<3,3>(0,0)=R;
    temp_transform.block<3,1>(0,3)=T;
    
    // distance = frobenius_norm(logm(trans))
    float d = temp_transform.log().norm();
    
    return d;
  }

  inline void acvo::update_tf(){
    // transform = [R', -R'*T; 0,0,0,1]
    transform.matrix().block<3,3>(0,0) = R.transpose();
    transform.matrix().block<3,1>(0,3) = -R.transpose()*T;

    // if (debug_print) std::cout<<"transform mat "<<transform.matrix()<<"\n";
  }


  typedef KDTreeVectorOfVectorsAdaptor<cloud_t, float>  kd_tree_t;

  /*
  __global__
  void cvo::se_kernelgpu(point_cloud* cloud_a, point_cloud* cloud_b, 
                               cloud_t* cloud_a_pos, cloud_t* cloud_b_pos,
                               kd_tree_t * mat_index, 
                               Eigen::SparseMatrix<float,Eigen::RowMajor>& A_temp){
    A_trip_concur.clear();
    const float s2= sigma*sigma;
    const float l = ell;
    // convert k threshold to d2 threshold (so that we only need to calculate k when needed)
    const float d2_thres = -2.0*l*l*log(sp_thres/s2);
    const float d2_c_thres = -2.0*c_ell*c_ell*log(sp_thres/c_sigma/c_sigma);
    // loop through points
    tbb::parallel_for(int(0),cloud_a->num_points,[&](int i){
        // for(int i=0; i<num_fixed; ++i){
        const float search_radius = d2_thres;
        std::vector<std::pair<size_t,float>>  ret_matches;
        nanoflann::SearchParams params;
        //params.sorted = false;
        const size_t nMatches = mat_index->index->radiusSearch(&(*cloud_a_pos)[i](0), search_radius, ret_matches, params);
        Eigen::Matrix<float,Eigen::Dynamic,1> feature_a = cloud_a->features.row(i).transpose();
        Eigen::VectorXf label_a = cloud_a->labels.row(i);
        // for(int j=0; j<num_moving; j++){
        for(size_t j=0; j<nMatches; ++j){
          int idx = ret_matches[j].first;
          float d2 = ret_matches[j].second;
          // d2 = (x-y)^2
          float k = 0;
          float ck = 0;
          float sk = 0;
          float d2_color = 0;
          float d2_semantic = 0;
          float a = 0;
          if(d2<d2_thres){
            Eigen::Matrix<float,Eigen::Dynamic,1> feature_b = cloud_b->features.row(idx).transpose();
            Eigen::VectorXf label_b = cloud_b->labels.row(idx);
            d2_color = ((feature_a-feature_b).squaredNorm());
            d2_semantic = ((label_a-label_b).squaredNorm());
            if(d2_color<d2_c_thres){
              k = s2*exp(-d2/(2.0*l*l));
              ck = c_sigma*c_sigma*exp(-d2_color/(2.0*c_ell*c_ell));
              sk = s_sigma*s_sigma*exp(-d2_semantic/(2.0*s_ell*s_ell));
              a = ck*k*sk;
              if (a > sp_thres) A_trip_concur.push_back(Trip_t(i,idx,a));
            }
          }
        }
      });
    // }
    // form A
    A_temp.setFromTriplets(A_trip_concur.begin(), A_trip_concur.end());
    A_temp.makeCompressed();
  }
  */
  

  void acvo::se_kernel(const CvoPointCloud* cloud_a, const CvoPointCloud* cloud_b, \
                           cloud_t* cloud_a_pos, cloud_t* cloud_b_pos,\
                           Eigen::SparseMatrix<float,Eigen::RowMajor>& A_temp,
                           tbb::concurrent_vector<Trip_t> & A_trip_concur_)const {
    A_trip_concur_.clear();
    const float s2= sigma*sigma;
    const float l = ell;

    // convert k threshold to d2 threshold (so that we only need to calculate k when needed)
    const float d2_thres = -2.0*l*l*log(sp_thres/s2);
    // if (debug_print ) std::cout<<"l is "<<l<<",d2_thres is "<<d2_thres<<std::endl;
    const float d2_c_thres = -2.0*c_ell*c_ell*log(sp_thres/c_sigma/c_sigma);
    // if (debug_print) std::cout<<"d2_c_thres is "<<d2_c_thres<<std::endl;
    
    /** 
     * kdtreeeeeeeeeeeeeeeeeeeee
     **/
    typedef KDTreeVectorOfVectorsAdaptor<cloud_t, float>  kd_tree_t;

    kd_tree_t mat_index(3 /*dim*/, (*cloud_b_pos), 10 /* max leaf */ );
    mat_index.index->buildIndex();

    // loop through points
    tbb::parallel_for(int(0),cloud_a->num_points(),[&](int i){
    //for(int i=0; i<num_fixed; ++i){

        const float search_radius = d2_thres;
        std::vector<std::pair<size_t,float>>  ret_matches;

        nanoflann::SearchParams params;
        //params.sorted = false;

        const size_t nMatches = mat_index.index->radiusSearch(&(*cloud_a_pos)[i](0), search_radius, ret_matches, params);

        Eigen::Matrix<float,Eigen::Dynamic,1> feature_a = cloud_a->features().row(i).transpose();

#ifdef IS_USING_SEMANTICS        
        Eigen::VectorXf label_a = cloud_a->labels().row(i);
#endif
        // for(int j=0; j<num_moving; j++){
        for(size_t j=0; j<nMatches; ++j){
          int idx = ret_matches[j].first;
          float d2 = ret_matches[j].second;
          // d2 = (x-y)^2
          float k = 0;
          float ck = 0;
          float sk = 1;
          float d2_color = 0;
          float d2_semantic = 0;
          float a = 0;
          if(d2<d2_thres){
            Eigen::Matrix<float,Eigen::Dynamic,1> feature_b = cloud_b->features().row(idx).transpose();
            d2_color = ((feature_a-feature_b).squaredNorm());
#ifdef IS_USING_SEMANTICS            
            Eigen::VectorXf label_b = cloud_b->labels().row(idx);
            d2_semantic = ((label_a-label_b).squaredNorm());
#endif
            
            if(d2_color<d2_c_thres){
              k = s2*exp(-d2/(2.0*l*l));
              ck = c_sigma*c_sigma*exp(-d2_color/(2.0*c_ell*c_ell));
#ifdef IS_USING_SEMANTICS              
              sk = s_sigma*s_sigma*exp(-d2_semantic/(2.0*s_ell*s_ell));
#else
              sk = 1;
#endif              
              a = ck*k*sk;

              if (a > sp_thres){
                A_trip_concur_.push_back(Trip_t(i,idx,a));
              }
             
            
            }
          }
        }
      });

        //}
    // form A
    A_temp.setFromTriplets(A_trip_concur_.begin(), A_trip_concur_.end());
    A_temp.makeCompressed();
  }

  

  void acvo::se_kernel_init_ell(const CvoPointCloud* cloud_a, const CvoPointCloud* cloud_b, \
                               cloud_t* cloud_a_pos, cloud_t* cloud_b_pos, \
                               Eigen::SparseMatrix<float,Eigen::RowMajor>& A_temp,
                               tbb::concurrent_vector<Trip_t> & A_trip_concur_)const {
    A_trip_concur_.clear();
    const float s2= sigma*sigma;
    const float l = ell_init;

    // convert k threshold to d2 threshold (so that we only need to calculate k when needed)
    const float d2_thres = -2.0*l*l*log(sp_thres/s2);
    if (debug_print ) std::cout<<"l is "<<l<<",d2_thres is "<<d2_thres<<std::endl;
    const float d2_c_thres = -2.0*c_ell*c_ell*log(sp_thres/c_sigma/c_sigma);
    if (debug_print) std::cout<<"d2_c_thres is "<<d2_c_thres<<std::endl;
    
    /** 
     * kdtreeeeeeeeeeeeeeeeeeeee
     **/
    typedef KDTreeVectorOfVectorsAdaptor<cloud_t, float>  kd_tree_t;

    kd_tree_t mat_index(3 /*dim*/, (*cloud_b_pos), 10 /* max leaf */ );
    mat_index.index->buildIndex();

    // loop through points
    tbb::parallel_for(int(0),cloud_a->num_points(),[&](int i){
    //for(int i=0; i<num_fixed; ++i){

        const float search_radius = d2_thres;
        std::vector<std::pair<size_t,float>>  ret_matches;

        nanoflann::SearchParams params;
        //params.sorted = false;

        const size_t nMatches = mat_index.index->radiusSearch(&(*cloud_a_pos)[i](0), search_radius, ret_matches, params);

        Eigen::Matrix<float,Eigen::Dynamic,1> feature_a = cloud_a->features().row(i).transpose();

#ifdef IS_USING_SEMANTICS        
        Eigen::VectorXf label_a = cloud_a->labels().row(i);
#endif
        // for(int j=0; j<num_moving; j++){
        for(size_t j=0; j<nMatches; ++j){
          int idx = ret_matches[j].first;
          float d2 = ret_matches[j].second;
          // d2 = (x-y)^2
          float k = 0;
          float ck = 0;
          float sk = 1;
          float d2_color = 0;
          float d2_semantic = 0;
          float a = 0;
          if(d2<d2_thres){
            Eigen::Matrix<float,Eigen::Dynamic,1> feature_b = cloud_b->features().row(idx).transpose();
            d2_color = ((feature_a-feature_b).squaredNorm());
#ifdef IS_USING_SEMANTICS            
            Eigen::VectorXf label_b = cloud_b->labels().row(idx);
            d2_semantic = ((label_a-label_b).squaredNorm());
#endif
            
            if(d2_color<d2_c_thres){
              k = s2*exp(-d2/(2.0*l*l));
              ck = c_sigma*c_sigma*exp(-d2_color/(2.0*c_ell*c_ell));
#ifdef IS_USING_SEMANTICS              
              sk = s_sigma*s_sigma*exp(-d2_semantic/(2.0*s_ell*s_ell));
#else
              sk = 1;
#endif              
              a = ck*k*sk;

              if (a > sp_thres){
                A_trip_concur_.push_back(Trip_t(i,idx,a));
              }
             
            
            }
          }
        }
      });

        //}
    // form A
    A_temp.setFromTriplets(A_trip_concur_.begin(), A_trip_concur_.end());
    A_temp.makeCompressed();
  }

  
  
  void acvo::compute_flow(){

    auto start = chrono::system_clock::now();
    auto end = chrono::system_clock::now();
 
    // compute SE kernel for Axy
    se_kernel(ptr_fixed_pcd,ptr_moving_pcd,cloud_x,cloud_y,A, A_trip_concur);

    if (debug_print ) {std::cout<<"nonzeros in A "<<A.nonZeros()<<std::endl;
    }
    // compute SE kernel for Axx and Ayy
    se_kernel(ptr_fixed_pcd,ptr_fixed_pcd,cloud_x,cloud_x,Axx, A_trip_concur);
    //start = chrono::system_clock::now();
    se_kernel(ptr_moving_pcd,ptr_moving_pcd,cloud_y,cloud_y,Ayy, A_trip_concur);
    //end = chrono::system_clock::now();
    //std::cout<<"time for this kernel is "<<(end- start).count()<<std::endl;
    // some initialization of the variables
    omega = Eigen::Vector3f::Zero();
    v = Eigen::Vector3f::Zero();
    Eigen::Vector3d double_omega = Eigen::Vector3d::Zero(); // this is omega in double precision
    Eigen::Vector3d double_v = Eigen::Vector3d::Zero();
    dl = 0;
    tbb::spin_mutex omegav_lock;
    int sum = 0;

    float ell_3 = ell*ell*ell;

    // loop through points in cloud_x
    //start = chrono::system_clock::now();
    tbb::parallel_for(int(0),num_fixed,[&](int i){
                                         // for(int i=0; i<num_fixed; ++i){
                                         // initialize reused varaibles
                                         int num_non_zeros = A.innerVector(i).nonZeros();
                                         int num_non_zeros_xx = Axx.innerVector(i).nonZeros();
                                         int num_non_zeros_yy; 
                                         num_non_zeros_yy = (i<num_moving)?Ayy.innerVector(i).nonZeros():0;
                                         Eigen::MatrixXf Ai = Eigen::MatrixXf::Zero(1,num_non_zeros);
                                         Eigen::MatrixXf Axxi = Eigen::MatrixXf::Zero(1,num_non_zeros_xx);
                                         Eigen::MatrixXf Ayyi = Eigen::MatrixXf::Zero(1,num_non_zeros_yy);
                                         Eigen::MatrixXf cross_xy = Eigen::MatrixXf::Zero(num_non_zeros,3);
                                         Eigen::MatrixXf diff_yx = Eigen::MatrixXf::Zero(num_non_zeros,3);
                                         Eigen::MatrixXf diff_xx = Eigen::MatrixXf::Zero(num_non_zeros_xx,3);
                                         Eigen::MatrixXf diff_yy = Eigen::MatrixXf::Zero(num_non_zeros_yy,3);
                                         Eigen::MatrixXf sum_diff_yx_2 = Eigen::MatrixXf::Zero(num_non_zeros,1);
                                         Eigen::MatrixXf sum_diff_xx_2 = Eigen::MatrixXf::Zero(num_non_zeros_xx,1);
                                         Eigen::MatrixXf sum_diff_yy_2 = Eigen::MatrixXf::Zero(num_non_zeros_yy,1);
                                         Eigen::Matrix<double, 1, 3> partial_omega;
                                         Eigen::Matrix<double, 1, 3> partial_v;
                                         double partial_dl = 0;

                                         int j = 0;
                                         // loop through non-zero ids in ith row of A
                                         for(Eigen::SparseMatrix<float,Eigen::RowMajor>::InnerIterator it(A,i); it; ++it){
                                           int idx = it.col();
                                           Ai(0,j) = it.value();    // extract current value in A
                                           cross_xy.row(j) = ((*cloud_x)[i].transpose().cross((*cloud_y)[idx].transpose()));
                                           diff_yx.row(j) = ((*cloud_y)[idx]-(*cloud_x)[i]).transpose();
                                           sum_diff_yx_2(j,0) = diff_yx.row(j).squaredNorm();
                                           ++j;
                                         }
                                         j = 0; 
                                         for(Eigen::SparseMatrix<float,Eigen::RowMajor>::InnerIterator it(Axx,i); it; ++it){
                                           int idx = it.col();
                                           Axxi(0,j) = it.value();    // extract current value in A
                                           diff_xx.row(j) = ((*cloud_x)[idx]-(*cloud_x)[i]).transpose();
                                           sum_diff_xx_2(j,0) = diff_xx.row(j).squaredNorm();
                                           ++j;
                                         }
                                         if(i<num_moving){
                                           j = 0; 
                                           for(Eigen::SparseMatrix<float,Eigen::RowMajor>::InnerIterator it(Ayy,i); it; ++it){
                                             int idx = it.col();
                                             Ayyi(0,j) = it.value();    // extract current value in A
                                             diff_yy.row(j) = ((*cloud_y)[idx]-(*cloud_y)[i]).transpose();
                                             sum_diff_yy_2(j,0) = diff_yy.row(j).squaredNorm();
                                             ++j;
                                           }
                                           // update dl from Ayy
                                           partial_dl += double((1/ell_3*Ayyi*sum_diff_yy_2)(0,0));
                                         }
                                         partial_omega = (1/c*Ai*cross_xy).cast<double>();
                                         partial_v = (1/d*Ai*diff_yx).cast<double>();

                                         // update dl from Axy
                                         partial_dl -= double(2*(1/ell_3*Ai*sum_diff_yx_2)(0,0));
        
                                         // update dl from Axx
                                         partial_dl += double((1/ell_3*Axxi*sum_diff_xx_2)(0,0));

                                         // sum them up
                                         omegav_lock.lock();
                                         double_omega += partial_omega.transpose();
                                         double_v += partial_v.transpose();
                                         dl += partial_dl;
                                         omegav_lock.unlock();
                                       });
    //end = chrono::system_clock::now();
    //std::cout<<"time for this tbb gradient flow is "<<(end- start).count()<<std::endl;
    
    // }

    // if num_moving > num_fixed, update the rest of Ayy to dl 
    if(num_moving>num_fixed){
      tbb::parallel_for(int(num_fixed),num_moving,[&](int i){
                                                    int num_non_zeros_yy = Ayy.innerVector(i).nonZeros();
                                                    Eigen::MatrixXf Ayyi = Eigen::MatrixXf::Zero(1,num_non_zeros_yy);
                                                    Eigen::MatrixXf diff_yy = Eigen::MatrixXf::Zero(num_non_zeros_yy,3);
                                                    Eigen::MatrixXf sum_diff_yy_2 = Eigen::MatrixXf::Zero(num_non_zeros_yy,1);
                                                    double partial_dl = 0;

                                                    int j = 0; 
                                                    for(Eigen::SparseMatrix<float,Eigen::RowMajor>::InnerIterator it(Ayy,i); it; ++it){
                                                      int idx = it.col();
                                                      Ayyi(0,j) = it.value();    // extract current value in A
                                                      diff_yy.row(j) = ((*cloud_y)[idx]-(*cloud_y)[i]).transpose();
                                                      sum_diff_yy_2(j,0) = diff_yy.row(j).squaredNorm();
                                                      ++j;
                                                    }
                                                    partial_dl += double((1/ell_3*Ayyi*sum_diff_yy_2)(0,0));

                                                    omegav_lock.lock();
                                                    dl += partial_dl;
                                                    omegav_lock.unlock();
                                                  });
    }
    

    // update them to class-wide variables
    omega = double_omega.cast<float>();
    v = double_v.cast<float>();
    dl = dl/(Axx.nonZeros()+Ayy.nonZeros()-2*A.nonZeros());

  }


  void acvo::compute_step_size(){
    // compute skew matrix
    Eigen::Matrix3f omega_hat = skew(omega);
    
    // compute xi*z+v, xi^2*z+xi*v, xi^3*z+xi^2*v, xi^4*z+xi^3*v
    Eigen::MatrixXf xiz(num_moving,3);
    Eigen::MatrixXf xi2z(num_moving,3);
    Eigen::MatrixXf xi3z(num_moving,3);
    Eigen::MatrixXf xi4z(num_moving,3);
    Eigen::MatrixXf normxiz2(num_moving,1);
    Eigen::MatrixXf xiz_dot_xi2z(num_moving,1);
    Eigen::MatrixXf epsil_const(num_moving,1);

    tbb::parallel_for( int(0), num_moving, [&]( int j ){
                                             Eigen::Vector3f cloud_yi = (*cloud_y)[j];
                                             // xiz is w^ * z
                                             xiz.row(j) = omega.transpose().cross(cloud_yi.transpose())+v.transpose(); // (xi*z+v)
                                             xi2z.row(j) = (omega_hat*omega_hat*cloud_yi\
                                                            +(omega_hat*v)).transpose();    // (xi^2*z+xi*v)
                                             xi3z.row(j) = (omega_hat*omega_hat*omega_hat*cloud_yi\
                                                            +(omega_hat*omega_hat*v)).transpose();  // (xi^3*z+xi^2*v)
                                             xi4z.row(j) = (omega_hat*omega_hat*omega_hat*omega_hat*cloud_yi\
                                                            +(omega_hat*omega_hat*omega_hat*v)).transpose();    // (xi^4*z+xi^3*v)
                                             normxiz2(j,0) = xiz.row(j).squaredNorm();
                                             xiz_dot_xi2z(j,0) = (-xiz.row(j).dot(xi2z.row(j)));
                                             epsil_const(j,0) = xi2z.row(j).squaredNorm()+2*xiz.row(j).dot(xi3z.row(j));
                                           });

    // initialize coefficients
    float temp_coef = 1/(2.0*ell*ell);   // 1/(2*l^2)
    double B = 0;
    double C = 0;
    double D = 0;
    double E = 0;

    tbb::spin_mutex BCDE_lock;
    
    tbb::parallel_for(int(0),num_fixed,[&](int i){
                                         // loop through used index in ith row
                                         double Bi=0;
                                         double Ci=0;
                                         double Di=0;
                                         double Ei=0;

                                         for(Eigen::SparseMatrix<float,Eigen::RowMajor>::InnerIterator it(A,i); it; ++it){
                                           int idx = it.col();
            
                                           // diff_xy = x[i] - y[used_idx[j]]
                                           auto diff_xy = ((*cloud_x)[i] - (*cloud_y)[idx]);    
                                           // beta_i = -1/l^2 * dot(xiz,diff_xy)
                                           float beta_ij = (-2.0*temp_coef * xiz.row(idx)*diff_xy).value();
                                           // gamma_i = -1/(2*l^2) * (norm(xiz).^2 + 2*dot(xi2z,diff_xy))
                                           float gamma_ij = (-temp_coef * (normxiz2.row(idx)\
                                                                           + 2.0*xi2z.row(idx)*diff_xy)).value();
                                           // delta_i = 1/l^2 * (dot(-xiz,xi2z) + dot(-xi3z,diff_xy))
                                           float delta_ij = (2.0*temp_coef * (xiz_dot_xi2z.row(idx)\
                                                                              + (-xi3z.row(idx)*diff_xy))).value();
                                           // epsil_i = -1/(2*l^2) * (norm(xi2z).^2 + 2*dot(xiz,xi3z) + 2*dot(xi4z,diff_xy))
                                           float epsil_ij = (-temp_coef * (epsil_const.row(idx)\
                                                                           + 2.0*xi4z.row(idx)*diff_xy)).value();

                                           float A_ij = it.value();
                                           // eq (34)
                                           Bi += double(A_ij * beta_ij);
                                           Ci += double(A_ij * (gamma_ij+beta_ij*beta_ij/2.0));
                                           Di += double(A_ij * (delta_ij+beta_ij*gamma_ij + beta_ij*beta_ij*beta_ij/6.0));
                                           Ei += double(A_ij * (epsil_ij+beta_ij*delta_ij+1/2.0*beta_ij*beta_ij*gamma_ij\
                                                                + 1/2.0*gamma_ij*gamma_ij + 1/24.0*beta_ij*beta_ij*beta_ij*beta_ij));
                                         }
        
                                         // sum them up
                                         BCDE_lock.lock();
                                         B+=Bi;
                                         C+=Ci;
                                         D+=Di;
                                         E+=Ei;
                                         BCDE_lock.unlock();
                                       });

    Eigen::VectorXf p_coef(4);
    p_coef << 4.0*float(E),3.0*float(D),2.0*float(C),float(B);
    
    // solve polynomial roots
    Eigen::VectorXcf rc = poly_solver(p_coef);
    
    // find usable step size
    float temp_step = numeric_limits<float>::max();
    for(int i=0;i<rc.real().size();i++)
      if(rc(i,0).real()>0 && rc(i,0).real()<temp_step && rc(i,0).imag()==0)
        temp_step = rc(i,0).real();
    
    // if none of the roots are suitable, use min_step
    step = temp_step==numeric_limits<float>::max()? min_step:temp_step;


    // if step>0.8, just use 0.8 as step
    step = step>0.8 ? 0.8:step;
    //step *= 10;
    //step = step>0.001 ? 0.001:step;
    if (debug_print){
      std::cout<<"B: "<<B<<"C: "<<C<<"D: "<<D<<"E: "<<E<<std::endl;
      std::cout<<"step size "<<step<<std::endl;
    }
  }

  void acvo::transform_pcd(){
    tbb::parallel_for(int(0), num_moving, [&]( int j ){
                                            (*cloud_y)[j] = transform.linear()*ptr_moving_pcd->positions()[j]+transform.translation();
                                          });
    
  }

  // void acvo::convert_to_pcl_cloud(const CvoPointCloud& cvo_cloud, pcl::PointCloud<PointSegmentedDistribution> pcl_cloud){
  //   int num_points = cvo_cloud.num_points();
  //   ArrayVec3f positions = cvo_cloud.positions();
  //   Eigen::Matrix<float, Eigen::Dynamic, FEATURE_DIMENSIONS> features = cvo_cloud.features();
  //   Eigen::Matrix<float, Eigen::Dynamic, NUM_CLASSES> labels = cvo_cloud.labels();

  //   // set basic informations for pcl_cloud
  //   pcl_cloud.points.resize(num_points);
  //   pcl_cloud.width = num_points;
  //   pcl_cloud.height = 1;

  //   // loop through all points 
  //   for(int i=0; i<num_points; ++i){
  //     // set positions
  //     pcl_cloud.points[i].x = positions[i](0);
  //     pcl_cloud.points[i].y = positions[i](1);
  //     pcl_cloud.points[i].z = positions[i](2);
      
  //     // set rgb
  //     pcl_cloud.points[i].r = features(i,0);
  //     pcl_cloud.points[i].g = features(i,1);
  //     pcl_cloud.points[i].b = features(i,2);

  //     // extract features
  //     for(int j=0; j<FEATURE_DIMENSIONS; ++j){
  //       pcl_cloud.points[i].features[j] = features(i,j);
  //     }
  //     float cur_label_value = -1;
  //     pcl_cloud.points[i].label = -1;

  //     // extract labels
  //     for(int k=0; k<NUM_CLASSES; ++k){
  //       pcl_cloud.points[i].label_distribution[k] = labels(i,k);
  //       if(pcl_cloud.points[i].label_distribution[k]>cur_label_value){
  //         pcl_cloud.points[i].label = k;
  //         cur_label_value = pcl_cloud.points[i].label_distribution[k];
  //       }
  //     }
  //   }
  // }

  int acvo::align(){
    int ret = 0;
    
    int n = tbb::task_scheduler_init::default_num_threads();
    std::cout<<"num_thread: "<<n<<std::endl;

    // loop until MAX_ITER
    iter = MAX_ITER;

    auto start = chrono::system_clock::now();

    chrono::duration<double> t_transform_pcd = chrono::duration<double>::zero();
    chrono::duration<double> t_compute_flow = chrono::duration<double>::zero();
    chrono::duration<double> t_compute_step = chrono::duration<double>::zero();
    for(int k=0; k<MAX_ITER; k++){
      // update transformation matrix
      if(debug_print) std::cout<<"-------------------- iteration: "<<k<<"--------------"<<std::endl;

      update_tf();

      start = chrono::system_clock::now();
      // apply transform to the point cloud
      transform_pcd();
      auto end = std::chrono::system_clock::now();
      t_transform_pcd += (end - start); 
        
      // compute omega and v
      start = chrono::system_clock::now();
      compute_flow();
      if (debug_print)std::cout<<"omega: \n"<<omega<<"\nv: \n"<<v<<std::endl;
      end = std::chrono::system_clock::now();
      t_compute_flow += (end - start);

      // compute step size for integrating the flow
      start = chrono::system_clock::now();
      compute_step_size();
      end = std::chrono::system_clock::now();
      t_compute_step += (end -start);
      
      // stop if the step size is too small
      if(omega.cast<double>().norm()<eps && v.cast<double>().norm()<eps){
        iter = k;
        std::cout<<"norm, omega: "<<omega.norm()<<", v: "<<v.norm()<<std::endl;
        if (omega.norm() < 1e-8 && v.norm() < 1e-8)
          ret = -1;
        break;
      }

      // stacked omega and v for finding dtrans
      Eigen::VectorXf vec_joined(omega.size()+v.size());
      vec_joined << omega, v;

      // find the change of translation matrix dtrans
      Eigen::MatrixXf dtrans = Exp_SEK3(vec_joined, step);

      // extract dR and dT from dtrans
      Eigen::Matrix3f dR = dtrans.block<3,3>(0,0);
      Eigen::Vector3f dT = dtrans.block<3,1>(0,3);

      // calculate new R and T
      T = R * dT + T;
      R = R * dR;

      // if the se3 distance is smaller than eps2, break
      if (debug_print)std::cout<<"dist: "<<dist_se3(dR, dT)<<std::endl;
      if(dist_se3(dR,dT)<eps_2){
        iter = k;
        std::cout<<"dist: "<<dist_se3(dR,dT)<<std::endl;
        break;
      }

      ell = ell - dl_step*dl;
      if(ell>=ell_max){
        ell = ell_max*0.7;
        ell_max = ell_max*0.7;
      }
              
      ell = (ell<ell_min)? ell_min:ell;
      // std::cout<<ell<<std::endl;

      // std::cout<<"iter: "<<k<<std::endl;
      if(debug_print){
        std::cout<<"ell: "<<ell<<std::endl;
        std::cout<<"dl: "<<dl<<std::endl;
        std::cout<<"R: "<<R<<std::endl;
        std::cout<<"T: "<<T<<std::endl;
        std::cout<<"num non zeros in A: "<<A.nonZeros()<<std::endl;
        std::cout<<"inner product before normalized: "<<A.sum()<<std::endl;
        std::cout<<"inner product after normalized: "<<A.sum()/num_fixed/num_moving*1e6<<std::endl; 
        // std::cout<<transform.matrix()<<std::endl;
      }

      // std::cout<<dist_se3(dR,dT)<<std::endl;
    }

    std::cout<<"cvo # of iterations is "<<iter<<std::endl;
    std::cout<<"t_transform_pcd is "<<t_transform_pcd.count()<<"\n";
    std::cout<<"t_compute_flow is "<<t_compute_flow.count()<<"\n";
    std::cout<<"t_compute_step is "<<t_compute_step.count()<<"\n"<<std::flush;
    std::cout<<" final ell is "<<ell<<std::endl;
    prev_transform = transform.matrix();
    // accum_tf.matrix() = transform.matrix().inverse() * accum_tf.matrix();
    accum_tf = accum_tf * transform.matrix();
    accum_tf_vis = accum_tf_vis * transform.matrix();   // accumilate tf for visualization
    update_tf();

    delete cloud_x;
    delete cloud_y;
    if (is_logging) {
      auto & Tmat = transform.matrix();
      fprintf(relative_transform_file , "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
              Tmat(0,0), Tmat(0,1), Tmat(0,2), Tmat(0,3),
              Tmat(1,0), Tmat(1,1), Tmat(1,2), Tmat(1,3),
              Tmat(2,0), Tmat(2,1), Tmat(2,2), Tmat(2,3)
              );
      fflush(relative_transform_file);
    }

    return ret;

  }
 

  void acvo::set_pcd(const CvoPointCloud& source_points,
                    const CvoPointCloud& target_points,
                    const Eigen::Affine3f & init_guess_transform,
                    bool is_using_init_guess) {

    if (source_points.num_points() == 0 || target_points.num_points() == 0) {
      return;
    }

    //  set the unique_ptr to the source and target point clouds
    ptr_fixed_pcd = & source_points;
    ptr_moving_pcd = & target_points;

    
    // get total number of points
    num_fixed = ptr_fixed_pcd->num_points();
    num_moving = ptr_moving_pcd->num_points();
    std::cout<<"num fixed: "<<num_fixed<<std::endl;
    std::cout<<"num moving: "<<num_moving<<std::endl;

    // extract cloud x and y
    cloud_x = new ArrayVec3f (ptr_fixed_pcd->positions());
    cloud_y = new ArrayVec3f (ptr_moving_pcd->positions());
    // std::cout<<"fixed[0] \n"<<ptr_fixed_pcd->positions()[0]<<"\nmoving[0] "<<ptr_moving_pcd->positions()[0]<<"\n";
    // std::cout<<"fixed[0] \n"<<(*cloud_x)[0]<<"\nmoving[0] "<<(*cloud_y)[0]<<"\n";
    // std::cout<<"fixed[0] features \n "<<ptr_fixed_pcd->features().row(0)<<"\n  moving[0] feature "<<ptr_moving_pcd->features().row(0)<<"\n";

    // std::cout<<"init cvo: \n"<<transform.matrix()<<std::endl;
    if (is_using_init_guess) {
      transform = init_guess_transform;
      R = transform.linear();
      T = transform.translation();
    }
    std::cout<<"[Cvo ] the init guess for the transformation is \n"
             <<transform.matrix()<<std::endl;
    if (is_logging) {
      auto & Tmat = transform.matrix();
      fprintf(init_guess_file, "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
              Tmat(0,0), Tmat(0,1), Tmat(0,2), Tmat(0,3),
              Tmat(1,0), Tmat(1,1), Tmat(1,2), Tmat(1,3),
              Tmat(2,0), Tmat(2,1), Tmat(2,2), Tmat(2,3)
              );
      fflush(init_guess_file);
    }

    ell = ell_init;
    dl = 0;
    ell_max = ell_max_fixed;
    A_trip_concur.reserve(num_moving*20);
    A.resize(num_fixed,num_moving);
    Axx.resize(num_fixed,num_fixed);
    Ayy.resize(num_moving,num_moving);
    A.setZero();
    Axx.setZero();
    Ayy.setZero();
  }

  
  float acvo::inner_product() const {
    return A.sum()/num_fixed*1e6/num_moving;
  }

  float acvo::inner_product_normalized() const {
    return A.sum()/A.nonZeros();
    // return A.sum()/num_fixed*1e6/num_moving;
  }

  int acvo::number_of_non_zeros_in_A() const{
    std::cout<<"num of non-zeros in A: "<<A.nonZeros()<<std::endl;
    return A.nonZeros();
  }

  float acvo::inner_product(const CvoPointCloud& source_points,
                                const CvoPointCloud& target_points,
                                const Eigen::Affine3f & s2t_frame_transform) const {
    if (source_points.num_points() == 0 || target_points.num_points() == 0) {
      return 0;
      }

    ArrayVec3f fixed_positions = source_points.positions();
    ArrayVec3f moving_positions = target_points.positions();

    Eigen::Matrix3f rot = s2t_frame_transform.linear();
    Eigen::Vector3f trans = s2t_frame_transform.translation();

    // transform moving points
    tbb::parallel_for(int(0), target_points.num_points(), [&]( int j ){
                                                       moving_positions[j] = (rot*moving_positions[j]+trans).eval();
                                                     });
    Eigen::SparseMatrix<float,Eigen::RowMajor> A_mat;
    tbb::concurrent_vector<Trip_t> A_trip_concur_;
    A_trip_concur_.reserve(target_points.num_points() * 20);
    A_mat.resize(source_points.num_points(), target_points.num_points());
    A_mat.setZero();
    se_kernel_init_ell(&source_points, &target_points, &fixed_positions, &moving_positions, A_mat, A_trip_concur_  );
    //std::cout<<"num of non-zeros in A: "<<A_mat.nonZeros()<<std::endl;
    // return A_mat.sum()/A_mat.nonZeros();
    return A_mat.sum()/fixed_positions.size()*1e6/moving_positions.size() ;
  }

}