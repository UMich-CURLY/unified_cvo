/* ----------------------------------------------------------------------------
 * Copyright 2019, Tzu-yuan Lin <tzuyuan@umich.edu>, Maani Ghaffari <maanigj@umich.edu>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   cvo.cpp
 *  @author Tzu-yuan Lin, Maani Ghaffari 
 *  @brief  Source file for contineuous visual odometry registration
 *  @date   November 03, 2019
 **/

#include "cvo/CvoGPU.hpp"
#include "cvo/SparseKernelMat.cuh"
#include "cvo/CvoGPU_impl.cuh"
#include "cvo/CvoState.cuh"
#include "cvo/KDTreeVectorOfVectorsAdaptor.h"
#include "cvo/LieGroup.h"
#include "cvo/nanoflann.hpp"
#include "cvo/CvoParams.hpp"
#include "cvo/gpu_utils.cuh"
#include "utils/PointSegmentedDistribution.hpp"
#include "cupointcloud/point_types.h"
#include "cupointcloud/cupointcloud.h"
#include "cukdtree/cukdtree.cuh"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <ctime>
#include <functional>
#include <cassert>

using namespace std;
using namespace nanoflann;

namespace cvo{

  namespace cukdtree = perl_registration;

  static bool is_logging = false;
  static bool debug_print = false;
  
  void CvoPointCloud_to_gpu(const CvoPointCloud & cvo_cloud, CvoPointCloudGPU::SharedPtr gpu_cloud ) {
    int num_points = cvo_cloud.num_points();
    const ArrayVec3f & positions = cvo_cloud.positions();
    const Eigen::Matrix<float, Eigen::Dynamic, FEATURE_DIMENSIONS> & features = cvo_cloud.features();
    const Eigen::Matrix<float, Eigen::Dynamic, NUM_CLASSES> & labels = cvo_cloud.labels();

    // set basic informations for pcl_cloud
    thrust::host_vector<CvoPoint> host_cloud(num_points);
    //gpu_cloud->resize(num_points);

    // loop through all points
    for(int i=0; i<num_points; ++i){
      // set positions
      (host_cloud)[i].x = positions[i](0);
      (host_cloud)[i].y = positions[i](1);
      (host_cloud)[i].z = positions[i](2);
      (host_cloud)[i].r = (uint8_t)std::min(255.0, (features(i,0) * 255.0));
      (host_cloud)[i].g = (uint8_t)std::min(255.0, (features(i,1) * 255.0));
      (host_cloud)[i].b = (uint8_t)std::min(255.0, (features(i,2) * 255.0));

      memcpy(host_cloud[i].features, features.row(i).data(), FEATURE_DIMENSIONS * sizeof(float));

      //float cur_label_value = -1;
      labels.row(i).maxCoeff(&host_cloud[i].label);

      memcpy(host_cloud[i].label_distribution, labels.row(i).data(), NUM_CLASSES * sizeof(float));
    }

    gpu_cloud->points = host_cloud;
    
    
  }

  void CvoPointCloud_to_pcl(const CvoPointCloud& cvo_cloud, pcl::PointCloud<CvoPoint> & pcl_cloud){
    int num_points = cvo_cloud.num_points();
    const ArrayVec3f & positions = cvo_cloud.positions();
    const Eigen::Matrix<float, Eigen::Dynamic, FEATURE_DIMENSIONS> & features = cvo_cloud.features();
    const Eigen::Matrix<float, Eigen::Dynamic, NUM_CLASSES> & labels = cvo_cloud.labels();

    // set basic informations for pcl_cloud
    pcl_cloud.points.resize(num_points);
    pcl_cloud.width = num_points;
    pcl_cloud.height = 1;
    // loop through all points
    for(int i=0; i<num_points; ++i){
      // set positions
      pcl_cloud.points[i].x = positions[i](0);
      pcl_cloud.points[i].y = positions[i](1);
      pcl_cloud.points[i].z = positions[i](2);
      pcl_cloud.points[i].r = (uint8_t)std::min(255.0, (features(i,0) * 255.0));
      pcl_cloud.points[i].g = (uint8_t)std::min(255.0, (features(i,1) * 255.0)) ;
      pcl_cloud.points[i].b = (uint8_t)std::min(255.0, (features(i,2) * 255.0));

      memcpy(pcl_cloud[i].features, features.row(i).data(), FEATURE_DIMENSIONS * sizeof(float));

      labels.row(i).maxCoeff(&pcl_cloud.points[i].label);
      memcpy(pcl_cloud[i].label_distribution, labels.row(i).data(), NUM_CLASSES * sizeof(float));
    }
  }

  CvoGPU::CvoGPU(const std::string & param_file) {
    read_CvoParams(param_file.c_str(), &params);

    cudaMalloc((void**)&params_gpu, sizeof(CvoParams) );
    cudaMemcpy( (void*)params_gpu, &params, sizeof(CvoParams), cudaMemcpyHostToDevice  );
  }

  CvoGPU::~CvoGPU() {
    cudaFree(params_gpu);
    
  }


  Eigen::VectorXcf poly_solver(const Eigen::VectorXf& coef){
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

  float dist_se3(const Eigen::Matrix3f& R, const Eigen::Vector3f& T)  {
    // create transformation matrix
    Eigen::Matrix4f temp_transform = Eigen::Matrix4f::Identity();
    temp_transform.block<3,3>(0,0)=R;
    temp_transform.block<3,1>(0,3)=T;
    
    // distance = frobenius_norm(logm(trans))
    float d = temp_transform.log().norm();
    
    return d;
  }

  struct transform_point
  {
    const Mat33f * R;
    const Vec3f * T;

    transform_point(const Mat33f * R_gpu, const Vec3f * T_gpu): R(R_gpu), T(T_gpu){}
    
    __host__ __device__
    CvoPoint operator()(const CvoPoint & p_init)
    {
      CvoPoint result;
      Eigen::Vector3f input;
      input << p_init.x, p_init.y, p_init.z;
      Eigen::Vector3f trans = (*R) * input + (*T);
      result.x = trans(0);
      result.y = trans(1);
      result.z = trans(2);
      result.r = p_init.r;
      result.g = p_init.g;
      result.b = p_init.b;
      result.label = p_init.label;
      memcpy(result.features, p_init.features, FEATURE_DIMENSIONS * sizeof(float) );
      memcpy(result.label_distribution, p_init.label_distribution, NUM_CLASSES * sizeof(float) );
      return result;
    }
  };


  void  transform_pointcloud_thrust(std::shared_ptr<CvoPointCloudGPU> init_cloud,
                                            std::shared_ptr<CvoPointCloudGPU> transformed_cloud,
                                            Mat33f * R_gpu, Vec3f * T_gpu
                                            ) {
    thrust::transform( init_cloud->begin(), init_cloud->end(),  transformed_cloud->begin(), 
                       transform_point(R_gpu, T_gpu));
    
  }
  

  void update_tf(const Mat33f & R, const Vec3f & T,
                 // outputs
                 CvoState * cvo_state,
                 Eigen::Ref<Mat44f> transform
                 )  {
    // transform = [R', -R'*T; 0,0,0,1]
    transform.block<3,3>(0,0) = R.transpose();
    transform.block<3,1>(0,3) = -R.transpose()*T;

    cudaMemcpy(cvo_state->R_gpu->data(), R.data(), sizeof(float)*9, cudaMemcpyHostToDevice);
    cudaMemcpy(cvo_state->T_gpu->data(), T.data(), sizeof(float)*3, cudaMemcpyHostToDevice );

    if (debug_print) std::cout<<"transform mat "<<transform<<"\n";
  }


  typedef KDTreeVectorOfVectorsAdaptor<cloud_t, float>  kd_tree_t;

  __global__ void fill_in_A_mat_gpu(const CvoParams * cvo_params,
                                    //SquareExpParams * se_params,
                                    CvoPoint * points_a,
                                    int a_size,
                                    CvoPoint * points_b,
                                    int * kdtree_inds,
                                    float ell,
                                    // output
                                    SparseKernelMat * A_mat // the kernel matrix!
                                    ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > a_size - 1)
      return;

    float s2= cvo_params->sigma * cvo_params->sigma;
    float l = ell ;
    float c_ell = cvo_params->c_ell;
    float s_ell = cvo_params->s_ell;
    float sp_thres = cvo_params->sp_thres;
    float c_sigma = cvo_params->c_sigma;

    // convert k threshold to d2 threshold (so that we only need to calculate k when needed)
    float d2_thres = -2.0*l*l*log(sp_thres/s2);
    float d2_c_thres = -2.0*c_ell*c_ell*log(sp_thres/c_sigma/c_sigma);

    //Eigen::VectorXf feature_a = feature_a_gpu->row(i).transpose();
    CvoPoint * p_a =  &points_a[i];
    //float feature_a[5] = {(float)p_a->r, (float)p_a->g, (float)p_a->b,  p_a->gradient[0], p_a->gradient[1]  };
#ifdef IS_USING_SEMANTICS
    float * label_a = p_a ->semantics;
#endif


    //int * mat_inds = new int [kd_tree_max_leafIf they all have the same size, tha];
    int num_inds = 0;
    for (size_t j = 0; j < KDTREE_K_SIZE ; j++) {
      int ind_b = kdtree_inds[i * KDTREE_K_SIZE  + j];
      
      //float d2 = (cloud_y_gpu[ind_b] - cloud_x_gpu[i]).squaredNorm();
      // d2 = (x-y)^2
      float d2 = squared_dist( points_b[j] ,*p_a );
      
      if(d2<d2_thres  ){
        //float feature_b[5] = {(float)p_a->r, (float)p_a->g, (float)p_a->b,  p_a->gradient[0], p_a->gradient[1]  };
        CvoPoint * p_b = &points_b[ind_b];
        float d2_color = squared_dist<float>(p_a->features, p_b->features, FEATURE_DIMENSIONS);

#ifdef IS_USING_SEMANTICS            
        float d2_semantic = squared_dist<float>(p_a->label_distribution, p_b->label_distribution, NUM_CLASSES);
#endif
            
        if(d2_color<d2_c_thres){
          float k = s2*exp(-d2/(2.0*l*l));
          float ck = c_sigma*c_sigma*exp(-d2_color/(2.0*c_ell*c_ell));
#ifdef IS_USING_SEMANTICS              
          float sk = cvo_params->s_sigma*cvo_params->s_sigma*exp(-d2_semantic/(2.0*s_ell*s_ell));
#else
          float sk = 1;
#endif              
          float a = ck*k*sk;
          // concrrent access !
          if (a > cvo_params->sp_thres){
            A_mat->mat[i * A_mat->cols + num_inds] = a;
            A_mat->ind_row2col[i * A_mat->cols + num_inds] = ind_b;
          }// else {
           // A_mat->mat[i][num_inds] = 0;
           // A_mat->ind_row2col[i][num_inds] = -1;
          //}
          num_inds++;             
            
        }
      }

      

      
    }
    //delete mat_inds;
    
  }

  void se_kernel(//SquareExpParams * se_params_gpu,
                 const CvoParams * params_gpu,
                 std::shared_ptr<CvoPointCloudGPU> points_fixed,
                 std::shared_ptr<CvoPointCloudGPU> points_moving,
                 float ell,
                 perl_registration::cuKdTree<CvoPoint>::SharedPtr kdtree,
                 // output
                 SparseKernelMat * A_mat
                 )  {
    if (!kdtree && !kdtree->IsInputCloudSet()) {
      kdtree->SetInputCloud(points_moving );
    }
    thrust::device_vector<int> indices;
    kdtree->NearestKSearch(points_fixed, KDTREE_K_SIZE, indices );
    int * ind_device = thrust::raw_pointer_cast(&indices[0]);
    //float ** A_mat_gpu = thrust::raw_pointer_cast(&A_mat[0]);
    cudaMemset( (void *)(A_mat->mat), 0,  A_mat->rows * A_mat->cols * sizeof(float)   );
    cudaMemset( (void *)(A_mat->ind_row2col), 0xff,  A_mat->rows * A_mat->cols * sizeof(int));

    int fixed_size = points_fixed->points.size();

    CvoPoint * points_fixed_raw = thrust::raw_pointer_cast (  &(points_fixed->points)[0] );
    CvoPoint * points_moving_raw = thrust::raw_pointer_cast( &(points_moving->points)[0] );
    

    fill_in_A_mat_gpu<<<(points_fixed->size() / CUDA_BLOCK_SIZE)+1, CUDA_BLOCK_SIZE  >>>(params_gpu,
                                                                                         //se_params_gpu,
                                                                                         points_fixed_raw,
                                                                                         fixed_size,
                                                                                         points_moving_raw,
                                                                                         ind_device,
                                                                                         ell,
                                                                                         // output
                                                                                         A_mat // the kernel matrix!
                                                                                         );
  }

  __global__ void compute_flow_gpu(const CvoParams * cvo_params,
                                   float ell,
                                   CvoPoint * cloud_x, CvoPoint * cloud_y,
                                   SparseKernelMat * A, SparseKernelMat * Axx, SparseKernelMat * Ayy,
                                   // outputs: thrust vectors
                                   Eigen::Vector3d * omega_all_gpu, Eigen::Vector3d * v_all_gpu,
                                   double * partial_dl ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > A->rows - 1)
      return;

    float ell_3 = (ell) * (ell) * (ell);
    
    VecKDf_row Ai = Eigen::Map<VecKDf_row>(A->mat + i * KDTREE_K_SIZE);
    VecKDf_row Axxi = Eigen::Map<VecKDf_row>(Axx->mat + i * KDTREE_K_SIZE) ;
    VecKDf_row Ayyi = VecKDf_row::Zero();
    if (i < Ayy->rows)
      Ayyi = Eigen::Map<VecKDf_row>(Ayy->mat + i * KDTREE_K_SIZE);
    MatKD3f cross_xy = MatKD3f::Zero();
    MatKD3f diff_yx = MatKD3f::Zero();
    MatKD3f diff_xx = MatKD3f::Zero();
    MatKD3f diff_yy = MatKD3f::Zero();
    VecKDf sum_diff_yx_2 = VecKDf::Zero();
    VecKDf sum_diff_xx_2 = VecKDf::Zero();
    VecKDf sum_diff_yy_2 = VecKDf::Zero();
    partial_dl[i] = 0;
    
    CvoPoint * px = &cloud_x[i];
    Eigen::Vector3f px_eig;
    px_eig<< px->x , px->y, px->z;
    //float px_arr[3] = {px->x, px->y, px->z};

    for (int j = 0; j < KDTREE_K_SIZE; j++) {
      int idx = A->ind_row2col[i * KDTREE_K_SIZE + j];
      //float val = A->mat[i * KDTREE_K_SIZE + j];
      CvoPoint * py = &cloud_y[idx];
      //float py_arr = {py->x, py->y, py->z};
      Eigen::Vector3f py_eig;
      py_eig << py->x, py->y, py->z;
      cross_xy.row(j) = px_eig.cross(py_eig).transpose();
      diff_yx.row(j) = (py_eig - px_eig ).transpose();
      sum_diff_yx_2(j) = (py_eig - px_eig).squaredNorm();
      //cross3(px_arr, py_arr, cross_xy + 3*j);
      //float diff_yx_arr = {py->x-px->x, py->y-px->y, py->z-px->z};
      //memcpy( diff_yx + 3 * j, diff_yx_arr, 3 * sizeof(float) );
      //sum_diff_yx_2[j] = square_norm(diff_yx_arr, 3);
    }
    for (int j = 0; j<KDTREE_K_SIZE; j++) {
      int idx = Axx->ind_row2col[i * KDTREE_K_SIZE + j];
      //float val = Axx->mat[i*KDTREE_K_SIZE +j];
      CvoPoint * py = &cloud_x[idx];
      Eigen::Vector3f py_eig;
      py_eig << py->x, py->y, py->z;
      //cross_xx.row(j) = px_eig.cross(py_eig).transpose();
      diff_xx.row(j) = (py_eig - px_eig ).transpose();
      sum_diff_xx_2(j) = (py_eig - px_eig).squaredNorm();
    }
    if (i < Ayy->rows) {
      auto py_left = &cloud_y[i];
      Eigen::Vector3f py_left_eig;
      py_left_eig << py_left->x, py_left->y, py_left->z;
      for (int j = 0; j<KDTREE_K_SIZE; j++) {
        int idx = Ayy->ind_row2col[i * KDTREE_K_SIZE + j];
        //float val = Ayy->mat[i*KDTREE_K_SIZE +j];
        CvoPoint * py = &cloud_y[idx];
        Eigen::Vector3f py_eig;
        py_eig << py->x, py->y, py->z;
        //cross_xx.row(j) = px_eig.cross(py_eig).transpose();
        diff_yy.row(j) = (py_eig - py_left_eig ).transpose();
        sum_diff_yy_2(j) = (py_eig - py_left_eig).squaredNorm();
      }

      // partial dl
      // TOOD
      partial_dl[i] += double(1/ell * (Ayyi*sum_diff_yy_2).value()  );
    }

    omega_all_gpu[i] =  (1/cvo_params->c*Ai*cross_xy).cast<double>();
    v_all_gpu[i] = (1/cvo_params->d * Ai * diff_yx).cast<double>();

  partial_dl[i] -= double(2*(1/ell_3*(Ai*sum_diff_yx_2).value() )) ;
    
    // update dl from Axx
  partial_dl[i] += double((1/ell_3*(Axxi*sum_diff_xx_2).value()));    
    
  }

  
  __global__ void compute_flow_gpu_ell_Ayy(const CvoParams * cvo_params,
                                           float  ell,
                                           CvoPoint * cloud_y,
                                           SparseKernelMat * Ayy,
                                           // outputs: thrust vectors
                                           double * partial_dl_Ayy ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > Ayy->rows - 1)
      return;

    CvoPoint * px = &cloud_y[i];
    Eigen::Vector3f px_eig;
    px_eig << px->x , px->y, px->z;
    partial_dl_Ayy[i] = 0;
    float ell_3 = (ell) * (ell) * (ell);
    
    VecKDf_row Ayyi = Eigen::Map<VecKDf_row>(Ayy->mat + i * KDTREE_K_SIZE);
    MatKD3f diff_yy = MatKD3f::Zero();
    VecKDf sum_diff_yy_2 = VecKDf::Zero();
    for (int j = 0; j < KDTREE_K_SIZE; j++) {
      int idx = Ayy->ind_row2col[i * KDTREE_K_SIZE + j];
      CvoPoint * py = &cloud_y[idx];
      Eigen::Vector3f py_eig;
      py_eig << py->x, py->y, py->z;
      diff_yy.row(j) = (py_eig - px_eig).transpose();
      sum_diff_yy_2(j) = diff_yy.row(j).squaredNorm();
    }
    partial_dl_Ayy[i] += double(1/ell_3 * (Ayyi * sum_diff_yy_2).value()); 
    
  }  

  void compute_flow(CvoState * cvo_state, const CvoParams * params_gpu )  {

    auto start = chrono::system_clock::now();
    //auto end = chrono::system_clock::now();
                            
    // compute SE kernel for Axy
    cukdtree::cuKdTree<CvoPoint>::SharedPtr kdtree_moving_new(new cukdtree::cuKdTree<CvoPoint>);
    se_kernel(params_gpu, cvo_state->cloud_x_gpu, cvo_state->cloud_y_gpu ,
              cvo_state->ell, 
              kdtree_moving_new,
              cvo_state->A);
    if (debug_print ) std::cout<<"nonzeros in A "<<nonzeros(cvo_state->A)<<std::endl;
    // compute SE kernel for Axx and Ayy
    se_kernel(params_gpu, cvo_state->cloud_x_gpu  ,cvo_state->cloud_x_gpu,
              cvo_state->ell, 
              cvo_state->kdtree_fixed_points, cvo_state->Axx);
    se_kernel(params_gpu, cvo_state->cloud_y_gpu  ,cvo_state->cloud_y_gpu,
              cvo_state->ell,
              kdtree_moving_new, cvo_state->Ayy);

    auto end = chrono::system_clock::now();
    std::cout<<"time for se_kernel is "<<(end- start).count()<<std::endl;

    // some initialization of the variables
    compute_flow_gpu<<<cvo_state->A->rows / CUDA_BLOCK_SIZE + 1 ,CUDA_BLOCK_SIZE>>>(params_gpu,
                                                                                    cvo_state->ell,
                                                                                    thrust::raw_pointer_cast(&(cvo_state->cloud_x_gpu->points)[0]),
                                                                                    thrust::raw_pointer_cast(&(cvo_state->cloud_y_gpu->points)[0]),
                                                                                    cvo_state->A, cvo_state->Axx, cvo_state->Ayy,
                                                                                    thrust::raw_pointer_cast(&cvo_state->omega_gpu[0]),
                                                                                    thrust::raw_pointer_cast(&cvo_state->v_gpu[0]),
                                                                                    thrust::raw_pointer_cast(&cvo_state->partial_dl_gradient[0])
                                                                                    );
    if (cvo_state->num_moving > cvo_state->num_fixed )
      compute_flow_gpu_ell_Ayy<<<cvo_state->Ayy->rows / CUDA_BLOCK_SIZE + 1 ,CUDA_BLOCK_SIZE>>>(params_gpu,
                                                                                                cvo_state->ell,
                                                                                                thrust::raw_pointer_cast(&(cvo_state->cloud_y_gpu->points)[0]),
                                                                                                cvo_state->Ayy,
                                                                                                thrust::raw_pointer_cast(&cvo_state->partial_dl_Ayy[0])
                                                                                                );

    // update them to class-wide variables
    thrust::plus<double> plus_double;
    thrust::plus<Eigen::Vector3d> plus_vector;
    Eigen::Vector3f omega = (thrust::reduce(cvo_state->omega_gpu.begin(), cvo_state->omega_gpu.end())).cast<float>();
                                            //Eigen::Vector3d::Zero(), thrust::plus<Eigen::Vector3d>() )).cast<float>() ;
    cudaMemcpy(cvo_state->omega, &omega, sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice );
    Eigen::Vector3f v = (thrust::reduce(cvo_state->v_gpu.begin(), cvo_state->v_gpu.end())).cast<float>();
                                        // Eigen::Vector3d::Zero(), plus_vector)).cast<float>();
    cudaMemcpy(cvo_state->v, &v, sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice );
    cvo_state->dl = thrust::reduce(cvo_state->partial_dl_gradient.begin(), cvo_state->partial_dl_gradient.end()) +
      thrust::reduce(cvo_state->partial_dl_Ayy.begin(), cvo_state->partial_dl_Ayy.end());
    
    int Axx_nonzero = nonzeros(cvo_state->Axx);
    int Ayy_nonzero = nonzeros(cvo_state->Ayy);
    int A_nonzero = nonzeros(cvo_state->A);
    cvo_state->dl = cvo_state->dl/double ( Axx_nonzero+Ayy_nonzero-2*A_nonzero);
    
  }

  __global__ void compute_step_size_xi(Eigen::Vector3f * omega ,
                                       Eigen::Vector3f * v,
                                       CvoPoint * cloud_y,
                                       int num_moving,
                                       // outputs
                                       Eigen::Vector3f_row * xiz,
                                       Eigen::Vector3f_row * xi2z,
                                       Eigen::Vector3f_row * xi3z,
                                       Eigen::Vector3f_row * xi4z,
                                       float * normxiz2,
                                       float * xiz_dot_xi2z,
                                       float * epsil_const
                                       ) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j > num_moving-1 )
      return;
    Eigen::Matrix3f omega_hat;
    skew(omega, &omega_hat);
    Eigen::Vector3f cloud_yi;
    cloud_yi << cloud_y[j].x , cloud_y[j].y, cloud_y[j].z;

    xiz[j] = omega->transpose().cross(cloud_yi.transpose()) + v->transpose();
    xi2z[j] = (omega_hat*omega_hat*cloud_yi                         \
                   +(omega_hat*(*v))).transpose();    // (xi^2*z+xi*v)
    xi3z[j] = (omega_hat*omega_hat*omega_hat*cloud_yi               \
                   +(omega_hat*omega_hat*(*v)  )).transpose();  // (xi^3*z+xi^2*v)
    xi4z[j] = (omega_hat*omega_hat*omega_hat*omega_hat*cloud_yi     \
                   +(omega_hat*omega_hat*omega_hat*(*v)  )).transpose();    // (xi^4*z+xi^3*v)
    normxiz2[j] = xiz[j].squaredNorm();
    xiz_dot_xi2z[j]  = (-xiz[j] .dot(xi2z[j]));
    epsil_const[j] = xi2z[j].squaredNorm()+2*xiz[j].dot(xi3z[j]);
    
  }

  __global__ void compute_step_size_poly_coeff(float temp_coef,
                                               int num_fixed,
                                               SparseKernelMat * A,
                                               CvoPoint * cloud_x,
                                               CvoPoint * cloud_y,
                                               Eigen::Vector3f_row * xiz,
                                               Eigen::Vector3f_row * xi2z,
                                               Eigen::Vector3f_row * xi3z,
                                               Eigen::Vector3f_row * xi4z,
                                               float * normxiz2,
                                               float * xiz_dot_xi2z,
                                               float * epsil_const,
                                               
                                               // output
                                               double * B,
                                               double * C,
                                               double * D,
                                               double * E
                                               ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > num_fixed-1 )
      return;
    B[i] = 0;
    C[i] = 0;
    D[i] = 0;
    E[i] = 0;
    Eigen::Vector3f px;
    px << cloud_x[i].x, cloud_x[i].y, cloud_x[i].z;
    for (int j = 0; j < A->cols; j++) {
      int idx = A->ind_row2col[i * KDTREE_K_SIZE + j];
      Eigen::Vector3f py;
      py << cloud_y[idx].x, cloud_y[idx].y, cloud_y[idx].z;
      Eigen::Vector3f diff_xy = (px - py);

      // beta_i = -1/l^2 * dot(xiz,diff_xy)
      float beta_ij = (-2.0*temp_coef * (xiz[idx] *diff_xy  ).value()  );
      // gamma_i = -1/(2*l^2) * (norm(xiz).^2 + 2*dot(xi2z,diff_xy))
      float gamma_ij = (-temp_coef * (normxiz2[idx] \
                                      + (2.0*xi2z[idx]  *diff_xy).value())  );
      // delta_i = 1/l^2 * (dot(-xiz,xi2z) + dot(-xi3z,diff_xy))
      float delta_ij = (2.0*temp_coef * (xiz_dot_xi2z[idx]  \
                                         + (-xi3z[idx] *diff_xy).value ( ) ));
      // epsil_i = -1/(2*l^2) * (norm(xi2z).^2 + 2*dot(xiz,xi3z) + 2*dot(xi4z,diff_xy))
      float epsil_ij = (-temp_coef * (epsil_const[idx] \
                                      + (2.0*xi4z[idx]*diff_xy).value()  ));

      float A_ij = A->mat[i * KDTREE_K_SIZE + j];
      // eq (34)
      B[i] += double(A_ij * beta_ij);
      C[i] += double(A_ij * (gamma_ij+beta_ij*beta_ij/2.0));
      D[i] += double(A_ij * (delta_ij+beta_ij*gamma_ij + beta_ij*beta_ij*beta_ij/6.0));
      E[i] += double(A_ij * (epsil_ij+beta_ij*delta_ij+1/2.0*beta_ij*beta_ij*gamma_ij\
                           + 1/2.0*gamma_ij*gamma_ij + 1/24.0*beta_ij*beta_ij*beta_ij*beta_ij));

    }
    
  }

  void compute_step_size(CvoState * cvo_state, const CvoParams * params) {
    compute_step_size_xi<<<cvo_state->num_moving / CUDA_BLOCK_SIZE + 1, CUDA_BLOCK_SIZE>>>
      (cvo_state->omega, cvo_state->v,
       thrust::raw_pointer_cast( &(cvo_state->cloud_y_gpu->points[0] )), cvo_state->num_moving,
       thrust::raw_pointer_cast(cvo_state->xiz.data()),
       thrust::raw_pointer_cast(cvo_state->xi2z.data()),
       thrust::raw_pointer_cast(cvo_state->xi3z.data()),
       thrust::raw_pointer_cast(cvo_state->xi4z.data()),
       thrust::raw_pointer_cast(cvo_state->normxiz2.data()),
       thrust::raw_pointer_cast(cvo_state->xiz_dot_xi2z.data()),
       thrust::raw_pointer_cast(cvo_state->epsil_const.data())
       );
    float temp_coef = 1/(2.0*cvo_state->ell*cvo_state->ell);   // 1/(2*l^2)
    compute_step_size_poly_coeff<<<cvo_state->num_moving / CUDA_BLOCK_SIZE + 1, CUDA_BLOCK_SIZE>>>
      ( temp_coef, cvo_state->num_fixed, cvo_state->A,

        thrust::raw_pointer_cast( &(cvo_state->cloud_x_gpu->points[0])),
        thrust::raw_pointer_cast( &(cvo_state->cloud_y_gpu->points[0]  )),
       thrust::raw_pointer_cast(cvo_state->xiz.data()),
       thrust::raw_pointer_cast(cvo_state->xi2z.data()),
       thrust::raw_pointer_cast(cvo_state->xi3z.data()),
       thrust::raw_pointer_cast(cvo_state->xi4z.data()),
       thrust::raw_pointer_cast(cvo_state->normxiz2.data()),
       thrust::raw_pointer_cast(cvo_state->xiz_dot_xi2z.data()),
        thrust::raw_pointer_cast(cvo_state->epsil_const.data()),
        thrust::raw_pointer_cast(cvo_state->B.data()),
        thrust::raw_pointer_cast(cvo_state->C.data()),
        thrust::raw_pointer_cast(cvo_state->D.data()),
        thrust::raw_pointer_cast(cvo_state->E.data())
        );

    thrust::plus<double> plus_double;
    double B = thrust::reduce(cvo_state->B.begin(), cvo_state->B.end(), 0.0, plus_double);
    double C = thrust::reduce(cvo_state->C.begin(), cvo_state->C.end(), 0.0, plus_double);
    double D = thrust::reduce(cvo_state->D.begin(), cvo_state->D.end(), 0.0, plus_double);
    double E = thrust::reduce(cvo_state->E.begin(), cvo_state->E.end(), 0.0, plus_double);

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
    cvo_state->step = temp_step==numeric_limits<float>::max()? params->min_step:temp_step;


    // if step>0.8, just use 0.8 as step
    cvo_state->step = cvo_state->step>0.8 ? 0.8:cvo_state->step;
    //step *= 10;
    //step = step>0.001 ? 0.001:step;
    if (debug_print)
      std::cout<<"step size "<<cvo_state->step<<"\n\n--------------------------------------------------------------\n";

        
    
  }

  

  int CvoGPU::align(const CvoPointCloud& source_points,
                    const CvoPointCloud& target_points,
                    const Eigen::Matrix4f & init_guess_transform,
                    Eigen::Ref<Eigen::Matrix4f> transform) const {
    
    Mat33f R = init_guess_transform.block<3,3>(0,0);
    Vec3f T= init_guess_transform.block<3,1>(0,3);

    std::shared_ptr<CvoPointCloudGPU> source_gpu, target_gpu;
    CvoPointCloud_to_gpu(source_points, source_gpu);
    CvoPointCloud_to_gpu(source_points, target_gpu);

    CvoState cvo_state(source_gpu, target_gpu, params);

    int num_moving = cvo_state.num_moving;
    int num_fixed = cvo_state.num_fixed;
    
    int ret = 0;
    // loop until MAX_ITER
    int iter = params.MAX_ITER;
    Eigen::Vector3f omega, v;
    
    auto start = chrono::system_clock::now();
    chrono::duration<double> t_transform_pcd = chrono::duration<double>::zero();
    chrono::duration<double> t_compute_flow = chrono::duration<double>::zero();
    chrono::duration<double> t_compute_step = chrono::duration<double>::zero();
    
    for(int k=0; k<params.MAX_ITER; k++){
      
      cvo_state.reset_state_at_new_iter();
      
      // update transformation matrix to CvoState
      update_tf(R, T, &cvo_state, transform);

      start = chrono::system_clock::now();
      // apply transform to the point cloud
      //transform_pcd(*cvo_data, R, T );
      transform_pointcloud_thrust(cvo_state.cloud_y_gpu_init, cvo_state.cloud_y_gpu,
                                  cvo_state.R_gpu, cvo_state.T_gpu );
      auto end = std::chrono::system_clock::now();
      t_transform_pcd += (end - start); 
        
      // compute omega and v
      start = chrono::system_clock::now();
      compute_flow(&cvo_state, params_gpu);
      if (debug_print)std::cout<<"iter "<<k<< "omega: \n"<<omega<<"\nv: \n"<<v<<std::endl;
      end = std::chrono::system_clock::now();
      t_compute_flow += (end - start);

      // compute step size for integrating the flow
      start = chrono::system_clock::now();
      compute_step_size(&cvo_state, &params);
      end = std::chrono::system_clock::now();
      t_compute_step += (end -start);
      
      // stop if the step size is too small
      // TOOD: GPU!

      cudaMemcpy(omega.data(), cvo_state.omega->data(), sizeof(float)*3, cudaMemcpyDeviceToHost);
      cudaMemcpy(v.data(), cvo_state.v->data(), sizeof(float)*3, cudaMemcpyDeviceToHost);
      if(omega.cast<double>().norm()<params.eps && v.cast<double>().norm()<params.eps){
        iter = k;
        std::cout<<"norm, omega: "<<omega.norm()<<", v: "<<v.norm()<<std::endl;
        if (omega.norm() < 1e-8 && v.norm() < 1e-8) ret = -1;
        break;
      }

      // stacked omega and v for finding dtrans
      Eigen::VectorXf vec_joined(omega.size()+v.size());
      vec_joined << omega, v;

      // find the change of translation matrix dtrans
      Eigen::MatrixXf dtrans = Exp_SEK3(vec_joined, cvo_state.step);

      // extract dR and dT from dtrans
      Eigen::Matrix3f dR = dtrans.block<3,3>(0,0);
      Eigen::Vector3f dT = dtrans.block<3,1>(0,3);

      // calculate new R and T
      T = R * dT + T;
      R = R * dR;

      // if the se3 distance is smaller than eps2, break
      if (debug_print)std::cout<<"dist: "<<dist_se3(dR, dT)<<std::endl;
      if(dist_se3(dR,dT)<params.eps_2){
        iter = k;
        std::cout<<"dist: "<<dist_se3(dR,dT)<<std::endl;
        break;
      }

      cvo_state.ell = cvo_state.ell + params.dl_step*cvo_state.dl;
      if(cvo_state.ell>=cvo_state.ell_max){
        cvo_state.ell = cvo_state.ell_max*0.7;
        cvo_state.ell_max = cvo_state.ell_max*0.7;
      }
              
      cvo_state.ell = (cvo_state.ell<params.ell_min)? params.ell_min:cvo_state.ell;

      // std::cout<<"iter: "<<k<<std::endl;
      // if(debug_print){
      // std::cout<<"num non zeros in A: "<<A.nonZeros()<<std::endl;
      // std::cout<<"inner product before normalized: "<<A.sum()<<std::endl;
      // std::cout<<"inner product after normalized: "<<A.sum()/num_fixed/num_moving*1e6<<std::endl; 
      // std::cout<<transform.matrix()<<std::endl;
      // }
    }

    std::cout<<"cvo # of iterations is "<<iter<<std::endl;
    std::cout<<"t_transform_pcd is "<<t_transform_pcd.count()<<"\n";
    std::cout<<"t_compute_flow is "<<t_compute_flow.count()<<"\n";
    std::cout<<"t_compute_step is "<<t_compute_step.count()<<"\n"<<std::flush;
    std::cout<<" final ell is "<<cvo_state.ell<<std::endl;
    // prev_transform = transform.matrix();
    // accum_tf.matrix() = transform.matrix().inverse() * accum_tf.matrix();
    //accum_tf = accum_tf * transform.matrix();
    //accum_tf_vis = accum_tf_vis * transform.matrix();   // accumilate tf for visualization
    update_tf(R, T, &cvo_state, transform);
    
    /*
    if (is_logging) {
      auto & Tmat = transform.matrix();
      fprintf(relative_transform_file , "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
              Tmat(0,0), Tmat(0,1), Tmat(0,2), Tmat(0,3),
              Tmat(1,0), Tmat(1,1), Tmat(1,2), Tmat(1,3),
              Tmat(2,0), Tmat(2,1), Tmat(2,2), Tmat(2,3)
              );
      fflush(relative_transform_file);
    }
    */
    return ret;

  }


  void se_kernel_init_ell_cpu(const CvoPointCloud* cloud_a, const CvoPointCloud* cloud_b, \
                              cloud_t* cloud_a_pos, cloud_t* cloud_b_pos, \
                              Eigen::SparseMatrix<float,Eigen::RowMajor>& A_temp,
                              tbb::concurrent_vector<Trip_t> & A_trip_concur_,
                              const CvoParams & params) {
    bool debug_print = false;
    A_trip_concur_.clear();
    const float s2= params.sigma*params.sigma;
    const float l = params.ell_init;

    // convert k threshold to d2 threshold (so that we only need to calculate k when needed)
    const float d2_thres = -2.0*l*l*log(params.sp_thres/s2);
    if (debug_print ) std::cout<<"l is "<<l<<",d2_thres is "<<d2_thres<<std::endl;
    const float d2_c_thres = -2.0*params.c_ell*params.c_ell*log(params.sp_thres/params.c_sigma/params.c_sigma);
    if (debug_print) std::cout<<"d2_c_thres is "<<d2_c_thres<<std::endl;
    
    typedef KDTreeVectorOfVectorsAdaptor<cloud_t, float>  kd_tree_t;

    kd_tree_t mat_index(3 , (*cloud_b_pos), 10  );
    mat_index.index->buildIndex();

    // loop through points
    tbb::parallel_for(int(0),cloud_a->num_points(),[&](int i){
        //for(int i=0; i<num_fixed; ++i){

        const float search_radius = d2_thres;
        std::vector<std::pair<size_t,float>>  ret_matches;

        nanoflann::SearchParams params_flann;
        //params.sorted = false;

        const size_t nMatches = mat_index.index->radiusSearch(&(*cloud_a_pos)[i](0), search_radius, ret_matches, params_flann);

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
              ck = params.c_sigma*params.c_sigma*exp(-d2_color/(2.0*params.c_ell*params.c_ell));
#ifdef IS_USING_SEMANTICS              
              sk = s_sigma*s_sigma*exp(-d2_semantic/(2.0*params.s_ell*params.s_ell));
#else
              sk = 1;
#endif              
              a = ck*k*sk;

              if (a > params.sp_thres){
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


  
  float CvoGPU::inner_product(const CvoPointCloud& source_points,
                           const CvoPointCloud& target_points,
                              const Eigen::Matrix4f & s2t_frame_transform
                              ) const {
    if (source_points.num_points() == 0 || target_points.num_points() == 0) {
      return 0;
    }

    ArrayVec3f fixed_positions = source_points.positions();
    ArrayVec3f moving_positions = target_points.positions();

    Eigen::Matrix3f rot = s2t_frame_transform.block<3,3>(0,0) ;
    Eigen::Vector3f trans = s2t_frame_transform.block<3,1>(0,3) ;

    // transform moving points
    tbb::parallel_for(int(0), target_points.num_points(), [&]( int j ){
                                                            moving_positions[j] = (rot*moving_positions[j]+trans).eval();
                                                          });
    Eigen::SparseMatrix<float,Eigen::RowMajor> A_mat;
    tbb::concurrent_vector<Trip_t> A_trip_concur_;
    A_trip_concur_.reserve(target_points.num_points() * 20);
    A_mat.resize(source_points.num_points(), target_points.num_points());
    A_mat.setZero();
    se_kernel_init_ell_cpu(&source_points, &target_points, &fixed_positions, &moving_positions, A_mat, A_trip_concur_ , params );
    //std::cout<<"num of non-zeros in A: "<<A_mat.nonZeros()<<std::endl;
    // return A_mat.sum()/A_mat.nonZeros();
    return A_mat.sum()/fixed_positions.size()*1e6/moving_positions.size() ;
  }


  

 

  /*
  
  
  void CvoGPU::transform_pcd(CvoData & cvo_data, const & Mat33f R, const & Vec3f &T) {
    
    tbb::parallel_for(int(0), cvo_data.num_moving, [&]( int j ){
                                                     (cvo_data.cloud_y)[j] = R*cvo_data.ptr_moving_pcd->positions()[j]+T;
                                                   });

    
  }

  std::unique_ptr<CvoData> cvo::set_pcd(const CvoPointCloud& source_points,
                                        const CvoPointCloud& target_points,
                                        const Eigen::Matrix4f & init_guess_transform,
                                        bool is_using_init_guess,
                                        float ell_init_val,
                                        Eigen::Ref<Mat33f> R,
                                        Eigen::Ref<Vec3f> T
                                        ) const  {
    std::unique_ptr<CvoData> cvo_data(new CvoData(source_points, target_points,ell_init_val));
    // std::cout<<"fixed[0] \n"<<ptr_fixed_pcd->positions()[0]<<"\nmoving[0] "<<ptr_moving_pcd->positions()[0]<<"\n";
    // std::cout<<"fixed[0] \n"<<(*cloud_x)[0]<<"\nmoving[0] "<<(*cloud_y)[0]<<"\n";
    // std::cout<<"fixed[0] features \n "<<ptr_fixed_pcd->features().row(0)<<"\n  moving[0] feature "<<ptr_moving_pcd->features().row(0)<<"\n";

    // std::cout<<"init cvo: \n"<<transform.matrix()<<std::endl;
    Aff3f transform = init_guess_transform;
    R = transform.linear();
    T = transform.translation();
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

    return std::move(cvo_data);
  }

  
  float cvo::inner_product() const {
    return A.sum()/num_fixed*1e6/num_moving;
  }

  float cvo::inner_product_normalized() const {
    return A.sum()/A.nonZeros();
    // return A.sum()/num_fixed*1e6/num_moving;
  }

  int cvo::number_of_non_zeros_in_A() const{
    std::cout<<"num of non-zeros in A: "<<A.nonZeros()<<std::endl;
    return A.nonZeros();
  }
  */
}
