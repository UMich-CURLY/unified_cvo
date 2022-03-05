#include "cvo/CvoGPU.hpp"
#include "cvo/SparseKernelMat.cuh"
#include "cvo/Association.hpp"
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
//#include "cukdtree/cukdtree.h"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/impl/instantiate.hpp>
#include <sophus/se3.hpp>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <tbb/tbb.h>
#include <Eigen/Dense>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <ctime>
#include <functional>
#include <cassert>
#include <memory>
#include <cmath>
using namespace std;
using namespace nanoflann;


extern template struct pcl::PointSegmentedDistribution<FEATURE_DIMENSIONS, NUM_CLASSES>;

namespace cvo{
  
  typedef Eigen::Triplet<float> Trip_t;

  static bool is_logging = false;
  static bool debug_print = false;


  

  
  CvoGPU::CvoGPU(const std::string & param_file) {
    // read_CvoParams(param_file.c_str(), &params);
    read_CvoParams_yaml(param_file.c_str(), &params);
    printf("Some Cvo Params are: ell_init: %f, eps_2: %f\n", params.ell_init, params.eps_2);
    cudaMalloc((void**)&params_gpu, sizeof(CvoParams) );
    cudaMemcpy( (void*)params_gpu, &params, sizeof(CvoParams), cudaMemcpyHostToDevice  );

  }
  
  void CvoGPU::write_params(const CvoParams * p_cpu) {
    //params = *p_cpu;
    cudaMemcpy( (void*)params_gpu, p_cpu, sizeof(CvoParams), cudaMemcpyHostToDevice  );
    
  }


  CvoGPU::~CvoGPU() {
    cudaFree(params_gpu);
    
  }


  struct transform_point : public thrust::unary_function<CvoPoint,CvoPoint>
  {
    const Mat33f * R;
    const Vec3f * T;
    const bool update_normal_and_cov;

    transform_point(const Mat33f * R_gpu, const Vec3f * T_gpu,
                    bool update_normal_and_covariance): R(R_gpu), T(T_gpu),
                                                        update_normal_and_cov(update_normal_and_covariance){}
    
    __host__ __device__
    CvoPoint operator()(const CvoPoint & p_init)
    {
      CvoPoint result(p_init);


      Eigen::Vector3f input;
      input << p_init.x, p_init.y, p_init.z;

      Eigen::Vector3f trans = (*R) * input + (*T);
      result.x = trans(0);
      result.y = trans(1);
      result.z = trans(2);
      
      if (update_normal_and_cov) {
        Eigen::Vector3f input_normal;
        input_normal << p_init.normal[0], p_init.normal[1], p_init.normal[2];
        Eigen::Vector3f trans_normal = (*R) * input_normal + (*T);
        result.normal[0] = trans_normal(0);
        result.normal[1] = trans_normal(1);
        result.normal[2] = trans_normal(2);



        Eigen::Matrix3f point_cov;
        point_cov << p_init.covariance[0], p_init.covariance[1], p_init.covariance[2],
          p_init.covariance[3], p_init.covariance[4], p_init.covariance[5],
          p_init.covariance[6], p_init.covariance[7], p_init.covariance[8];
      
        point_cov = ((*R) * point_cov * (R->transpose())).eval();
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            result.covariance[i*3 + j] = point_cov(i,j);
          }
        }
        memcpy(result.cov_eigenvalues, p_init.cov_eigenvalues, sizeof(float)*3);
      }

      return result;

    }
  };


  void  transform_pointcloud_thrust(std::shared_ptr<CvoPointCloudGPU> init_cloud,
                                    std::shared_ptr<CvoPointCloudGPU> transformed_cloud,
                                    Mat33f * R_gpu, Vec3f * T_gpu,
                                    bool update_normal_and_cov
                                    ) {

    thrust::transform( init_cloud->begin(), init_cloud->end(),  transformed_cloud->begin(), 
                       transform_point(R_gpu, T_gpu, update_normal_and_cov));
    /*
    if (debug_print) {
      std::cout<<"transformed size is "<<transformed_cloud->size()<<std::endl;
      Mat33f R_cpu;
      cudaMemcpy(&R_cpu, R_gpu, sizeof(Mat33f), cudaMemcpyDeviceToHost);
      printf("Transform point : R_gpu is \n");
      std::cout<<R_cpu<<std::endl;
    
      CvoPoint from_point;
      CvoPoint transformed_point;
      for (int i = 1000; i < 1001; i++){
        cudaMemcpy((void*)&transformed_point, (void*)thrust::raw_pointer_cast( transformed_cloud->points.data() +i), sizeof(CvoPoint), cudaMemcpyDeviceToHost  );
        cudaMemcpy((void*)&from_point, (void*)thrust::raw_pointer_cast( init_cloud->points.data() +i), sizeof(CvoPoint), cudaMemcpyDeviceToHost  );

          printf("print %d point after transformation:\n", i);
          printf("from\n ");
          pcl::print_point(from_point);
          printf("to\n");
          pcl::print_point(transformed_point  );

      }     
      }*/
  
  }
  

  __device__
  float compute_range_ell(float curr_ell, float curr_dist_to_sensor, float min_dist, float max_dist ) {
    float final_ell = ((curr_dist_to_sensor) / 500.0 + 1.0)* curr_ell;
    return final_ell;
  }


  
  void update_tf(const Mat33f & R, const Vec3f & T,
                 // outputs
                 CvoState * cvo_state,
                 Eigen::Ref<Mat44f> transform
                 )  {
    // transform = [R', -R'*T; 0,0,0,1]
    Mat33f R_inv = R.transpose();
    Vec3f T_inv = -R_inv * T;
    
    transform.block<3,3>(0,0) = R_inv;
    transform.block<3,1>(0,3) = T_inv;
    transform.block<1,4>(3,0) << 0,0,0,1;


    cudaMemcpy(cvo_state->R_gpu->data(), R_inv.data(), sizeof(Eigen::Matrix3f), cudaMemcpyHostToDevice);
    cudaMemcpy(cvo_state->T_gpu->data(), T_inv.data(), sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice );
    //if (debug_print) std::cout<<"R,T is "<<R<<std::endl<<T<<std::endl;
    //if (debug_print) std::cout<<"transform mat R"<<transform.block<3,3>(0,0)<<"\nT: "<<transform.block<3,1>(0,3)<<std::endl;
  }

  void update_tf(const Mat33f & R, const Vec3f & T,
                 // outputs
                 CvoState * cvo_state
                 )  {
    // transform = [R', -R'*T; 0,0,0,1]
    Mat33f R_inv = R.transpose();
    Vec3f T_inv = -R_inv * T;
    
    cudaMemcpy(cvo_state->R_gpu->data(), R_inv.data(), sizeof(Eigen::Matrix3f), cudaMemcpyHostToDevice);
    cudaMemcpy(cvo_state->T_gpu->data(), T_inv.data(), sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice );
    //if (debug_print) std::cout<<"R,T is "<<R<<std::endl<<T<<std::endl;
    //if (debug_print) std::cout<<"transform mat R"<<transform.block<3,3>(0,0)<<"\nT: "<<transform.block<3,1>(0,3)<<std::endl;
  }
  

  /*
  void update_tf(const Mat33f & R, const Vec3f & T,
                 // outputs
                 CvoState * cvo_state,
                 Eigen::Ref<Eigen::Matrix<float,3,4>> transform
                 )  {
    // transform = [R', -R'*T; 0,0,0,1]
    Mat33f R_inv = R.transpose();
    Vec3f T_inv = -R_inv * T;
    
    transform.block<3,3>(0,0) = R_inv;
    transform.block<3,1>(0,3) = T_inv;

    cudaMemcpy(cvo_state->R_gpu->data(), R_inv.data(), sizeof(Eigen::Matrix3f), cudaMemcpyHostToDevice);
    cudaMemcpy(cvo_state->T_gpu->data(), T_inv.data(), sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice );
    //if (debug_print) std::cout<<"R,T is "<<R<<std::endl<<T<<std::endl;
    //if (debug_print) std::cout<<"transform mat R"<<transform.block<3,3>(0,0)<<"\nT: "<<transform.block<3,1>(0,3)<<std::endl;
  }
  */

  typedef KDTreeVectorOfVectorsAdaptor<cloud_t, float>  kd_tree_t;


  __device__
  float mahananobis_distance( const CvoPoint & p_a,const  CvoPoint &p_b,
                              const Eigen::Matrix3f & kernel_inv) {
    //const Eigen::Matrix3f cov_a = Eigen::Map<const Eigen::Matrix3f>(p_a.covariance);
    //const Eigen::Matrix3f cov_b = Eigen::Map<const Eigen::Matrix3f>(p_b.covariance);
    //Eigen::Matrix3f cov = cov_a + cov_b;

    Eigen::Vector3f a;
    a << p_a.x,p_a.y, p_a.z;
    Eigen::Vector3f b;
    b << p_b.x, p_b.y, p_b.z;

    Eigen::Vector3f dist = a-b;
    //float cov_inv[9];
    //Eigen::Matrix3f cov_curr_inv = Eigen::Map<Eigen::Matrix3f>(cov_inv);

    return (dist.transpose() * kernel_inv * dist).value();

    
  }
  __device__
  float eigenvalue_distance( const CvoPoint & p_a,const  CvoPoint &p_b, float ell_shrink_ratio) {
    //const Eigen::Matrix3f cov_a = Eigen::Map<const Eigen::Matrix3f>(p_a.covariance);
    //const Eigen::Matrix3f cov_b = Eigen::Map<const Eigen::Matrix3f>(p_b.covariance);
    auto e_values_a = p_a.cov_eigenvalues;
    auto e_values_b = p_a.cov_eigenvalues;

    Eigen::Vector3f a;
    a << p_a.x,p_a.y, p_a.z;
    Eigen::Vector3f b;
    b << p_b.x, p_b.y, p_b.z;

    Eigen::Vector3f dist = a-b;
    //float cov_inv[9];
    //Eigen::Matrix3f cov_curr_inv = Eigen::Map<Eigen::Matrix3f>(cov_inv);
    
    //Eigen::Matrix3f cov_curr_inv = Inverse(cov) ;
    float e_value_max_sum = e_values_a[2] + e_values_b[2];
    float e_value_min_sum = e_values_a[0] + e_values_b[0];

    float e_value = (e_value_max_sum )/2 * ell_shrink_ratio;

    if (e_value > 4.0) e_value = 1.0;
    if (e_value < 0.01) e_value = 0.01;

    return squared_dist(p_a, p_b) / e_value / e_value;
    //return (dist.transpose() * cov_curr_inv * dist).value();

    
  }

  __device__
  float compute_geometric_type_ip(const float * geo_type_a,
                                  const float * geo_type_b,
                                  int size
                                  ) {
    float norm2_a = square_norm(geo_type_a, size);
    float norm2_b = square_norm(geo_type_b, size);
    float dot_ab = dot(geo_type_a, geo_type_b, size);
    //printf("norm2_a=%f, norm2_b=%f, dot_ab=%f\n", norm2_a, norm2_b, dot_ab);
    float geo_sim = dot_ab * dot_ab / (norm2_a * norm2_b);
    
    return geo_sim;
  }

  __global__
  void fill_in_A_mat_gpu_dense_mat_kernel(// input
                                          const CvoParams * cvo_params,
                                          CvoPoint * points_a,
                                          int a_size,
                                          CvoPoint * points_b,
                                          int b_size,
                                          int num_neighbors,
                                          Eigen::Matrix3f * kernel_inv,
                                          // output
                                          SparseKernelMat * A_mat // the inner product matrix!
                                          ) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > a_size - 1)
      return;

    // params
    float sigma_square= cvo_params->sigma * cvo_params->sigma;
    float c_ell_square = cvo_params->c_ell*cvo_params->c_ell;
    float s_ell_square = cvo_params->s_ell*cvo_params->s_ell;
    float sp_thres = cvo_params->sp_thres;
    float c_sigma_square = cvo_params->c_sigma*cvo_params->c_sigma;
    float s_sigma_square = cvo_params->s_sigma*cvo_params->s_sigma;

    // point a in the first point cloud
    CvoPoint * p_a =  &points_a[i];

    //A_mat->max_index[i] = -1;
    //float curr_max_ip = cvo_params->sp_thres;
    
    float d2_thres=1, d2_c_thres=1, d2_s_thres=1;
    //if (cvo_params->is_using_geometry)
    //  d2_thres = -2.0*l*l*log(cvo_params->sp_thres/sigma2);
    if (cvo_params->is_using_intensity)
      d2_c_thres = -2.0*c_ell_square*log(sp_thres/c_sigma_square);
    if (cvo_params->is_using_semantics)
      d2_s_thres = -2.0*s_ell_square*log(sp_thres/s_sigma_square);
    
    float * label_a = p_a ->label_distribution;

    unsigned int num_inds = 0;

    for (int j = 0; j < b_size ; j++) {
      int ind_b = j;
      if (num_inds == num_neighbors) break;
      CvoPoint * p_b = &points_b[ind_b];


      //float ell_shrink_ratio = ell_curr / cvo_params->ell_init;
      
      //float d2 = eigenvalue_distance(*p_a, *p_b, ell_shrink_ratio)  ;
      //if (i == 0 && j == 0) {
      //  float d2_l2 = squared_dist(*p_a, *p_b) / 0.25;
      //  printf("i==%d, ell is %f, mahananobis_distance is %f, while d2_l2 is %f\n", i,ell_curr, d2, d2_l2);
      //}

      //float d2_iso =  squared_dist(*p_a, *p_b) / ell_curr / ell_curr;
      //float d2_iso =  squared_dist(*p_a, *p_b) / cvo_params->ell_init / cvo_params->ell_init;
      //d2 = d2 < d2_iso ? d2 : d2_iso;

      // float l = ell_curr;
      float a = 1, sk=1, ck=1, k=1, geo_sim = 1;

      if (cvo_params->is_using_geometric_type) {
        geo_sim = compute_geometric_type_ip(p_a->geometric_type,
                                            p_b->geometric_type,
                                            2
                                            );
        if(geo_sim < 0.01)
          continue;        
      }
      
      if (cvo_params->is_using_geometry) {
        //if (i == 0)
        //  printf("computing mahanobis_dist");
        float d2 = mahananobis_distance(*p_a, *p_b, *kernel_inv);        
        //k= cvo_params->sigma * cvo_params->sigma*exp(-d2/(2.0*l*l));
        k = sigma_square * exp(-d2 / 2.0);
      }
      //if (i==0)
      //  printf("i==%d, geometric: k is %f\n", i, k); 
      if (cvo_params->is_using_intensity) {
        float d2_color = squared_dist<float>(p_a->features, p_b->features, FEATURE_DIMENSIONS);
        if (d2_color < d2_c_thres)
          ck = c_sigma_square* exp(-d2_color/(2.0*c_ell_square ));
        else
          continue;
      }
      if (cvo_params->is_using_semantics) {
        float d2_semantic = squared_dist<float>(p_a->label_distribution, p_b->label_distribution, NUM_CLASSES);
        if (d2_semantic < d2_s_thres)
          sk = s_sigma_square * exp(-d2_semantic/(2.0*s_ell_square));
        else
          continue;
      }
      a = ck*k*sk*geo_sim;
      if (a > cvo_params->sp_thres){
        A_mat->mat[i * num_neighbors + num_inds] = a;
        A_mat->ind_row2col[i * num_neighbors + num_inds] = ind_b;
        
        num_inds++;
        //if (i == 4058)
        //  printf("i==%d, j==%d, num_inds of %d, a is %f\n", i,ind_b, num_inds, a );        
      }


    }
    //printf("max row num is %d, num_inds of %d is %d\n", a_size, i, num_inds );
    A_mat->nonzeros[i] = num_inds;
  }

  __global__
  void fill_in_A_mat_cukdtree(const CvoParams * cvo_params,
                              CvoPoint * points_a, int a_size,
                              CvoPoint * points_b, int size_y,
                              int * kdtree_inds_results,
                              int num_neighbors,                              
                              float ell,
                              // output
                              SparseKernelMat * A_mat
                              ) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("\ni is %d\n\n", i);
    if (i > a_size - 1)
      return;
    
    //A_mat->max_index[i] = -1;
    float curr_max_ip = cvo_params->sp_thres;

    float sigma2= cvo_params->sigma * cvo_params->sigma;
    float c2 = cvo_params->c_ell * cvo_params->c_ell;
    float c_sigma2 = cvo_params->c_sigma * cvo_params->c_sigma;
    float s_ell = cvo_params->s_ell;
    float s_sigma2 = cvo_params->s_sigma * cvo_params->s_sigma;

    CvoPoint * p_a =  &points_a[i];

    float a_to_sensor = sqrtf(p_a->x * p_a->x + p_a->y * p_a->y + p_a->z * p_a->z);
    float l = compute_range_ell(ell, a_to_sensor , 1, 80 );

    float d2_thres=1, d2_c_thres=1, d2_s_thres=1;
    if (cvo_params->is_using_geometry)
      d2_thres = -2.0*l*l*log(cvo_params->sp_thres/sigma2);
    if (cvo_params->is_using_intensity)
      d2_c_thres = -2.0*c2*log(cvo_params->sp_thres/c_sigma2);
    if (cvo_params->is_using_semantics)
      d2_s_thres = -2.0*s_ell*s_ell*log(cvo_params->sp_thres/s_sigma2 );
    

    float * label_a = nullptr;
    if (cvo_params->is_using_semantics)
      label_a = p_a ->label_distribution;

    unsigned int num_inds = 0;
    for (int j = 0; j < num_neighbors ; j++) {
      int ind_b = kdtree_inds_results[j + i * num_neighbors];
      CvoPoint * p_b = &points_b[ind_b];
      //if (i <= 1)
      //  printf("p_a is (%f, %f, %f), p_b is (%f, %f, %f\n)", p_a->x,p_a->y, p_a->z, p_b->x, p_b->y, p_b->z);

      float a = 1, sk=1, ck=1, k=1, geo_sim=1;
      if (cvo_params->is_using_geometric_type) {
        geo_sim = compute_geometric_type_ip(p_a->geometric_type,
                                            p_b->geometric_type,
                                            2
                                            );


        if(geo_sim < 0.001)
          continue;
      }
      
      if (cvo_params->is_using_geometry) {
        float d2 = (squared_dist( *p_b ,*p_a ));
        if (d2 < d2_thres)
          k= sigma2*exp(-d2/(2.0*l*l));
        else continue;
      }
      //}
      if (cvo_params->is_using_intensity) {
        float d2_color = squared_dist<float>(p_a->features, p_b->features, FEATURE_DIMENSIONS);
        if (d2_color < d2_c_thres)
          ck = c_sigma2*exp(-d2_color/(2.0*c2 ));
        else
          continue;
      }
      if (cvo_params->is_using_semantics) {
        float d2_semantic = squared_dist<float>(p_a->label_distribution, p_b->label_distribution, NUM_CLASSES);
        if (d2_semantic < d2_s_thres )
          sk = s_sigma2*exp(-d2_semantic/(2.0*s_ell*s_ell));
        else
          continue;
      }
      a = ck*k*sk*geo_sim;      
      //if (i==0) 
      //   printf("point_a is (%f,%f,%f), point_b is (%f,%f,%f), k=%f,ck=%f, sk=%f, a=%f, \n", p_a->x, p_a->y, p_a->z, p_b->x, p_b->y, p_b->z, k, ck, sk,a );

      if (a > cvo_params->sp_thres){
        A_mat->mat[i * num_neighbors + num_inds] = a;
        A_mat->ind_row2col[i * num_neighbors + num_inds] = ind_b;
        //if (a > curr_max_ip) {
        //  curr_max_ip = a;
        //  A_mat->max_index[i] = ind_b;
        //}
        
        num_inds++;
      }

    }
    A_mat->nonzeros[i] = num_inds;

  }

  void find_nearby_source_points_cukdtree(//const CvoParams *cvo_params,
                                          std::shared_ptr<CvoPointCloudGPU> cloud_x_gpu,
                                          perl_registration::cuKdTree<CvoPoint> & kdtree,
                                          const Eigen::Matrix4f & transform_cpu_tf2sf,
                                          int num_neighbors,
                                          // output
                                          std::shared_ptr<CvoPointCloudGPU> cloud_x_gpu_transformed_kdtree,
                                          thrust::device_vector<int> & cukdtree_inds_results
                                          ) {
    cudaEvent_t cuda_start, cuda_stop;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);
    cudaEventRecord(cuda_start, 0);

    Eigen::Matrix3f * R_gpu_t2s;
    Eigen::Vector3f * T_gpu_t2s;
    cudaMalloc(&R_gpu_t2s, sizeof(Eigen::Matrix3f));
    cudaMalloc(&T_gpu_t2s, sizeof(Eigen::Vector3f));
    Eigen::Matrix3f R_cpu_t2s = transform_cpu_tf2sf.block<3,3>(0,0);
    Eigen::Vector3f T_cpu_t2s = transform_cpu_tf2sf.block<3,1>(0,3);
    cudaMemcpy(R_gpu_t2s, &R_cpu_t2s, sizeof(decltype(R_cpu_t2s)), cudaMemcpyHostToDevice);
    cudaMemcpy(T_gpu_t2s, &T_cpu_t2s, sizeof(decltype(T_cpu_t2s)), cudaMemcpyHostToDevice);
    
    transform_pointcloud_thrust(cloud_x_gpu, cloud_x_gpu_transformed_kdtree,
                                R_gpu_t2s, T_gpu_t2s, false);
    
    kdtree.NearestKSearch(cloud_x_gpu_transformed_kdtree, num_neighbors , cukdtree_inds_results);

    cudaFree(R_gpu_t2s);
    cudaFree(T_gpu_t2s);

    cudaEventRecord(cuda_stop, 0);
    cudaEventSynchronize(cuda_stop);
    float elapsedTime, totalTime;
    
    cudaEventElapsedTime(&elapsedTime, cuda_start, cuda_stop);
    cudaEventDestroy(cuda_start);
    cudaEventDestroy(cuda_stop);
    totalTime = elapsedTime/(1000);
    if (debug_print)std::cout<<"Kdtree query time is "<<totalTime<<std::endl;
    
  }
  
  

  __global__
  void fill_in_A_mat_gpu(const CvoParams * cvo_params,
                         CvoPoint * points_a,
                         int a_size,
                         CvoPoint * points_b,
                         int b_size,
                         int num_neighbors,
                         float ell,
                         // output
                         SparseKernelMat * A_mat // the kernel matrix!
                         ) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("\ni is %d\n\n", i);
    if (i > a_size - 1)
      return;
    
    //A_mat->max_index[i] = -1;
    float curr_max_ip = cvo_params->sp_thres;

    float sigma2= cvo_params->sigma * cvo_params->sigma;
    float c2 = cvo_params->c_ell * cvo_params->c_ell;
    float c_sigma2 = cvo_params->c_sigma * cvo_params->c_sigma;
    float s_ell = cvo_params->s_ell;
    float s_sigma2 = cvo_params->s_sigma * cvo_params->s_sigma;

    CvoPoint * p_a =  &points_a[i];

    float a_to_sensor = sqrtf(p_a->x * p_a->x + p_a->y * p_a->y + p_a->z * p_a->z);
    float l = compute_range_ell(ell, a_to_sensor , 1, 80 );

    float d2_thres=1, d2_c_thres=1, d2_s_thres=1;
    if (cvo_params->is_using_geometry)
      d2_thres = -2.0*l*l*log(cvo_params->sp_thres/sigma2);
    if (cvo_params->is_using_intensity)
      d2_c_thres = -2.0*c2*log(cvo_params->sp_thres/c_sigma2);
    if (cvo_params->is_using_semantics)
      d2_s_thres = -2.0*s_ell*s_ell*log(cvo_params->sp_thres/s_sigma2 );
    

    float * label_a = nullptr;
    if (cvo_params->is_using_semantics)
      label_a = p_a ->label_distribution;

    unsigned int num_inds = 0;
    for (int j = 0; j < b_size ; j++) {
      int ind_b = j;
      if (num_inds == num_neighbors) break;
      CvoPoint * p_b = &points_b[ind_b];
      //if (j == 0)
      //  printf("p_a is (%f, %f, %f), p_b is (%f, %f, %f\n)", p_a->x,p_a->y, p_a->z, p_b->x, p_b->y, p_b->z);

      float a = 1, sk=1, ck=1, k=1, geo_sim=1;
      if (cvo_params->is_using_geometric_type) {
        geo_sim = compute_geometric_type_ip(p_a->geometric_type,
                                            p_b->geometric_type,
                                            2
                                            );

        //if (i == 0 )
        //  printf("p_a is (%f, %f, %f), p_b is (%f, %f, %f), geo_sim is %f\n", p_a->x,p_a->y, p_a->z, p_b->x, p_b->y, p_b->z, geo_sim);
        
        if(geo_sim < 0.01)
          continue;        
      }
      
      if (cvo_params->is_using_geometry) {
        float d2 = (squared_dist( *p_b ,*p_a ));
        if (d2 < d2_thres)
          k= sigma2*exp(-d2/(2.0*l*l));
        else continue;
      }
      //}
      if (cvo_params->is_using_intensity) {
        float d2_color = squared_dist<float>(p_a->features, p_b->features, FEATURE_DIMENSIONS);
        if (d2_color < d2_c_thres)
          ck = c_sigma2*exp(-d2_color/(2.0*c2 ));
        else
          continue;
      }
      if (cvo_params->is_using_semantics) {
        float d2_semantic = squared_dist<float>(p_a->label_distribution, p_b->label_distribution, NUM_CLASSES);
        if (d2_semantic < d2_s_thres )
          sk = cvo_params->s_sigma*cvo_params->s_sigma*exp(-d2_semantic/(2.0*s_ell*s_ell));
        else
          continue;
      }
      a = ck*k*sk*geo_sim;
      if (a > cvo_params->sp_thres){
        A_mat->mat[i * num_neighbors + num_inds] = a;
        A_mat->ind_row2col[i * num_neighbors + num_inds] = ind_b;
        //if (i==10) 
        //  printf("point_a is (%f,%f,%f), point_b with index %d is (%f,%f,%f), k=%f,ck=%f, sk=%f \n", p_a->x, p_a->y, p_a->z, ind_b,  p_b->x, p_b->y, p_b->z,  k, ck, sk );
        
        //if (a > curr_max_ip) {
        //  curr_max_ip = a;
        //  A_mat->max_index[i] = ind_b;
        //}
        
        num_inds++;
      }

    }
    A_mat->nonzeros[i] = num_inds;
  }

  static
  void se_kernel_kdtree(//input
                        const CvoParams & params_cpu, const CvoParams * params_gpu,
                        std::shared_ptr<CvoPointCloudGPU> points_fixed,
                        std::shared_ptr<CvoPointCloudGPU> points_moving,
                        float ell,
                        perl_registration::cuKdTree<CvoPoint> & kdtree,
                        const Eigen::Matrix4f & transform_cpu_tf2sf,
                        int num_neighbors,
                        // output
                        std::shared_ptr<CvoPointCloudGPU> cloud_x_gpu_transformed_kdtree,
                        thrust::device_vector<int> & cukdtree_inds_results,
                        SparseKernelMat * A_mat,
                        SparseKernelMat * A_mat_gpu
                        ) {

    find_nearby_source_points_cukdtree(points_fixed, kdtree, transform_cpu_tf2sf, num_neighbors,
                                       cloud_x_gpu_transformed_kdtree, cukdtree_inds_results);

    /*
    std::cout<<"1st point neighbors are:\n";
    int inds[64];
    cudaMemcpy(inds, thrust::raw_pointer_cast(cukdtree_inds_results.data()), sizeof(int)*64, cudaMemcpyDeviceToHost);
    std::cout<<"neighbors of pt_fixed[0]: ";
    for (int i = 0; i < 32; i++) {
      std::cout<<inds[i]<<" ";
    }
    
    std::cout<<"neighbors of pt_fixed[1]: ";
    for (int i = 32; i < 64; i++) {
      std::cout<<inds[i]<<" ";
    }
    */
    
    fill_in_A_mat_cukdtree<<< (points_fixed->points.size() / CUDA_BLOCK_SIZE)+1, CUDA_BLOCK_SIZE  >>>
      (params_gpu, thrust::raw_pointer_cast(points_fixed->points.data()), points_fixed->size(),
       thrust::raw_pointer_cast(points_moving->points.data()), points_moving->size(),
       thrust::raw_pointer_cast(cukdtree_inds_results.data()),
       num_neighbors, ell,
       A_mat_gpu
       );
    cudaDeviceSynchronize();    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { 
        fprintf(stderr, "Failed to run fill_in_A_mat_cukdtree %s .\n", cudaGetErrorString(err)); 
        exit(EXIT_FAILURE); 
    }

    compute_nonzeros(A_mat);
    
  }
  

  static
  void se_kernel(// input
                 const CvoParams * params_gpu,
                 std::shared_ptr<CvoPointCloudGPU> points_fixed,
                 std::shared_ptr<CvoPointCloudGPU> points_moving,
                 int num_neighbors,
                 float ell,
                 // output
                 SparseKernelMat * A_mat,
                 SparseKernelMat * A_mat_gpu
                 )  {

    auto start = chrono::system_clock::now();
    int fixed_size = points_fixed->points.size();
    CvoPoint * points_fixed_raw = thrust::raw_pointer_cast (  points_fixed->points.data() );
    CvoPoint * points_moving_raw = thrust::raw_pointer_cast( points_moving->points.data() );

    fill_in_A_mat_gpu<<<(points_fixed->points.size() / CUDA_BLOCK_SIZE)+1, CUDA_BLOCK_SIZE  >>>(params_gpu,
                                                                                                points_fixed_raw,
                                                                                                fixed_size,
                                                                                                points_moving_raw,
                                                                                                points_moving->points.size(),
                                                                                                num_neighbors,
                                                                                                ell,
                                                                                                // output
                                                                                                A_mat_gpu // the kernel mat
                                                                                                );
    cudaDeviceSynchronize();    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { 
        fprintf(stderr, "Failed to run fill_in_A_mat_gpu %s .\n", cudaGetErrorString(err)); 
        exit(EXIT_FAILURE); 
    }

    compute_nonzeros(A_mat);
  }

  static
  void se_kernel_dense(// input
                       const CvoParams * params_gpu,
                       std::shared_ptr<CvoPointCloudGPU> points_fixed,
                       std::shared_ptr<CvoPointCloudGPU> points_moving,
                       int num_neighbors,
                       const Eigen::Matrix3f & kernel_inv_cpu,
                       // output
                       SparseKernelMat * A_mat,
                       SparseKernelMat * A_mat_gpu
                       )  {
    
    auto start = chrono::system_clock::now();
    int fixed_size = points_fixed->points.size();
    CvoPoint * points_fixed_raw = thrust::raw_pointer_cast (  points_fixed->points.data() );
    CvoPoint * points_moving_raw = thrust::raw_pointer_cast( points_moving->points.data() );

    Eigen::Matrix3f * kernel_inv_gpu;
    cudaMalloc((void**)&kernel_inv_gpu, sizeof(Eigen::Matrix3f));
    cudaMemcpy(kernel_inv_gpu, &kernel_inv_cpu, sizeof(decltype(kernel_inv_cpu)), cudaMemcpyHostToDevice);
    //std::cout<<"Start fill_in_A_mat_gpu_dense_mat_kernel\n"<<std::endl;
    fill_in_A_mat_gpu_dense_mat_kernel<<<(points_fixed->points.size() / CUDA_BLOCK_SIZE)+1, CUDA_BLOCK_SIZE  >>>(params_gpu,
                                                                                                                 points_fixed_raw,
                                                                                                                 fixed_size,
                                                                                                                 points_moving_raw,
                                                                                                                 points_moving->points.size(),
                                                                                                                 num_neighbors,
                                                                                                                 kernel_inv_gpu,
                                                                                                                 // output
                                                                                                                 A_mat_gpu // the kernel mat
                                                                                                                 );
    cudaDeviceSynchronize();    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { 
        fprintf(stderr, "Failed to run fill_in_A_mat_gpu_dense_mat_kernel %s .\n", cudaGetErrorString(err)); 
        exit(EXIT_FAILURE); 
    }
    
    compute_nonzeros(A_mat);
    cudaFree(kernel_inv_gpu);

  }
  

  __global__ void compute_flow_gpu_no_eigen(const CvoParams * cvo_params,
                                            CvoPoint * cloud_x, CvoPoint * cloud_y,
                                            SparseKernelMat * A,
                                            int num_neighbors,
                                            // outputs: thrust vectors
                                            Eigen::Vector3d * omega_all_gpu,
                                            Eigen::Vector3d * v_all_gpu
                                            ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > A->rows - 1)
      return;
    int A_rows = A->rows;
    int A_cols = A->cols;
    //float * Ai = A->mat + i * A->cols;
    float * Ai = A->mat + i * num_neighbors;
    
    CvoPoint * px = &cloud_x[i];
    Eigen::Vector3f px_eig;
    px_eig<< px->x , px->y, px->z;
    float px_arr[3] = {px->x, px->y, px->z};

    Eigen::Vector3f omega_i = Eigen::Vector3f::Zero();
    Eigen::Vector3f v_i = Eigen::Vector3f::Zero();
    //#ifdef IS_USING_COVARIANCE
    //Eigen::Matrix3f cov_i = Eigen::Map<Eigen::Matrix3f>(px->covariance);
    //#endif    
    float dl_i = 0;
    for (int j = 0; j < num_neighbors; j++) {
      int idx = A->ind_row2col[i*num_neighbors+j];
      if (idx == -1) break;
      
      CvoPoint * py = &cloud_y[idx];
      float py_arr[3] = {py->x, py->y, py->z};
      Eigen::Vector3f py_eig;
      py_eig << py->x, py->y, py->z;
      
      Eigen::Vector3f cross_xy_j = px_eig.cross(py_eig) ;
      Eigen::Vector3f diff_yx_j = py_eig - px_eig;
      float sum_diff_yx_2_j = diff_yx_j.squaredNorm();
      /*
        #ifdef IS_USING_COVARIANCE      
        Eigen::Matrix3f cov_j = Eigen::Map<Eigen::Matrix3f>(py->covariance);
        omega_i = omega_i + cov_j * cross_xy_j *  *(Ai + j );
        v_i = v_i + cov_j * diff_yx_j *  *(Ai + j);
        //float eigenvalue_sum = px->cov_eigenvalues(0)

        //omega_i = omega_i +  cross_xy_j *  *(Ai + j );
        // v_i = v_i + diff_yx_j *  *(Ai + j);      
      
        #else   
      */   
      omega_i = omega_i + cross_xy_j *  *(Ai + j );
      v_i = v_i + diff_yx_j *  *(Ai + j);

    }

    Eigen::Vector3d & omega_i_eig = omega_all_gpu[i];
    omega_i_eig = (omega_i / cvo_params->c ).cast<double>();
    Eigen::Vector3d & v_i_eig = v_all_gpu[i];
    v_i_eig = (v_i /  cvo_params->d).cast<double>();

  }


  void compute_flow(CvoState * cvo_state, const CvoParams * params_gpu,
                    Eigen::Vector3f * omega, Eigen::Vector3f * v, int num_neighbors)  {

    if (debug_print ) {
      std::cout<<"nonzeros in A "<<nonzeros(&cvo_state->A_host)<<std::endl;
      std::cout<<"A rows is "<<cvo_state->A_host.rows<<", A cols is "<<cvo_state->A_host.cols<<std::endl;
    }
    auto start = chrono::system_clock::now();
    compute_flow_gpu_no_eigen<<<cvo_state->A_host.rows / CUDA_BLOCK_SIZE + 1 ,CUDA_BLOCK_SIZE>>>(params_gpu,
                                                                                                 thrust::raw_pointer_cast(cvo_state->cloud_x_gpu->points.data()   ),
                                                                                                 thrust::raw_pointer_cast(cvo_state->cloud_y_gpu->points.data()   ),
                                                                                                 cvo_state->A,
                                                                                                 num_neighbors,
                                                                                                 thrust::raw_pointer_cast(cvo_state->omega_gpu.data()  ),
                                                                                                 thrust::raw_pointer_cast(cvo_state->v_gpu.data() ));
    ;
    //cudaDeviceSynchronize();
    if (debug_print) {
      printf("finish compute_flow_gpu_no_eigen\n");

    }
    
    auto end = chrono::system_clock::now();
    //std::cout<<"time for compute_gradient is "<<std::chrono::duration_cast<std::chrono::milliseconds>((end- start)).count()<<std::endl;
    
    start = chrono::system_clock::now();
    // update them to class-wide variables
    //thrust::plus<double> plus_double;
    //thrust::plus<Eigen::Vector3d> plus_vector;
    
    //(thrust::reduce(thrust::device, cvo_state->omega_gpu.begin(), cvo_state->omega_gpu.end(), Eigen::Vector3d(0,0,0)  ) );
    *omega = (thrust::reduce(thrust::device, cvo_state->omega_gpu.begin(), cvo_state->omega_gpu.end(), Eigen::Vector3d(0,0,0) )).cast<float>();
    *v = (thrust::reduce(thrust::device, cvo_state->v_gpu.begin(), cvo_state->v_gpu.end(), Eigen::Vector3d(0,0,0))).cast<float>();
    // normalize the gradient
    Eigen::Matrix<float, 6, 1> ov;
    ov.segment<3>(0) = *omega;
    ov.segment<3>(3) = *v;
    ov.normalize();
    *omega = ov.segment<3>(0);
    *v = ov.segment<3>(3);
    //omega->normalize();
    //v->normalize();

    // Eigen::Vector3d::Zero(), plus_vector)).cast<float>();
    cudaMemcpy(cvo_state->omega, omega, sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice );
    cudaMemcpy(cvo_state->v, v, sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice );

    
    end = chrono::system_clock::now();
    //std::cout<<"time for thrust_reduce is "<<std::chrono::duration_cast<std::chrono::milliseconds>((end- start)).count()<<std::endl;
    start = chrono::system_clock::now();
    int A_nonzero = nonzeros(&cvo_state->A_host);
    if (debug_print) std::cout<<"compute flow result: omega "<<omega->transpose()<<", v: "<<v->transpose()<<std::endl;
    end = chrono::system_clock::now();
    //std::cout<<"time for nonzeros "<<std::chrono::duration_cast<std::chrono::milliseconds>((end- start)).count()<<std::endl;
  }


  __global__ void fill_in_residual_and_jacobian(float ell,
                                                CvoPoint * cloud_x, CvoPoint * cloud_y,
                                                SparseKernelMat * A,
                                                // output
                                                Eigen::Matrix<float, 6,6> * ls_lhs,
                                                Eigen::Matrix<float, 6,1> * ls_rhs
                                                ) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > A->rows - 1)
      return;

    ls_lhs[i] = Eigen::Matrix<float, 6,6>::Zero();
    ls_rhs[i] = Eigen::Matrix<float, 6,1>::Zero();
    Eigen::Vector3f x(cloud_x[i].x, cloud_x[i].y, cloud_x[i].z);

    int cloud_x_size = A->rows;
    int cloud_y_size = A->cols;
    
    //if (A->max_index[i] == -1) {
    //  return;
    //}

    for (int j = 0; j < A->cols; j++) {
      //int id_y_closest = A->max_index[i];
      int id_y = A->ind_row2col[i*A->cols+j];
      if (id_y == -1) break;
      float w_ij = A->mat[i*A->cols+j];

      //int id_y = A->max_index[i];
      //if (id_y == -1) return;
      //float w_ij = 1;    


      Eigen::Vector3f y(cloud_y[id_y].x, cloud_y[id_y].y, cloud_y[id_y].z);

      Eigen::Vector3f r_ij = (x-y) / ell;

      float dist = sqrtf(r_ij(0)*r_ij(0) + r_ij(1) *r_ij(1) + r_ij(2)*r_ij(2) ) * ell ;
      if (dist > 0.2) return;

    
      Eigen::Matrix<float, 3, 6> J_i;

      Eigen::Matrix3f skew_y = Eigen::Matrix3f::Zero();
      skew_gpu( &y, &skew_y);
      J_i.block<3,3>(0,0) = -1 * skew_y;
      J_i.block<3,3>(0,3) = Eigen::Matrix3f::Identity().eval();
      J_i = (J_i / ell).eval();

      //Eigen::Matrix<float, 3,3> weight =
      ls_lhs[i] += J_i.transpose() * J_i * w_ij;
      ls_rhs[i] += J_i.transpose() * r_ij * w_ij;
    }
    if (i < 100)  {

      //printf("fill-in-least-square: i=%d, j=%d, x=(%f,%f,%f), y=(%f,%f,%f), |x-y| is %f, J_i(3,3) is \n%f, %f, %f\n %f, %f, %f\n %f,%f,%f\n", i,x(0),x(1),x(2), y(0),y(1),y(2),  dist , J_i(0,0), J_i(0,1), J_i(0,2), J_i(1,0), J_i(1,1), J_i(1,2), J_i(2,0), J_i(2,1), J_i(2,2)  );
      
    }
    
  }

  void compute_flow_least_square(CvoState * cvo_state, const CvoParams * params_gpu,
                                 Eigen::Vector3f * omega, Eigen::Vector3f * v) {

    fill_in_residual_and_jacobian<<< cvo_state->A_host.rows / CUDA_BLOCK_SIZE + 1, CUDA_BLOCK_SIZE >>> (cvo_state->ell,
                                                                                                        thrust::raw_pointer_cast(cvo_state->cloud_x_gpu->points.data()),
                                                                                                        thrust::raw_pointer_cast(cvo_state->cloud_y_gpu->points.data()),
                                                                                                        cvo_state->A,

                                                                                                        thrust::raw_pointer_cast(cvo_state->least_square_LHS.data()),
                                                                                                        thrust::raw_pointer_cast(cvo_state->least_square_RHS.data())
                                                                                                        );

    thrust::plus<Eigen::Matrix<float,6,6>> plus_mat66;

    Eigen::Matrix<float, 6,6> zero_66;
    zero_66 = Eigen::Matrix<float, 6,6>::Zero().eval();
    auto ls_lhs = thrust::reduce(cvo_state->least_square_LHS.begin(),
                                 cvo_state->least_square_LHS.end(),
                                 zero_66
                                 );

    Eigen::Matrix<float, 6,1> zero_61;
    zero_61 = Eigen::Matrix<float, 6,1>::Zero().eval();
    Eigen::Matrix<float, 6,1> ls_rhs = thrust::reduce(cvo_state->least_square_RHS.begin(),
                                                      cvo_state->least_square_RHS.end(),
                                                      zero_61
                                                      );
    Eigen::Matrix<float, 6,1> epsilon = - ls_lhs.inverse() * ls_rhs;
    
    *omega = epsilon.block<3,1>(0,0);
    *v = epsilon.block<3,1>(3,0);

    if (debug_print)
      std::cout<<"using least square, omega is "<<omega->transpose()
               <<", v is "<<v->transpose()
               <<"\n ls_lhs is \n"<<ls_lhs
               <<"\n ls_rhs is \n"<<ls_rhs
               <<std::endl;
  }
  
  __global__ void compute_step_size_xi(Eigen::Vector3f * omega ,
                                       Eigen::Vector3f * v,
                                       CvoPoint * cloud_y,
                                       int num_moving,
                                       int num_neighbors,
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
    skew_gpu(omega, &omega_hat);
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
    /*
      if ( j == 1000) {
      printf("j==1000, cloud+yi is (%f,%f,%f), xiz=(%f %f %f), xi2z=(%f %f %f), xi3z=(%f %f %f), xi4z=(%f %f %f), normxiz2=%f, xiz_dot_xi2z=%f, epsil_const=%f\n ",
      cloud_y[j].x , cloud_y[j].y, cloud_y[j].z,
      xiz[j](0), xiz[j](1), xiz[j](2),
      xi2z[j](0),xi2z[j](1),xi2z[j](2),
      xi3z[j](0),xi3z[j](1),xi3z[j](2),
      xi4z[j](0),xi4z[j](1),xi4z[j](2),
      normxiz2[j], xiz_dot_xi2z[j], epsil_const[j]
      );
      
      }*/

    
  }


  __global__ void compute_step_size_poly_coeff(float ell,
                                               float ell_init,
                                               int is_using_location_dependent_ell,
                                               int num_moving,
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
                                               int num_neighbors,
                                               // output
                                               double * B,
                                               double * C,
                                               double * D,
                                               double * E
                                               ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > A->rows-1 )
      return;
    int A_cols = A->cols;
    B[i] = 0;
    C[i] = 0;
    D[i] = 0;
    E[i] = 0;
    Eigen::Vector3f px;
    px << cloud_x[i].x, cloud_x[i].y, cloud_x[i].z;
    B[i] = C[i] = D[i] = E[i] = 0.0;

    float d2_sqrt = px.norm();
    float temp_ell = ell; // init value
    if (is_using_location_dependent_ell)
      temp_ell = compute_range_ell(ell,d2_sqrt, 1, 80 );

    
    for (int j = 0; j < num_neighbors; j++) {
      int idx = A->ind_row2col[i * num_neighbors + j];
      if (idx == -1) break;
      /*
        #ifdef IS_USING_COVARIANCE
        temp_ell = (cloud_x[i].cov_eigenvalues[2] + cloud_y[idx].cov_eigenvalues[2] + cloud_x[i].cov_eigenvalues[0] + cloud_y[idx].cov_eigenvalues[0])/4.0 ;
        //temp_ell = ( cloud_x[i].cov_eigenvalues[2] + cloud_y[idx].cov_eigenvalues[2])/2.0 ;
        temp_ell *= (ell/ell_init);
        if (temp_ell > 1.0) temp_ell = 1.0;
        if (temp_ell < 0.01) temp_ell =0.01;
        #endif  */    
      float temp_coef = 1/(2.0*temp_ell*temp_ell);   // 1/(2*l^2)       
      
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

      float A_ij = A->mat[i * num_neighbors + j];
      double bi = double(A_ij * beta_ij);
      B[i] += bi;
      double ci = double(A_ij * (gamma_ij+beta_ij*beta_ij/2.0));
      C[i] += ci;
      double di = double(A_ij * (delta_ij+beta_ij*gamma_ij + beta_ij*beta_ij*beta_ij/6.0));
      D[i] += di;
      double ei = double(A_ij * (epsil_ij+beta_ij*delta_ij+1/2.0*beta_ij*beta_ij*gamma_ij\
                                 + 1/2.0*gamma_ij*gamma_ij + 1/24.0*beta_ij*beta_ij*beta_ij*beta_ij));
      E[i] += ei;

    }
    
  }

  void compute_step_size(CvoState * cvo_state, const CvoParams * params, int num_neighbors) {
    compute_step_size_xi<<<cvo_state->num_moving / CUDA_BLOCK_SIZE + 1, CUDA_BLOCK_SIZE>>>
      (cvo_state->omega, cvo_state->v,
       thrust::raw_pointer_cast( cvo_state->cloud_y_gpu->points.data()  ), cvo_state->num_moving, num_neighbors,
       thrust::raw_pointer_cast(cvo_state->xiz.data()),
       thrust::raw_pointer_cast(cvo_state->xi2z.data()),
       thrust::raw_pointer_cast(cvo_state->xi3z.data()),
       thrust::raw_pointer_cast(cvo_state->xi4z.data()),
       thrust::raw_pointer_cast(cvo_state->normxiz2.data()),
       thrust::raw_pointer_cast(cvo_state->xiz_dot_xi2z.data()),
       thrust::raw_pointer_cast(cvo_state->epsil_const.data())
       );

    //float temp_coef = 1/(2.0*cvo_state->ell*cvo_state->ell);   // 1/(2*l^2)
    //compute_step_size_poly_coeff_range_ell<<<cvo_state->num_fixed / CUDA_BLOCK_SIZE + 1, CUDA_BLOCK_SIZE>>>
    compute_step_size_poly_coeff<<<cvo_state->num_fixed / CUDA_BLOCK_SIZE + 1, CUDA_BLOCK_SIZE>>>
      ( cvo_state->ell, params->ell_init, params->is_using_range_ell,
        cvo_state->num_fixed, cvo_state->A,
        thrust::raw_pointer_cast( cvo_state->cloud_x_gpu->points.data()  ),
        thrust::raw_pointer_cast( cvo_state->cloud_y_gpu->points.data() ),
        thrust::raw_pointer_cast(cvo_state->xiz.data()), 
        thrust::raw_pointer_cast(cvo_state->xi2z.data()),
        thrust::raw_pointer_cast(cvo_state->xi3z.data()),
        thrust::raw_pointer_cast(cvo_state->xi4z.data()),
        thrust::raw_pointer_cast(cvo_state->normxiz2.data()),
        thrust::raw_pointer_cast(cvo_state->xiz_dot_xi2z.data()),
        thrust::raw_pointer_cast(cvo_state->epsil_const.data()),
        num_neighbors,
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
    //Eigen::Vector4f p_coef(4);
    //p_coef << 4.0*float(E),3.0*float(D),2.0*float(C),float(B);
    Eigen::Matrix<double, 4, 1, Eigen::DontAlign> p_coef;

    //Eigen::VectorXd p_coef(4);
    p_coef << 4.0*(E),3.0*(D),2.0*(C),(B);
    if (debug_print)
      std::cout<<"BCDE is "<<p_coef.transpose()<<std::endl;
    
    // solve polynomial roots
    //Eigen::VectorXcd rc = poly_solver(p_coef);
    Eigen::Vector3cd rc = poly_solver_order3(p_coef);
    
    
    // find usable step size
    //float temp_step = numeric_limits<float>::max();
    double temp_step = numeric_limits<double>::max();
    for(int i=0;i<rc.real().size();i++) {
      if(rc(i,0).real()>0 && rc(i,0).real()<temp_step && std::fabs(rc(i,0).imag())<1e-5) {
        //if( fabs( rc(i,0).real())<temp_step && std::fabs(rc(i,0).imag())<1e-5) {

        temp_step = rc(i,0).real();
        //break;
      }
    }
    if (debug_print)
      std::cout<<"step size "<<temp_step<<"\n original_rc is \n"<< rc<<std::endl;

    // if none of the roots are suitable, use min_step
    cvo_state->step = temp_step==numeric_limits<double>::max()? params->min_step:temp_step;
    // if step>0.8, just use 0.8 as step
    if (temp_step > params->max_step)
      cvo_state->step = params->max_step;
    else if (temp_step < params->min_step)
      cvo_state->step = params->min_step;
    else
      cvo_state->step = temp_step;

    if (debug_print) 
      std::cout<<"step size "<<cvo_state->step<<"\n";
        
    
  }


  static bool A_sparsity_indicator_ell_update(std::queue<float> & indicator_start_queue,
                                              std::queue<float> & indicator_end_queue,
                                              float & indicator_start_sum,
                                              float & indicator_end_sum,
                                              const float indicator,
                                              const CvoParams & params
                                              ) {
    // decrease or increase lengthscale using indicator
    // start queue is not full yet
    bool decrease = false;
    int queue_len = params.indicator_window_size;
    if(indicator_start_queue.size() < queue_len){
      // add current indicator to the start queue
      indicator_start_queue.push(indicator);
      indicator_start_sum += indicator;
    } 
    if( indicator_start_queue.size() >= queue_len && indicator_end_queue.size() < queue_len){
      // add current indicator to the end queue and compute sum
      indicator_end_queue.push(indicator);
      indicator_end_sum += indicator;
    }

    //std::cout<<"Indicator is "<<indicator<<std::endl;
    //std::cout<<"ratio is "<<indicator_end_sum / indicator_start_sum<<std::endl;
    //static std::ofstream ratio("ip_ratio.txt",std::ofstream::out | std::ofstream::app );
    // start queue is full, start building the end queue
    if (indicator_start_queue.size() >= queue_len && indicator_end_queue.size() >= queue_len){
      //ratio<< indicator_end_sum / indicator_start_sum<<"\n"<<std::flush;
      //std::cout<<"ip ratio is "<<indicator_end_sum / indicator_start_sum<<" from "<<indicator_end_sum<<" over "<<indicator_start_sum<<std::endl;
      // check if criteria for decreasing legnthscale is satisfied
      if(indicator_end_sum / indicator_start_sum > 1 - params.indicator_stable_threshold
         &&
         indicator_end_sum / indicator_start_sum  < 1 + params.indicator_stable_threshold){
        decrease = true;
        std::queue<float> empty;
        std::swap( indicator_start_queue, empty );
        // std::swap( indicator_start_queue, indicator_end_queue );
        // std::swap( indicator_end_queue, empty );
        
        std::queue<float> empty2;
        std::swap( indicator_end_queue, empty2 );
        indicator_start_sum = 0;
        //indicator_start_sum = indicator_end_sum;
        indicator_end_sum = 0;
      }
      /*
      // check if criteria for increasing legnthscale is satisfied
      else if(indicator_end_sum / indicator_start_sum < 0.7){
      increase = true;
      std::queue<float> empty;
      std::swap( indicator_start_queue, empty );
      std::queue<float> empty2;
      std::swap( indicator_end_queue, empty2 );
      indicator_start_sum = 0;
      indicator_end_sum = 0;
      }*/
      else {
        //move the first indicator in the end queue to the start queue 
        indicator_end_sum -= indicator_end_queue.front();
        indicator_start_sum += indicator_end_queue.front();
        indicator_start_queue.push(indicator_end_queue.front());
        indicator_end_queue.pop();
        indicator_start_sum -= indicator_start_queue.front();
        indicator_start_queue.pop();
        // add current indicator to the end queue and compute sum
        indicator_end_queue.push(indicator);
        indicator_end_sum += indicator;
      }
    } //else {
      //ratio << 1.0 <<"\n"<< std::flush;
      
    // }
      

    /*
    // detect indicator drop and skip iteration
    if((last_indicator - indicator) / last_indicator > 0.2){
    // suddenly drop
    if(last_decrease){
    // drop because of decreasing lenthscale, keep track of the last indicator
    last_indicator = indicator;
    }
    else{
    // increase lengthscale and skip iteration
    increase = true;
    skip_iteration = true;
    }
    }
    else{
    // nothing bad happened, keep track of the last indicator
    last_indicator = indicator;
    }*/

    // DEBUG
    // std::cout << "indicator=" << indicator << ", start size=" << indicator_start_queue.size() << ", sum=" << indicator_start_sum \
    // << ", end size=" << indicator_end_queue.size() << ", sum=" << indicator_end_sum << std::endl;


    /*
      if(decrease && cvo_state.ell > params.ell_min){
      cvo_state.ell = cvo_state.ell * 0.9;
      last_decrease = true;
      decrease = false;
      }
      else if(~decrease && last_decrease){
      last_decrease = false;
      }
      if(increase && cvo_state.ell < params.ell_max){
      cvo_state.ell = cvo_state.ell * 1.1;
      increase = false;
      }

      if(skip_iteration){
      continue;
      }
    */

    return decrease;
  }


  __global__
  void fill_in_A_mat_euclidean(// input
                               CvoPoint * points_a,
                               int a_size,
                               CvoPoint * points_b,
                               int b_size,
                               // output
                               double * l2_sum
                               ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > a_size - 1)
      return;

    CvoPoint * p_a =  &points_a[i];

    for (int j = 0; j < b_size; j++) {
      CvoPoint * p_b = &points_b[j];
      l2_sum[i] += (double)squared_dist(*p_a, *p_b );
    }

  }
  
  float compute_std_dist_between_two_pc(std::shared_ptr<CvoPointCloudGPU> source_points, 
                                        std::shared_ptr<CvoPointCloudGPU> target_points_transformed
                                        ) {
    // asssume state is clean
    double total = 0;


    int num_points_source = source_points->size();
    int num_points_target = target_points_transformed->size();
    thrust::device_vector<double> l2_sum(num_points_source);

    CvoPoint * points_fixed_raw = thrust::raw_pointer_cast (  source_points->points.data() );
    CvoPoint * points_moving_raw = thrust::raw_pointer_cast( target_points_transformed->points.data() );

    
    fill_in_A_mat_euclidean<<<(num_points_target / CUDA_BLOCK_SIZE )+1, CUDA_BLOCK_SIZE >>>
      ( points_fixed_raw, num_points_source,
        points_moving_raw, num_points_target,
        thrust::raw_pointer_cast(l2_sum.data()));

    double dist_square_sum = thrust::reduce(l2_sum.begin(), l2_sum.end());
    double std_dist = sqrt( dist_square_sum /(double)num_points_source / (double) num_points_target   );
    
    return static_cast<float> (std_dist);
    
    
  }

  static
  __attribute__((force_align_arg_pointer)) 
  int align_impl(const CvoParams & params,
                 const CvoParams * params_gpu,
                 CvoState & cvo_state,
                 const Eigen::Matrix4f & init_guess_transform,
                 // output
                 Eigen::Ref<Eigen::Matrix4f> transform,
                 Association * association_mat,
                 double * registration_seconds
                 ) {
    
    std::ofstream ell_file;
    std::ofstream dist_change_file;
    std::ofstream transform_file;
    std::ofstream step_file;
    std::ofstream inner_product_file;
    if (is_logging) {
      ell_file.open("ell_history.txt", std::ofstream::out);
      dist_change_file.open("dist_change_history.txt", std::ofstream::out);
      transform_file.open("transformation_history.txt", std::ofstream::out);
      step_file.open("step_history.txt", std::ofstream::out);
      inner_product_file.open("inner_product_history.txt", std::ofstream::out);
    }

    Mat33f R = init_guess_transform.block<3,3>(0,0);
    Vec3f T= init_guess_transform.block<3,1>(0,3);    
    int ret = 0;
    Eigen::Vector3f omega, v;
    
    cudaEvent_t cuda_start, cuda_stop;                                                                              
    cudaEventCreate(&cuda_start);                                                                              
    cudaEventCreate(&cuda_stop);                                                                               
    cudaEventRecord(cuda_start, 0);

    std::cout<<"Start iteration, init transform is \n";
    std::cout<<init_guess_transform<<std::endl;
    std::cout<<"Max iter is "<<params.MAX_ITER<<std::endl;

    std::queue<float> indicator_start_queue;
    std::queue<float> indicator_end_queue;
    float indicator_start_sum = 0;
    float indicator_end_sum = 0;
    int use_least_square = 0;
    int min_ell_iters  = 0;
    
    int k = 0;
    int num_neighbors = params.is_using_kdtree? perl_registration::KDTREE_K_SIZE : params.nearest_neighbors_max;
    for(; k<params.MAX_ITER; k++){
      if (debug_print) printf("new iteration %d, ell is %f\n", k, cvo_state.ell);
      cvo_state.reset_state_at_new_iter(num_neighbors);
      if (debug_print) printf("just reset A mat\n");
      
      // update transformation matrix to CvoState
      update_tf(R, T, &cvo_state, transform);

      if (is_logging) {
        Eigen::Matrix4f Tmat = transform;
        transform_file << Tmat(0,0) <<" "<< Tmat(0,1) <<" "<< Tmat(0,2) <<" "<< Tmat(0,3) <<" "
                       << Tmat(1,0) <<" "<< Tmat(1,1) <<" "<< Tmat(1,2) <<" "<< Tmat(1,3) <<" "
                       << Tmat(2,0) <<" "<< Tmat(2,1) <<" "<< Tmat(2,2) <<" "<< Tmat(2,3)
                       <<"\n"<< std::flush;
      }

      // transform point cloud
      transform_pointcloud_thrust(cvo_state.cloud_y_gpu_init, cvo_state.cloud_y_gpu,
                                  cvo_state.R_gpu, cvo_state.T_gpu, false ); 

      // update the inner product matrix
      if ( params.is_using_kdtree) {
        se_kernel_kdtree(params, params_gpu,
                         cvo_state.cloud_x_gpu, cvo_state.cloud_y_gpu,
                         cvo_state.ell,
                         *cvo_state.kdtree_moving_points,
                         transform, num_neighbors,
                         cvo_state.cloud_x_gpu_transformed_kdtree,
                         cvo_state.kdtree_inds_results,
                         &cvo_state.A_host, cvo_state.A);
      } else
        se_kernel(params_gpu, cvo_state.cloud_x_gpu, cvo_state.cloud_y_gpu ,
                  num_neighbors, cvo_state.ell,
                  &cvo_state.A_host, cvo_state.A);
      cudaDeviceSynchronize();
      if (debug_print ) {
        float sum_A = thrust::reduce(thrust::device, cvo_state.A_host.mat, cvo_state.A_host.mat + cvo_state.A_host.rows * num_neighbors, (float)0);
        std::cout<<"nonzeros in A "<<nonzeros(&cvo_state.A_host)<<", sum of A is "<<sum_A<<std::endl;
        //std::cout<<"time for se_kernel is "
        //         <<std::chrono::duration_cast<std::chrono::milliseconds>((end- start)).count()<<std::endl;
        std::cout<<"A rows is "<<cvo_state.A_host.rows<<", A cols is "<<cvo_state.A_host.cols<<std::endl;
        if (k %100 == 0) {
          std::ofstream neighbors_dist("neighbors_dist_" + std::to_string(k) + ".txt", std::ofstream::out);
          thrust::host_vector<int> neighbors(cvo_state.A_host.rows);
          cudaMemcpy(neighbors.data(), cvo_state.A_host.nonzeros, neighbors.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost );
          for (auto p: neighbors)
            neighbors_dist << p <<"\n";
          neighbors_dist.close();          
        }

      }

      // compute omega and v
      compute_flow(&cvo_state, params_gpu, &omega, &v, num_neighbors);
      if (debug_print) std::cout<<"iter "<<k<< "omega: \n"<<omega.transpose()<<"\nv: \n"<<v.transpose()<<std::endl;
      if (k == 0) {
        printf("iter=0: nonzeros in A is %d\n", cvo_state.A_host.nonzero_sum);
      }

      // compute indicator and change lenthscale if needed
      //float indicator = (double)cvo_state.A_host.nonzero_sum / (double)source_points.num_points() / (double) target_points.num_points();
      compute_step_size(&cvo_state, &params, num_neighbors);
      
      // stop if the step size is too small
      if (debug_print) printf("copy gradient to cpu...");
      cudaMemcpy(omega.data(), cvo_state.omega->data(), sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost);
      cudaMemcpy(v.data(), cvo_state.v->data(), sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost);
      if(omega.cast<double>().norm()<params.eps && v.cast<double>().norm()<params.eps){
         std::cout<<"break: norm, omega: "<<omega.norm()<<", v: "<<v.norm()<<std::endl;
        if (omega.norm() < 1e-8 && v.norm() < 1e-8) ret = -1;
        break;
      }
      // stacked omega and v for finding dtrans
      Eigen::Matrix<float, 6,1> vec_joined;
      vec_joined << omega, v;
      Eigen::Matrix<float,3,4> dtrans = Exp_SEK3(vec_joined, cvo_state.step).cast<float>();
      Eigen::Matrix3d dR = dtrans.block<3,3>(0,0).cast<double>();
      Eigen::Vector3d dT = dtrans.block<3,1>(0,3).cast<double>();
      T = (R.cast<double>() * dT + T.cast<double>()).cast<float>();
      //Eigen::Transform<double, 3>
      //Eigen::AngleAxisd R_angle(R.cast<double>() * dR);
      //R = R_angle.toRotationMatrix().cast<float>(); // re-orthogonalization
      R = (R.cast<double>() * dR).cast<float>();

      // reduce ell, if the se3 distance is smaller than eps2, break
      Eigen::Matrix4d dRT = Eigen::Matrix4d::Identity();
      dRT.block<3,3>(0,0) = dR;
      dRT.block<3,1>(0,3) = dT;
      Sophus::SE3d dRT_sophus(dRT);
      double dist_this_iter = dRT_sophus.log().norm();
      //double dist_this_iter = dist_se3(dR,dT);
      if (debug_print)  {
        std::cout<<"just computed distk. dR "<<dR<<"\n dt is "<<dT<<std::endl;
	std::cout<<"dist: "<<dist_this_iter <<std::endl<<"check bounds....\n";
	if (std::isnan(dist_this_iter)) {
          break; 
	}
      }
      // approximate the inner product further more
      float ip_curr = (float)((double)cvo_state.A_host.nonzero_sum / sqrt((double)cvo_state.num_fixed * (double) cvo_state.num_moving ));
      //float ip_curr = (float) this->inner_product(source_points, target_points, transform);
      bool need_decay_ell = A_sparsity_indicator_ell_update( indicator_start_queue,
                                                             indicator_end_queue,
                                                             indicator_start_sum,
                                                             indicator_end_sum,
                                                             ip_curr,
                                                             params);

      if (is_logging) {
        ell_file << cvo_state.ell<<"\n"<<std::flush;
        dist_change_file << dist_this_iter<<"\n"<<std::flush;

        //float ip_curr = (double)cvo_state.A_host.nonzero_sum / (double)source_points.num_points() / (double) target_points.num_points();
        //effective_points_file << indicator << "\n" << std::flush;
        inner_product_file<<ip_curr<<"\n"<<std::flush;
        //inner_product_file<<this->inner_product(source_points, target_points, transform)<<"\n"<<std::flush;
      }
      if (debug_print)  std::cout<<"dist: "<<dist_this_iter <<std::endl<<"check bounds....\n";
      if(dist_this_iter<params.eps_2 ){
        std::cout<<"break: dist: "<<dist_this_iter<<std::endl;
        break;
      }
      if (k>params.ell_decay_start && need_decay_ell  ) {
        cvo_state.ell = cvo_state.ell * params.ell_decay_rate;
        if (cvo_state.ell < params.ell_min)
          cvo_state.ell = params.ell_min;
      }
      if (params.is_using_kdtree)
        num_neighbors = std::min(perl_registration::KDTREE_K_SIZE, (int)std::ceil( (double) nonzeros(&cvo_state.A_host) / (double)cvo_state.A_host.rows  ) * params.is_using_kdtree);
      else {

          thrust::device_ptr<unsigned int> max_ind = thrust::device_pointer_cast(cvo_state.A_host.nonzeros);
          unsigned int max_ind_val = *thrust::max_element(thrust::device, max_ind, max_ind + cvo_state.A_host.rows);
          if (debug_print) {
            std::cout<<"max number of neighbors is "<<max_ind_val<<std::endl;
          }
          if (is_logging) {
            std::ofstream max_neighbor_f("max_neibor_all.txt", std::ofstream::out | std::ofstream::app);
            max_neighbor_f << max_ind_val<<std::endl<<std::flush;
            max_neighbor_f.close();
          }
          //if (max_ind_val >= num_neighbors - 10)
          num_neighbors  = std::min(params.nearest_neighbors_max , (int)(max_ind_val * 1.2));

      }
      if (debug_print) std::cout<<"num_neighbors is "<<num_neighbors<<std::endl;      
    }
    cudaEventRecord(cuda_stop, 0);                                                                             
    cudaEventSynchronize(cuda_stop);                                                                           
    float elapsedTime, totalTime;                                                                     
    cudaEventElapsedTime(&elapsedTime, cuda_start, cuda_stop);                                                    
    cudaEventDestroy(cuda_start);                                                                              
    cudaEventDestroy(cuda_stop);                                                                               
    totalTime = elapsedTime/(1000);                                                             
    //printf("2D: thrust min element = %d, max element = %d\n", minele, maxele);                            
    //printf("2D: thrust time = %f\n", totalTime2d);  
    //auto end_all = chrono::system_clock::now();    
    //t_all = end_all - start_all;
    std::cout<<"cvo # of iterations is "<<k<<std::endl;
    std::cout<<"final max num of neighbors is "<<num_neighbors<<"\n"<<std::flush;
    std::cout<<"t_all is "<<totalTime<<"\n"<<std::flush;
    std::cout<<"non adaptive cvo ends. final ell is "<<cvo_state.ell<<", final iteration is "<<k
             <<"MAX_ITER is "<<params.MAX_ITER<<std::endl;


    if (params.is_exporting_association && association_mat)
      gpu_association_to_cpu(cvo_state.A_host, *association_mat,
                             cvo_state.num_fixed,
                             cvo_state.num_moving,
                             num_neighbors);

    if (registration_seconds)
      //*registration_seconds = t_all.count();
      *registration_seconds = (double) totalTime;//t_all.count();

    update_tf(R, T, &cvo_state, transform);

    if (is_logging) {
      ell_file.close();
      dist_change_file.close();
      transform_file.close();
      step_file.close();
      inner_product_file.close();
    }
    return ret;
  }

  int CvoGPU::align(const pcl::PointCloud<CvoPoint>& source_points,
                    const pcl::PointCloud<CvoPoint>& target_points,
                    const Eigen::Matrix4f & init_guess_transform,
                    // outputs
                    Eigen::Ref<Eigen::Matrix4f> transform,
                    Association * association_mat,                    
                    double *registration_seconds) const {
    

    std::cout<<"[align] convert points to gpu\n"<<std::flush;
    if (source_points.size() == 0 || target_points.size() == 0) {
      std::cout<<"[align] point clouds inputs are empty\n";
      return 0;
    }
    CvoPointCloudGPU::SharedPtr source_gpu = pcl_PointCloud_to_gpu(source_points);
    CvoPointCloudGPU::SharedPtr target_gpu = pcl_PointCloud_to_gpu(target_points);

    CvoState cvo_state(source_gpu, target_gpu, params);
    std::cout<<"construct new cvo state..., init ell is "<<cvo_state.ell<<std::endl;

    int ret = align_impl(params, params_gpu,
                         cvo_state, init_guess_transform, 
                         transform,
                         association_mat,
                         registration_seconds);
    //std::cout<<"Result Transform is "<<transform<<std::endl;
    return ret;
    
  }

  
  int CvoGPU::align(// inputs
                    const CvoPointCloud& source_points,
                    const CvoPointCloud& target_points,
                    const Eigen::Matrix4f & init_guess_transform,
                    // outputs
                    Eigen::Ref<Eigen::Matrix4f> transform,
                    Association * association_mat,
                    double *registration_seconds) const {
    std::cout<<"[align] convert points to gpu\n"<<std::flush;
    if (source_points.num_points() == 0 || target_points.num_points() == 0) {
      std::cout<<"[align] point clouds inputs are empty\n";
      return 0;
    }
    CvoPointCloudGPU::SharedPtr source_gpu = CvoPointCloud_to_gpu(source_points);
    CvoPointCloudGPU::SharedPtr target_gpu = CvoPointCloud_to_gpu(target_points);

    CvoState cvo_state(source_gpu, target_gpu, params);

    std::cout<<"construct new cvo state..., init ell is "<<cvo_state.ell<<std::endl;

    int ret = align_impl(params, params_gpu,
                         cvo_state, init_guess_transform,
                         transform,
                         association_mat,
                         registration_seconds);
    //std::cout<<"Result Transform is "<<transform<<std::endl;
    return ret;
  }
  /*
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
    //return float(A_mat.sum())/float(A_mat.nonZeros());
    //return A_mat.sum()/(fixed_positions.size())*1e5/moving_positions.size() ;
    //return float(A_mat.nonZeros()) / float(fixed_positions.size()) / float(moving_positions.size() ) * 100 ;
    //}
  
    return A_mat.sum()/fixed_positions.size()/moving_positions.size() ;
  }
  */

  static
  //  __attribute__((force_align_arg_pointer)) 
  float inner_product_impl (CvoPointCloudGPU::SharedPtr source_gpu,
                            CvoPointCloudGPU::SharedPtr target_gpu,
                            const Eigen::Matrix4f & init_guess_transform,
                            const CvoParams & params,
                            const CvoParams * params_gpu,
                            bool is_using_isotropic_kernel,
                            float curr_ell,
                            const Eigen::Matrix3f * non_isotropic_kernel,
                            Association * association_output=nullptr) {

    Mat33f R = init_guess_transform.block<3,3>(0,0);
    Vec3f T= init_guess_transform.block<3,1>(0,3);
    
    CvoState cvo_state(source_gpu, target_gpu, params);
    
    //Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    Eigen::Vector3f omega, v;

    cvo_state.reset_state_at_new_iter();

    // update transformation matrix to CvoState
    //Eigen::Matrix<float,3,4> transform;

    //update_tf(R, T, &cvo_state, transform);
    update_tf(R, T, &cvo_state);

    transform_pointcloud_thrust(cvo_state.cloud_y_gpu_init, cvo_state.cloud_y_gpu,
                                cvo_state.R_gpu, cvo_state.T_gpu, false ); 

    // update the inner product matrix
    if (is_using_isotropic_kernel)
      se_kernel(params_gpu, cvo_state.cloud_x_gpu, cvo_state.cloud_y_gpu,
                params.nearest_neighbors_max, curr_ell, //cvo_state.ell, 
                &cvo_state.A_host, cvo_state.A);
    else {
      assert (non_isotropic_kernel != nullptr);
      Eigen::Matrix3f kernel_inv = non_isotropic_kernel->inverse();
      se_kernel_dense(// input
                      params_gpu,  cvo_state.cloud_x_gpu, cvo_state.cloud_y_gpu,
                      params.nearest_neighbors_max,
                      kernel_inv,
                      // output
                      &cvo_state.A_host, cvo_state.A);                      
    }
    cudaDeviceSynchronize();


    float ip_value = A_sum(&cvo_state.A_host);


    if (association_output) {
      gpu_association_to_cpu(cvo_state.A_host, *association_output,
                             cvo_state.num_fixed,
                             cvo_state.num_moving);
    }
    
    return ip_value;
  }
  
  float CvoGPU::inner_product_gpu(const CvoPointCloud& source_points,
                                  const CvoPointCloud& target_points,
                                  const Eigen::Matrix4f & init_guess_transform,
                                  float ell
                                  ) const {

    CvoPointCloudGPU::SharedPtr source_gpu = CvoPointCloud_to_gpu(source_points);
    CvoPointCloudGPU::SharedPtr target_gpu = CvoPointCloud_to_gpu(target_points);

    
    return inner_product_impl(source_gpu, target_gpu, init_guess_transform, params, params_gpu,
                              true, ell, nullptr,
                              nullptr
                              );
  }

  float CvoGPU::inner_product_gpu(const pcl::PointCloud<CvoPoint>& source_points_pcl,
                                  const pcl::PointCloud<CvoPoint>& target_points_pcl,
                                  const Eigen::Matrix4f & init_guess_transform,
                                  float ell
                                  ) const {
 
    CvoPointCloudGPU::SharedPtr source_gpu = pcl_PointCloud_to_gpu(source_points_pcl);
    CvoPointCloudGPU::SharedPtr target_gpu = pcl_PointCloud_to_gpu(target_points_pcl);


    return inner_product_impl(source_gpu, target_gpu, init_guess_transform, params, params_gpu,
                              true, ell, nullptr,
                              nullptr);
  }



  
  float CvoGPU::function_angle(const CvoPointCloud& source_points,
                               const CvoPointCloud& target_points,
                               const Eigen::Matrix4f & t2s_frame_transform,
                               float ell,
                               bool is_approximate,
                               bool is_gpu
                               ) const {
    if (source_points.num_points() == 0 || target_points.num_points() == 0) {
      return 0;
    }

    Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
    float fxfz = 0, fx_norm = 0, fz_norm = 0, cosine_value = 0;
    if (is_gpu)
      fxfz = inner_product_gpu(source_points, target_points, t2s_frame_transform, ell);
    else
      fxfz = inner_product_cpu(source_points, target_points, t2s_frame_transform, ell);
    if (is_approximate) {
      fx_norm = sqrt(source_points.num_points());
      fz_norm = sqrt(target_points.num_points());
    } else {
      if (is_gpu) {
        fx_norm = sqrt(inner_product_gpu(source_points, source_points, identity, ell));
        fz_norm = sqrt(inner_product_gpu(target_points, target_points, identity, ell));
      } else {
        fx_norm = sqrt(inner_product_cpu(source_points, source_points, identity, ell));
        fz_norm = sqrt(inner_product_cpu(target_points, target_points, identity, ell));        
      }
    }
    cosine_value = fxfz / (fx_norm * fz_norm);
    std::cout<<"fxfz is "<<fxfz<<", fx_norm is "<<fx_norm<<", fz_norm is "<<fz_norm<<std::endl;
    return cosine_value;
  }

  float CvoGPU::function_angle(const pcl::PointCloud<CvoPoint>& source_points_pcl,
                               const pcl::PointCloud<CvoPoint>& target_points_pcl,
                               const Eigen::Matrix4f & t2s_frame_transform,
                               float ell,
                               bool is_approximate
                               ) const {

    if (source_points_pcl.size() == 0 || target_points_pcl.size() == 0) {
      return 0;
    }

    Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
    float fxfz = 0, fx_norm = 0, fz_norm = 0, cosine_value = 0;
    fxfz = inner_product_gpu(source_points_pcl, target_points_pcl, t2s_frame_transform, ell);
    if (is_approximate) {
      fx_norm = sqrt(source_points_pcl.size());
      fz_norm = sqrt(target_points_pcl.size());
    } else {
      fx_norm = sqrt(inner_product_gpu(source_points_pcl, source_points_pcl, identity, ell));
      fz_norm = sqrt(inner_product_gpu(target_points_pcl, target_points_pcl, identity, ell));
    }
    cosine_value = fxfz / (fx_norm * fz_norm);    

    return cosine_value;

  }


  void CvoGPU::compute_association_gpu(const CvoPointCloud& source_points,
                                       const CvoPointCloud& target_points,
                                       const Eigen::Matrix4f & T_target_frame_to_source_frame,
                                       float lengthscale,
                                       // output
                                       Association & association
                                       ) const {

    if (source_points.num_points() == 0 || target_points.num_points() == 0)
      return;

    CvoPointCloudGPU::SharedPtr source_gpu = CvoPointCloud_to_gpu(source_points);
    CvoPointCloudGPU::SharedPtr target_gpu = CvoPointCloud_to_gpu(target_points);

    //CvoParams params_tmp = params;
    //params_tmp.ell_init = params.ell_min;
    //this->write_params(&params_tmp);
    inner_product_impl(source_gpu, target_gpu, T_target_frame_to_source_frame, params, params_gpu,
                       true, lengthscale, nullptr,
                       &association);
    //this->write_params(&params);
    /*
    // for debugging
    //for (int k=0; k<association.pairs.outerSize(); ++k) {
    std::cout<<"association result for i==10: ";
    for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(association.pairs,10); it; ++it) {
      //double grad_prefix = it.value();
      float val = it.value();
      int idx1 = it.row();   // row index
      int idx2 = it.col();   // col index (here it is equal to k)
      std::cout<<"j = "<<idx2<<" with value "<<val<<",  ";
    }
    std::cout<<std::endl;
      //}
      */
  }

  static
  //  __attribute__((force_align_arg_pointer)) 
  void inner_product_non_isotropic_impl (CvoPointCloudGPU::SharedPtr source_gpu,
                                         CvoPointCloudGPU::SharedPtr target_gpu,
                                         const Eigen::Matrix4f & init_guess_transform,
                                         const CvoParams & params,
                                         const CvoParams * params_gpu,
                                         //bool is_using_isotropic_kernel,
                                         //float curr_ell,
                                         const Eigen::Matrix3f * non_isotropic_kernel,
                                         Association * association_output=nullptr) {
    
    Mat33f R = init_guess_transform.block<3,3>(0,0);
    Vec3f T= init_guess_transform.block<3,1>(0,3);
    
    CvoState cvo_state(source_gpu, target_gpu, params);
    
    //Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    Eigen::Vector3f omega, v;

    cvo_state.reset_state_at_new_iter();

    // update transformation matrix to CvoState
    //Eigen::Matrix<float,3,4> transform;

    //update_tf(R, T, &cvo_state, transform);
    update_tf(R, T, &cvo_state);

    transform_pointcloud_thrust(cvo_state.cloud_y_gpu_init, cvo_state.cloud_y_gpu,
                                cvo_state.R_gpu, cvo_state.T_gpu, false ); 


    assert (non_isotropic_kernel != nullptr);
    Eigen::Matrix3f kernel_inv = non_isotropic_kernel->inverse();

    CvoParams new_params = params;
    new_params.is_using_geometric_type = false;
    CvoParams * new_params_gpu;
    cudaMalloc((void**)&new_params_gpu, sizeof(CvoParams) );
    cudaMemcpy( (void*)new_params_gpu, &new_params, sizeof(CvoParams), cudaMemcpyHostToDevice  );

    
    se_kernel_dense(// input
                    new_params_gpu,  cvo_state.cloud_x_gpu, cvo_state.cloud_y_gpu,
                    params.nearest_neighbors_max,
                    kernel_inv,
                    // output
                    &cvo_state.A_host, cvo_state.A);                      

    cudaDeviceSynchronize();

    gpu_association_to_cpu(cvo_state.A_host, *association_output,
                           cvo_state.num_fixed,
                           cvo_state.num_moving);

    cudaFree(new_params_gpu);
    
    return;
  }

  

  void CvoGPU::compute_association_gpu(const CvoPointCloud& source_points,
                                       const CvoPointCloud& target_points,
                                       const Eigen::Matrix4f & T_target_frame_to_source_frame,
                                       const Eigen::Matrix3f & non_isotropic_kernel,
                                       // output
                                       Association & association
                                       ) const {
    if (source_points.num_points() == 0 || target_points.num_points() == 0)
      return;

    CvoPointCloudGPU::SharedPtr source_gpu = CvoPointCloud_to_gpu(source_points);
    CvoPointCloudGPU::SharedPtr target_gpu = CvoPointCloud_to_gpu(target_points);

    //CvoParams params_tmp = params;
    //params_tmp.ell_init = params.ell_min;
    //this->write_params(&params_tmp);
    inner_product_non_isotropic_impl(source_gpu, target_gpu, T_target_frame_to_source_frame, params, params_gpu,
                                     &non_isotropic_kernel,
                                     &association);
    
  }
  
  


}
