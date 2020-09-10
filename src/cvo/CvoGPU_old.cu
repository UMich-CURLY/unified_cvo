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
//#include "cukdtree/cukdtree.h"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>


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
using namespace std;
using namespace nanoflann;

namespace cvo{
  
  typedef Eigen::Triplet<float> Trip_t;

  static bool is_logging = true;
  static bool debug_print =false;



  
  __global__
  void copy_covariances(// inputs
                        const float * covariance,
                        const float * eigenvalues,
                        int num_points,
                        // outputs
                        CvoPoint * points_gpu
                        ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > num_points - 1) {
      return;
    }
    memcpy(points_gpu[i].covariance, covariance+i*9, 9*sizeof(float));
    memcpy(points_gpu[i].cov_eigenvalues, eigenvalues+i*3, 3*sizeof(float));
    
  }
  
   
  CvoPointCloudGPU::SharedPtr CvoPointCloud_to_gpu(const CvoPointCloud & cvo_cloud ) {
    int num_points = cvo_cloud.num_points();
    const ArrayVec3f & positions = cvo_cloud.positions();
    const Eigen::Matrix<float, Eigen::Dynamic, FEATURE_DIMENSIONS> & features = cvo_cloud.features();
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> & normals = cvo_cloud.normals();
    // const Eigen::Matrix<float, Eigen::Dynamic, 2> & types = cvo_cloud.types();
#ifdef IS_USING_SEMANTICS
    auto & labels = cvo_cloud.labels();
#endif
    // set basic informations for pcl_cloud
    thrust::host_vector<CvoPoint> host_cloud;
    host_cloud.resize(num_points);

    int actual_num = 0;
    for(int i=0; i<num_points; ++i){
      (host_cloud)[i].x = positions[i](0);
      (host_cloud)[i].y = positions[i](1);
      (host_cloud)[i].z = positions[i](2);
      if (FEATURE_DIMENSIONS >= 3) {
        (host_cloud)[i].r = (uint8_t)std::min(255.0, (features(i,0) * 255.0));
        (host_cloud)[i].g = (uint8_t)std::min(255.0, (features(i,1) * 255.0));
        (host_cloud)[i].b = (uint8_t)std::min(255.0, (features(i,2) * 255.0));
      }

      ///memcpy(host_cloud[i].features, features.row(i).data(), FEATURE_DIMENSIONS * sizeof(float));
      for (int j = 0; j < FEATURE_DIMENSIONS; j++)
        host_cloud[i].features[j] = features(i,j);

#ifdef IS_USING_SEMANTICS
      labels.row(i).maxCoeff(&host_cloud[i].label);
      for (int j = 0; j < NUM_CLASSES; j++)
        host_cloud[i].label_distribution[j] = labels(i,j);
#endif
      
#ifdef IS_USING_NORMALS
      for (int j = 0; j < 3; j++)
        host_cloud[i].normal[j] = normals(i,j);
#endif


#ifdef IS_USING_COVARIANCE
      memcpy(host_cloud[i].covariance, cvo_cloud.covariance().data()+ i*9, sizeof(float)*9  );
      memcpy(host_cloud[i].cov_eigenvalues, cvo_cloud.eigenvalues().data() + i*3, sizeof(float)*3);
      if (i == 0)
        std::cout<<"cov_eigenvalues "<<host_cloud[i].cov_eigenvalues[0]<<", "<<host_cloud[i].cov_eigenvalues[1]<<", "<<host_cloud[i].cov_eigenvalues[2]<<std::endl;
#endif      

      //if (i == 1000) {
      //  printf("Total %d, Raw input from pcl at 1000th: \n", num_points);
      //  print_point(host_cloud[i]);
      //}
      actual_num ++;
    }
    //gpu_cloud->points = host_cloud;
    CvoPointCloudGPU::SharedPtr gpu_cloud(new CvoPointCloudGPU);
    gpu_cloud->points = host_cloud;

    /*
#ifdef IS_USING_COVARIANCE    
    auto covariance = &cvo_cloud.covariance();
    auto eigenvalues = &cvo_cloud.eigenvalues();
    thrust::device_vector<float> cov_gpu(cvo_cloud.covariance());
    thrust::device_vector<float> eig_gpu(cvo_cloud.eigenvalues());
    copy_covariances<<<host_cloud.size()/256 +1, 256>>>(thrust::raw_pointer_cast(cov_gpu.data()),
                                                        thrust::raw_pointer_cast(eig_gpu.data()),
                                                        host_cloud.size(),
                                                        thrust::raw_pointer_cast(gpu_cloud->points.data()));
#endif    
    */
    return gpu_cloud;
  }

  CvoGPU::CvoGPU(const std::string & param_file) {
    // read_CvoParams(param_file.c_str(), &params);
    read_CvoParams_yaml(param_file.c_str(), &params);
    printf("Some Cvo Params are: ell_init: %f, eps_2: %f\n", params.ell_init, params.eps_2);
    cudaMalloc((void**)&params_gpu, sizeof(CvoParams) );
    cudaMemcpy( (void*)params_gpu, &params, sizeof(CvoParams), cudaMemcpyHostToDevice  );

  }
  
  void CvoGPU::write_params(CvoParams * p_cpu) {
    //params = *p_cpu;
    cudaMemcpy( (void*)params_gpu, p_cpu, sizeof(CvoParams), cudaMemcpyHostToDevice  );
    
  }


  CvoGPU::~CvoGPU() {
    cudaFree(params_gpu);
    
  }


  struct transform_point
  {
    const Mat33f * R;
    const Vec3f * T;

    transform_point(const Mat33f * R_gpu, const Vec3f * T_gpu): R(R_gpu), T(T_gpu){}
    
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
      /*      
      result.r = p_init.r;
      result.g = p_init.g;
      result.b = p_init.b;
      result.label = p_init.label;
      memcpy(result.features, p_init.features, FEATURE_DIMENSIONS * sizeof(float) );
#ifdef IS_USING_SEMANTICS
      memcpy(result.label_distribution, p_init.label_distribution, NUM_CLASSES * sizeof(float) );
#endif
#ifdef IS_USING_NORMALS
      Eigen::Vector3f input_normal;
      input_normal << p_init.normal[0], p_init.normal[1], p_init.normal[2];
      Eigen::Vector3f trans_normal = (*R) * input_normal + (*T);
      result.normal[0] = trans_normal(0);
      result.normal[1] = trans_normal(1);
      result.normal[2] = trans_normal(2);
#endif
      */
#ifdef IS_USING_COVARIANCE
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
      
#endif      
      return result;
    }
  };


  void  transform_pointcloud_thrust(std::shared_ptr<CvoPointCloudGPU> init_cloud,
                                    std::shared_ptr<CvoPointCloudGPU> transformed_cloud,
                                    Mat33f * R_gpu, Vec3f * T_gpu
                                    ) {
    thrust::transform( init_cloud->begin(), init_cloud->end(),  transformed_cloud->begin(), 
                      transform_point(R_gpu, T_gpu));

    if (debug_print) {
    
      CvoPoint from_point;
      CvoPoint transformed_point;
      for (int i = 1000; i < 1001; i++){
        cudaMemcpy((void*)&transformed_point, (void*)thrust::raw_pointer_cast( transformed_cloud->points.data() +i), sizeof(CvoPoint), cudaMemcpyDeviceToHost  );
        cudaMemcpy((void*)&from_point, (void*)thrust::raw_pointer_cast( init_cloud->points.data() +i), sizeof(CvoPoint), cudaMemcpyDeviceToHost  );
        /*
        printf("print %d point after transformation:\n", i);
        printf("from\n ");
        pcl::print_point(from_point);
        printf("to\n");
        pcl::print_point(transformed_point  );
        */
      }     
    }
  
  }
  

  __device__
  float compute_range_ell(float curr_ell, float curr_dist_to_sensor, float min_dist, float max_dist ) {
#ifdef IS_USING_RANGE_ELL    
    float final_ell = ((curr_dist_to_sensor) / 500.0 + 1.0)* curr_ell;
    return  final_ell;
#else    
    return curr_ell;
#endif    
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

    if (debug_print) std::cout<<"transform mat R"<<transform.block<3,3>(0,0)<<"\nT: "<<transform.block<3,1>(0,3)<<std::endl;
  }


  typedef KDTreeVectorOfVectorsAdaptor<cloud_t, float>  kd_tree_t;

  __global__
  void fill_in_A_mat_gpu(const CvoParams * cvo_params,
                        //SquareExpParams * se_params,
                        CvoPoint * points_a,
                        int a_size,
                        CvoPoint * points_b,
                        int b_size,
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
    float sp_thres = cvo_params->sp_thres;
    float c_sigma = cvo_params->c_sigma;
#ifdef IS_USING_SEMANTICS
    float s_ell = cvo_params->s_ell;
#endif

    // convert k threshold to d2 threshold (so that we only need to calculate k when needed)
    float d2_thres = -2.0*l*l*log(sp_thres/s2);
    float d2_c_thres = -2.0*c_ell*c_ell*log(sp_thres/c_sigma/c_sigma);

    CvoPoint * p_a =  &points_a[i];
#ifdef IS_USING_SEMANTICS
    float * label_a = p_a ->label_distribution;
#endif


    unsigned int num_inds = 0;
    for (int j = 0; j < b_size ; j++) {
      int ind_b = j;
      if (num_inds == CVO_POINT_NEIGHBORS) break;

      CvoPoint * p_b = &points_b[ind_b];
      float d2 = (squared_dist( *p_b ,*p_a ));

      float normal_ip = 1;
#ifdef IS_USING_NORMALS
      normal_ip = fabs(p_a->normal[0] * p_b->normal[0] +
                       p_a->normal[1] * p_b->normal[1] +
                       p_a->normal[2] * p_b->normal[2]);
      //printf("normal ip is %f\n", normal_ip);
#endif      
      /*
      if ( i == 1000 && j== 1074) {
        CvoPoint * pb = points_b + j;
        printf("gpu se_kernel: i==%d,j==%d: d2 is %f, d2_thres is %f, point_a (%f, %f, %f), point_b: (%f, %f, %f)\n", i, j,d2, d2_thres,
              p_a->x, p_a->y, p_a->z,
              pb->x, pb->y,  pb->z );
              }*/
      
      if(d2<d2_thres  ){

#ifdef IS_GEOMETRIC_ONLY
        float a = s2*exp(-d2/(2.0*l*l)) * normal_ip;
        if (a > cvo_params->sp_thres){
          A_mat->mat[i * A_mat->cols + num_inds] = a;
          A_mat->ind_row2col[i * A_mat->cols + num_inds] = ind_b;
          num_inds++;
        }

#else
        float d2_color = squared_dist<float>(p_a->features, p_b->features, FEATURE_DIMENSIONS);

#ifdef IS_USING_SEMANTICS            
        float d2_semantic = squared_dist<float>(p_a->label_distribution, p_b->label_distribution, NUM_CLASSES);
#endif


        /*
        if (i == 1000 && j==1074) {
          float * fa = p_a->features;
          float *fb = p_b->features;
          printf("gpu se_kernel: i=%d,j=%d: d2_color is %f, d2_c_thres is %f,", i,j,d2_color, d2_c_thres );
          printf("color feature i == 0, a=(%f,%f,%f,%f,%f), b=(%f,%f,%f,%f,%f)\n",
                  fa[0], fa[1], fa[2], fa[3], fa[4], fb[0], fb[1], fb[2], fb[3], fb[4]);
                  }*/

        if(d2_color<d2_c_thres){

          //#ifdef IS_GEOMETRIC_ONLY
          //float a = s2*exp(-d2/(2.0*l*l));

          //#else
        
          
          float k = s2*exp(-d2/(2.0*l*l));
          float ck = c_sigma*c_sigma*exp(-d2_color/(2.0*c_ell*c_ell));
          // float ck = 1;
#ifdef IS_USING_SEMANTICS              
          float sk = cvo_params->s_sigma*cvo_params->s_sigma*exp(-d2_semantic/(2.0*s_ell*s_ell));
#else
          float sk = 1;
#endif              
          float a = ck*k*sk * normal_ip;
          //#endif          

          //if (i == 1000 && j == 51)
          //  printf("se_kernel: i=%d,j=%d: d2_color is %f, d2_c_thres is %f,k is %f, ck is %f\n", i,j,d2_color, d2_c_thres, k, ck );
          // if ( i == 1000 )
          // printf("se_kernel: i==1000, j==%d: k is %f, ck is %f\n", j, k, ck );



          // concrrent access !
          if (a > cvo_params->sp_thres){

            //A_mat->mat[i * A_mat->cols + j] = a;
            A_mat->mat[i * A_mat->cols + num_inds] = a;
            A_mat->ind_row2col[i * A_mat->cols + num_inds] = ind_b;
            num_inds++;
            /*
            if (i == 1000) {
              printf("[se_kernel] cloud_x[%d] (%f, %f, %f), cloud_y[%d] (%f, %f, %f):  non_zero_A ind %d, value %f\n", j, p_a->x, p_a->y, p_a->z, 1000, p_b->x, p_b->y, p_b->z, num_inds,  a);
              
              }*/

              

            // if (i == 1000 ) {
            //  printf("[se_kernel] i == 1000: non_zero_A is at %d value %f\n", j, a);
            //}

          }// else {
          // A_mat->mat[i][num_inds] = 0;
          // A_mat->ind_row2col[i][num_inds] = -1;
          //}


            
        }
#endif        
      }


      
    }
    A_mat->nonzeros[i] = num_inds;
    # if __CUDA_ARCH__>=200
    //if (i == 1000)
    //printf("se_kernel: i==1000: nonzeros is %d \n", num_inds );

    #endif  
    //delete mat_inds;
    
  }

  __device__
  float mahananobis_distance( const CvoPoint & p_a,const  CvoPoint &p_b) {
    const Eigen::Matrix3f cov_a = Eigen::Map<const Eigen::Matrix3f>(p_a.covariance);
    const Eigen::Matrix3f cov_b = Eigen::Map<const Eigen::Matrix3f>(p_b.covariance);
    Eigen::Matrix3f cov = cov_a + cov_b;

    Eigen::Vector3f a;
    a << p_a.x,p_a.y, p_a.z;
    Eigen::Vector3f b;
    b << p_b.x, p_b.y, p_b.z;

    Eigen::Vector3f dist = a-b;
    //float cov_inv[9];
    //Eigen::Matrix3f cov_curr_inv = Eigen::Map<Eigen::Matrix3f>(cov_inv);
    
    Eigen::Matrix3f cov_curr_inv = Inverse(cov) ;



    return (dist.transpose() * cov_curr_inv * dist).value();

    
  }
  __device__
  float eigenvalue_distance( const CvoPoint & p_a,const  CvoPoint &p_b) {
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

    float e_value = e_value_min_sum;

    if (e_value > 4.0) e_value = 4.0;
    if (e_value < 0.1) e_value = 0.1;

    return squared_dist(p_a, p_b) / e_value;
    //return (dist.transpose() * cov_curr_inv * dist).value();

    
  }

  __global__
  void fill_in_A_mat_gpu_dense_mat_kernel(// input
                                          const CvoParams * cvo_params,
                                          CvoPoint * points_a,
                                          int a_size,
                                          CvoPoint * points_b,
                                          int b_size,
                                          float ell_config,
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

    // point a in the first point cloud
    CvoPoint * p_a =  &points_a[i];
       
    float d2_thres = -2.0*log(sp_thres/sigma_square);
    float d2_c_thres = -2.0*c_ell_square*log(sp_thres/c_sigma_square);

#ifdef IS_USING_SEMANTICS
    float * label_a = p_a ->label_distribution;
#endif

    unsigned int num_inds = 0;
    for (int j = 0; j < b_size ; j++) {
      int ind_b = j;
      if (num_inds == CVO_POINT_NEIGHBORS) break;
      CvoPoint * p_b = &points_b[ind_b];
      float d2 = mahananobis_distance(*p_a, *p_b);
      //float d2 = eigenvalue_distance(*p_a, *p_b);
      //if (i < 100) {
      //  float d2_l2 = squared_dist(*p_a, *p_b) / 0.09;
      ///  printf("i==%d, ell is %f, mahananobis_distance is %f, while l2_dist is %f\n", i,ell_config, d2, d2_l2 );
      //}


      
      if(d2<d2_thres){

#ifdef IS_GEOMETRIC_ONLY
        float a = sigma_square*exp(-d2/2.0);
        if (a > cvo_params->sp_thres){
          A_mat->mat[i * A_mat->cols + num_inds] = a;
          A_mat->ind_row2col[i * A_mat->cols + num_inds] = ind_b;
          num_inds++;
        }

#else
        float d2_color = squared_dist<float>(p_a->features, p_b->features, FEATURE_DIMENSIONS);

#ifdef IS_USING_SEMANTICS            
        float d2_semantic = squared_dist<float>(p_a->label_distribution, p_b->label_distribution, NUM_CLASSES);
#endif

        if(d2_color<d2_c_thres){
          float k = sigma_square*exp(-d2/(2.0));
          float ck = c_sigma_square*exp(-d2_color/(2.0*c_ell_square));
#ifdef IS_USING_SEMANTICS              
          float sk = cvo_params->s_sigma*cvo_params->s_sigma*exp(-d2_semantic/(2.0*s_ell_square));
#else
          float sk = 1;
#endif              
          float a = ck*k*sk;

          if (a > cvo_params->sp_thres){

            A_mat->mat[i * A_mat->cols + num_inds] = a;
            A_mat->ind_row2col[i * A_mat->cols + num_inds] = ind_b;
            num_inds++;
          }
        }
#endif        
      }
    }
    A_mat->nonzeros[i] = num_inds;
  }


  

  __global__
  void fill_in_A_mat_gpu_range_ell(const CvoParams * cvo_params,
                                   //SquareExpParams * se_params,
                                   CvoPoint * points_a,
                                   int a_size,
                                   CvoPoint * points_b,
                                   int b_size,
                                   float ell,
                                   // output
                                   SparseKernelMat * A_mat // the kernel matrix!
                                   ) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > a_size - 1)
      return;

    float sigma2= cvo_params->sigma * cvo_params->sigma;
    float c2 = cvo_params->c_ell * cvo_params->c_ell;
    float c_sigma2 = cvo_params->c_sigma * cvo_params->c_sigma;
    float s_ell = cvo_params->s_ell;

    //Eigen::VectorXf feature_a = feature_a_gpu->row(i).transpose();
    CvoPoint * p_a =  &points_a[i];
    float a_to_sensor = sqrtf(p_a->x * p_a->x + p_a->y * p_a->y + p_a->z * p_a->z);
    float l = compute_range_ell(ell, a_to_sensor , 1, 80 );
    if (a_to_sensor > 40 ) {
      //printf("ell: from %f to %f, curr_dist_to_sensor is %f\n", ell, l,a_to_sensor);
      
    }
       
    // convert k threshold to d2 threshold (so that we only need to calculate k when needed)
    float d2_thres = -2.0*l*l*log(cvo_params->sp_thres/sigma2);
    float d2_c_thres = -2.0*c2*log(cvo_params->sp_thres/cvo_params->c_sigma/cvo_params->c_sigma);

    //float feature_a[5] = {(float)p_a->r, (float)p_a->g, (float)p_a->b,  p_a->gradient[0], p_a->gradient[1]  };
#ifdef IS_USING_SEMANTICS
    float * label_a = p_a ->label_distribution;
#endif

    //int * mat_inds = new int [kd_tree_max_leafIf they all have the same size, tha];
    unsigned int num_inds = 0;
    //#ifdef IS_USING_KDTREE
    //for (int j = 0; j < CVO_POINT_NEIGHBORS ; j++) {
      //int ind_b = kdtree_inds[i * CVO_POINT_NEIGHBORS  + j];
      //#else      
    for (int j = 0; j < b_size ; j++) {
      int ind_b = j;
      if (num_inds == CVO_POINT_NEIGHBORS) break;
      //#endif
      //A_mat->mat[i * A_mat->cols + ind_b] = 0;
      //A_mat->ind_row2col[i * A_mat->cols + ind_b] = -1;

      //float d2 = (cloud_y_gpu[ind_b] - cloud_x_gpu[i]).squaredNorm();
      // d2 = (x-y)^2
      CvoPoint * p_b = &points_b[ind_b];
      float d2 = (squared_dist( *p_b ,*p_a ));
      float normal_ip = 1;
#ifdef IS_USING_NORMALS
      normal_ip = fabs(p_a->normal[0] * p_b->normal[0] +
                       p_a->normal[1] * p_b->normal[1] +
                       p_a->normal[2] * p_b->normal[2]) * 0.81 * 0.01;
      //   printf("normal ip is %f between (%f,%f,%f) and (%f,%f,%f)\n", normal_ip,
      //       p_a->normal[0], p_a->normal[1], p_a->normal[2],
      //       p_b->normal[0], p_b->normal[1], p_b->normal[2]);
#endif      
        

      
      /*
      if ( i == 1000 && j== 1074) {
        CvoPoint * pb = points_b + j;
        printf("gpu se_kernel: i==%d,j==%d: d2 is %f, d2_thres is %f, point_a (%f, %f, %f), point_b: (%f, %f, %f)\n", i, j,d2, d2_thres,
              p_a->x, p_a->y, p_a->z,
              pb->x, pb->y,  pb->z );
      } */
      
      if(d2<d2_thres  ){

#ifdef IS_GEOMETRIC_ONLY
        float a = sigma2*exp(-d2/(2.0*l*l)) * normal_ip;
        //       printf("a is %f, normal_ip is %f, final_a is %f\n", s2*exp(-d2/(2.0*l*l)), normal_ip, a );
        if (a > cvo_params->sp_thres){
          A_mat->mat[i * A_mat->cols + num_inds] = a;
          A_mat->ind_row2col[i * A_mat->cols + num_inds] = ind_b;
          num_inds++;
        }

#else

        //float feature_b[5] = {(float)p_a->r, (float)p_a->g, (float)p_a->b,  p_a->gradient[0], p_a->gradient[1]  };
        float d2_color = squared_dist<float>(p_a->features, p_b->features, FEATURE_DIMENSIONS);

#ifdef IS_USING_SEMANTICS            
        float d2_semantic = squared_dist<float>(p_a->label_distribution, p_b->label_distribution, NUM_CLASSES);
#endif


        /*
        if (i == 1000 && j==1074) {
          float * fa = p_a->features;
          float *fb = p_b->features;
          printf("gpu se_kernel: i=%d,j=%d: d2_color is %f, d2_c_thres is %f,", i,j,d2_color, d2_c_thres );
          printf("color feature i == 0, a=(%f,%f,%f,%f,%f), b=(%f,%f,%f,%f,%f)\n",
                  fa[0], fa[1], fa[2], fa[3], fa[4], fb[0], fb[1], fb[2], fb[3], fb[4]);
                  }*/

        if(d2_color<d2_c_thres){

          //#ifdef IS_GEOMETRIC_ONLY
          //float a = s2*exp(-d2/(2.0*l*l));

          //#else
        
          
          float k = sigma2*exp(-d2/(2.0*l*l));
          float ck = c_sigma2*exp(-d2_color/(2.0*c2 ));
          // float ck = 1;
#ifdef IS_USING_SEMANTICS              
          float sk = cvo_params->s_sigma*cvo_params->s_sigma*exp(-d2_semantic/(2.0*s_ell*s_ell));
#else
          float sk = 1;
#endif              
          float a = ck*k*sk*normal_ip;
          //#endif          

          //if (i == 1000 && j == 51)
          //  printf("se_kernel: i=%d,j=%d: d2_color is %f, d2_c_thres is %f,k is %f, ck is %f\n", i,j,d2_color, d2_c_thres, k, ck );
          // if ( i == 1000 )
          // printf("se_kernel: i==1000, j==%d: k is %f, ck is %f\n", j, k, ck );



          // concrrent access !
          if (a > cvo_params->sp_thres){

            //A_mat->mat[i * A_mat->cols + j] = a;
            A_mat->mat[i * A_mat->cols + num_inds] = a;
            A_mat->ind_row2col[i * A_mat->cols + num_inds] = ind_b;
            num_inds++;
            /*
            if (i == 1000) {
              printf("[se_kernel] cloud_x[%d] (%f, %f, %f), cloud_y[%d] (%f, %f, %f):  non_zero_A ind %d, value %f\n", j, p_a->x, p_a->y, p_a->z, 1000, p_b->x, p_b->y, p_b->z, num_inds,  a);
              
              }*/

              

            // if (i == 1000 ) {
            //  printf("[se_kernel] i == 1000: non_zero_A is at %d value %f\n", j, a);
            //}

          }// else {
          // A_mat->mat[i][num_inds] = 0;
          // A_mat->ind_row2col[i][num_inds] = -1;
          //}


            
        }
#endif        
      }


      
    }
    A_mat->nonzeros[i] = num_inds;
    # if __CUDA_ARCH__>=200
    //if (i == 1000)
    //printf("se_kernel: i==1000: nonzeros is %d \n", num_inds );

    #endif  
    //delete mat_inds;
    
  }



  static
  void se_kernel(// input
                const CvoParams * params_gpu,
                std::shared_ptr<CvoPointCloudGPU> points_fixed,
                std::shared_ptr<CvoPointCloudGPU> points_moving,
                float ell,
                // output
                SparseKernelMat * A_mat,
                SparseKernelMat * A_mat_gpu
                )  {

    auto start = chrono::system_clock::now();
    /*
    thrust::device_vector<int> indices;
    kdtree->NearestKSearch(points_moving, CVO_POINT_NEIGHBORS, indices );
    ind_device = thrust::raw_pointer_cast(indices.data() );
    if (debug_print) {
      std::cout<<"kdtree search result size is "<<indices.size();
      std::cout<<"\n";
    }
    auto end = chrono::system_clock::now();
    chrono::duration<double> t_kdtree_search = end-start;
    std::cout<<"t kdtree search in se_kernel is "<<t_kdtree_search.count()<<std::endl;
    */
    int fixed_size = points_fixed->points.size();
    CvoPoint * points_fixed_raw = thrust::raw_pointer_cast (  points_fixed->points.data() );
    CvoPoint * points_moving_raw = thrust::raw_pointer_cast( points_moving->points.data() );
    
#ifdef IS_USING_COVARIANCE   
    fill_in_A_mat_gpu_dense_mat_kernel<<<(points_moving->size() / CUDA_BLOCK_SIZE)+1, CUDA_BLOCK_SIZE  >>>(// input
      params_gpu,
      points_fixed_raw,
      fixed_size,
      points_moving_raw,
      points_moving->points.size(),
      ell,
      // output
      A_mat_gpu // the inner product matrix!
      );
#else 
    fill_in_A_mat_gpu_range_ell<<<(points_moving->size() / CUDA_BLOCK_SIZE)+1, CUDA_BLOCK_SIZE  >>>(params_gpu,
      points_fixed_raw,
      fixed_size,
      points_moving_raw,
      points_moving->points.size(),
      ell,
      // output
      A_mat_gpu // the kernel matrix!
      );
    
#endif    
    compute_nonzeros(A_mat);
  }


  __global__ void compute_flow_gpu_no_eigen(const CvoParams * cvo_params,
                                            CvoPoint * cloud_x, CvoPoint * cloud_y,
                                            SparseKernelMat * A,
                                            // outputs: thrust vectors
                                            Eigen::Vector3d * omega_all_gpu,
                                            Eigen::Vector3d * v_all_gpu
                                            ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > A->rows - 1)
      return;
    int A_rows = A->rows;
    int A_cols = A->cols;
    float * Ai = A->mat + i * A->cols;
    
    CvoPoint * px = &cloud_x[i];
    Eigen::Vector3f px_eig;
    px_eig<< px->x , px->y, px->z;
    float px_arr[3] = {px->x, px->y, px->z};
    Eigen::Vector3f omega_i = Eigen::Vector3f::Zero();
    Eigen::Vector3f v_i = Eigen::Vector3f::Zero();
    float dl_i = 0;
    for (int j = 0; j < A_cols; j++) {
      int idx = A->ind_row2col[i*A_cols+j];
      if (idx == -1) break;
      
      CvoPoint * py = &cloud_y[idx];
      float py_arr[3] = {py->x, py->y, py->z};
      Eigen::Vector3f py_eig;
      py_eig << py->x, py->y, py->z;
      
      Eigen::Vector3f cross_xy_j = px_eig.cross(py_eig) ;
      Eigen::Vector3f diff_yx_j = py_eig - px_eig;
      float sum_diff_yx_2_j = diff_yx_j.squaredNorm();
      
      omega_i = omega_i + cross_xy_j *  *(Ai + j );
      v_i = v_i + diff_yx_j *  *(Ai + j);
    }

    Eigen::Vector3d & omega_i_eig = omega_all_gpu[i];
    omega_i_eig = (omega_i / cvo_params->c ).cast<double>();
    Eigen::Vector3d & v_i_eig = v_all_gpu[i];
    v_i_eig = (v_i /  cvo_params->d).cast<double>();
  }

  void compute_flow(CvoState * cvo_state, const CvoParams * params_gpu,
                    Eigen::Vector3f * omega, Eigen::Vector3f * v)  {

    if (debug_print ) {
      std::cout<<"nonzeros in A "<<nonzeros(&cvo_state->A_host)<<std::endl;
      //std::cout<<"time for se_kernel is "<<std::chrono::duration_cast<std::chrono::milliseconds>((end- start)).count()<<std::endl;
      std::cout<<"A rows is "<<cvo_state->A_host.rows<<", A cols is "<<cvo_state->A_host.cols<<std::endl;
    }
    
    auto start = chrono::system_clock::now();
    compute_flow_gpu_no_eigen<<<cvo_state->A_host.rows / CUDA_BLOCK_SIZE + 1 ,CUDA_BLOCK_SIZE>>>(params_gpu,
                                                                                                 thrust::raw_pointer_cast(cvo_state->cloud_x_gpu->points.data()   ),
                                                                                                 thrust::raw_pointer_cast(cvo_state->cloud_y_gpu->points.data()   ),
                                                                                                 cvo_state->A, 
                                                                                                 thrust::raw_pointer_cast(cvo_state->omega_gpu.data()  ),
                                                                                                 thrust::raw_pointer_cast(cvo_state->v_gpu.data() ));
                                                          ;
    //cudaDeviceSynchronize();
    if (debug_print)
      printf("finsih compute_flow_gpu_no_eigen\n");
    auto end = chrono::system_clock::now();
    //std::cout<<"time for compute_gradient is "<<std::chrono::duration_cast<std::chrono::milliseconds>((end- start)).count()<<std::endl;
    
    start = chrono::system_clock::now();
    // update them to class-wide variables
    thrust::plus<double> plus_double;
    thrust::plus<Eigen::Vector3d> plus_vector;
    *omega = (thrust::reduce(cvo_state->omega_gpu.begin(), cvo_state->omega_gpu.end())).cast<float>();
    *v = (thrust::reduce(cvo_state->v_gpu.begin(), cvo_state->v_gpu.end())).cast<float>();

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
    if (j == 1000) {
      printf("j==1000, cloud+yi is (%f,%f,%f), xiz=(%f %f %f), xi2z=(%f %f %f), xi3z=(%f %f %f), xi4z=(%f %f %f), normxiz2=%f, xiz_dot_xi2z=%f, epsil_const=%f\n ",
            cloud_y[j].x , cloud_y[j].y, cloud_y[j].z,
            xiz[j](0), xiz[j](1), xiz[j](2),
            xi2z[j](0),xi2z[j](1),xi2z[j](2),
            xi3z[j](0),xi3z[j](1),xi3z[j](2),
            xi4z[j](0),xi4z[j](1),xi4z[j](2),
            normxiz2[j], xiz_dot_xi2z[j], epsil_const[j]
            );
      
    }
    */
  }
  __global__ void compute_step_size_poly_coeff(float temp_coef,
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
    for (int j = 0; j < CVO_POINT_NEIGHBORS; j++) {
      int idx = A->ind_row2col[i * A_cols + j];
      if (idx == -1) break;
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
      float A_ij = A->mat[i * A_cols + j];
      // eq (34)
      B[i] += double(A_ij * beta_ij);
      C[i] += double(A_ij * (gamma_ij+beta_ij*beta_ij/2.0));
      D[i] += double(A_ij * (delta_ij+beta_ij*gamma_ij + beta_ij*beta_ij*beta_ij/6.0));
      E[i] += double(A_ij * (epsil_ij+beta_ij*delta_ij+1/2.0*beta_ij*beta_ij*gamma_ij \
                             + 1/2.0*gamma_ij*gamma_ij + 1/24.0*beta_ij*beta_ij*beta_ij*beta_ij));
      /*
        if (i == 1074 && j == 1000) {
        printf("x==1000, y==1074, Aij=%f, beta_ij=%f, gamma_ij=%f, delta_ij=%f, epsil_ij=%f\n",
        A_ij, beta_ij, gamma_ij, delta_ij, epsil_ij);
        
        }*/
    }
    
  }

__global__ void compute_step_size_poly_coeff_location_dependent_ell(float ell,
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
#ifdef IS_USING_RANGE_ELL    
    temp_ell = compute_range_ell(ell,d2_sqrt, 1, 80 );
#endif    
    
    for (int j = 0; j < CVO_POINT_NEIGHBORS; j++) {
      int idx = A->ind_row2col[i * A_cols + j];
      if (idx == -1) break;
#ifdef IS_USING_COVARIANCE
      temp_ell = (cloud_x[i].cov_eigenvalues[0] + cloud_y[idx].cov_eigenvalues[0])/10 ;
      if (temp_ell > 1.0) temp_ell = 1.0;
      if (temp_ell < 0.1) temp_ell = 0.1;
      if (i == 0) printf("temp ell in step size is %f, e_value_a is %f, e_value_b is %f\n", temp_ell, cloud_x[i].cov_eigenvalues[2], cloud_y[idx].cov_eigenvalues[2]  );
      //temp_ell = compute_range_ell(ell,d2_sqrt, 1, 80 );
#endif      
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

      float A_ij = A->mat[i * A_cols + j];
      // eq (34)
      B[i] += double(A_ij * beta_ij);
      C[i] += double(A_ij * (gamma_ij+beta_ij*beta_ij/2.0));
      D[i] += double(A_ij * (delta_ij+beta_ij*gamma_ij + beta_ij*beta_ij*beta_ij/6.0));
      E[i] += double(A_ij * (epsil_ij+beta_ij*delta_ij+1/2.0*beta_ij*beta_ij*gamma_ij\
                          + 1/2.0*gamma_ij*gamma_ij + 1/24.0*beta_ij*beta_ij*beta_ij*beta_ij));
      /*
    if (i == 1074 && j == 1000) {
        printf("x==1000, y==1074, Aij=%f, beta_ij=%f, gamma_ij=%f, delta_ij=%f, epsil_ij=%f\n",
              A_ij, beta_ij, gamma_ij, delta_ij, epsil_ij);
        
              }*/
    }
    
  }

  
  void compute_step_size(CvoState * cvo_state, const CvoParams * params) {
    compute_step_size_xi<<<cvo_state->num_moving / CUDA_BLOCK_SIZE + 1, CUDA_BLOCK_SIZE>>>
      (cvo_state->omega, cvo_state->v,
       thrust::raw_pointer_cast( cvo_state->cloud_y_gpu->points.data()  ), cvo_state->num_moving,
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
    compute_step_size_poly_coeff_location_dependent_ell<<<cvo_state->num_fixed / CUDA_BLOCK_SIZE + 1, CUDA_BLOCK_SIZE>>>
      ( cvo_state->ell, cvo_state->num_fixed, cvo_state->A,
        thrust::raw_pointer_cast( cvo_state->cloud_x_gpu->points.data()  ),
        thrust::raw_pointer_cast( cvo_state->cloud_y_gpu->points.data() ),
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
    Eigen::Vector4f p_coef(4);
    p_coef << 4.0*float(E),3.0*float(D),2.0*float(C),float(B);
    if (debug_print)
      std::cout<<"BCDE is "<<p_coef.transpose()<<std::endl;
    
    // solve polynomial roots
    //Eigen::VectorXcf rc = poly_solver(p_coef);
    Eigen::Vector3cf rc = poly_solver_order3(p_coef);
    
    
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

    //std::cout<<"ratio is "<<indicator_end_sum / indicator_start_sum<<std::endl;
    //static std::ofstream ratio("ip_ratio.txt",std::ofstream::out | std::ofstream::app );
    // start queue is full, start building the end queue
    if (indicator_start_queue.size() >= queue_len && indicator_end_queue.size() >= queue_len){
      // ratio<< indicator_end_sum / indicator_start_sum<<"\n"<<std::flush;
      //std::cout<<"ip ratio is "<<indicator_end_sum / indicator_start_sum<<" from "<<indicator_end_sum<<" over "<<indicator_start_sum<<std::endl;
      // check if criteria for decreasing legnthscale is satisfied
      if(indicator_end_sum / indicator_start_sum > 1 - params.indicator_stable_threshold
         &&
         indicator_end_sum / indicator_start_sum  < 1 + params.indicator_stable_threshold){
        decrease = true;
        std::queue<float> empty;
        std::swap( indicator_start_queue, indicator_end_queue );

        std::swap( indicator_end_queue, empty );
        
        std::queue<float> empty2;
        std::swap( indicator_start_queue, empty2 );
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
  
  int CvoGPU::align(const CvoPointCloud& source_points,
                    const CvoPointCloud& target_points,
                    const Eigen::Matrix4f & init_guess_transform,
                    // outputs
                    Eigen::Ref<Eigen::Matrix4f> transform,
                    double *registration_seconds) const {


    Mat33f R = init_guess_transform.block<3,3>(0,0);
    Vec3f T= init_guess_transform.block<3,1>(0,3);

    std::ofstream ell_file("ell_history.txt");
    std::ofstream dist_change_file("dist_change_history.txt");
    std::ofstream transform_file("transformation_history.txt");
    std::ofstream step_file("step_history.txt");
    std::ofstream inner_product_file("inner_product_history.txt");
    
    std::cout<<"[align] convert points to gpu\n"<<std::flush;
    CvoPointCloudGPU::SharedPtr source_gpu = CvoPointCloud_to_gpu(source_points);
    CvoPointCloudGPU::SharedPtr target_gpu = CvoPointCloud_to_gpu(target_points);

    assert(source_gpu != nullptr && target_gpu != nullptr);
    std::cout<<"Pc non-empty\n";
    CvoState cvo_state(source_gpu, target_gpu, params);
    std::cout<<"construct new cvo state..., init ell is "<<cvo_state.ell<<std::endl;

    int num_moving = cvo_state.num_moving;
    int num_fixed = cvo_state.num_fixed;
    
    int ret = 0;

    int iter = params.MAX_ITER;
    Eigen::Vector3f omega, v;
    auto start_all = chrono::system_clock::now();
    auto start = chrono::system_clock::now();
    auto end = chrono::system_clock::now();
    chrono::duration<double> t_transform_pcd = chrono::duration<double>::zero();
    chrono::duration<double> t_compute_flow = chrono::duration<double>::zero();
    chrono::duration<double> t_compute_step = chrono::duration<double>::zero();
    std::cout<<"Start iteration, init transform is \n";
    std::cout<<init_guess_transform<<std::endl;
    std::cout<<"Max iter is "<<params.MAX_ITER<<std::endl;

    std::queue<float> indicator_start_queue;
    std::queue<float> indicator_end_queue;
    float indicator_start_sum = 0;
    float indicator_end_sum = 0;

    int k = 0;
    for(; k<params.MAX_ITER; k++){
      if (debug_print) printf("new iteration %d, ell is %f\n", k, cvo_state.ell);
      cvo_state.reset_state_at_new_iter();
      if (debug_print) printf("just reset A mat\n");
      
      // update transformation matrix to CvoState
      update_tf(R, T, &cvo_state, transform);
      if (is_logging) {
        Eigen::Matrix4f Tmat = transform;
        transform_file << Tmat(0,0) <<" "<< Tmat(0,1) <<" "<< Tmat(0,2) <<" "<< Tmat(0,3) <<" "
                       << Tmat(1,0) <<" "<< Tmat(1,1) <<" "<< Tmat(1,2) <<" "<< Tmat(1,3) <<" "
                       << Tmat(2,0) <<" "<< Tmat(2,1) <<" "<< Tmat(2,2) <<" "<< Tmat(2,3) <<"\n"<< std::flush;
      }

      // transform point cloud
      start = chrono::system_clock::now();
      transform_pointcloud_thrust(cvo_state.cloud_y_gpu_init, cvo_state.cloud_y_gpu,
                                  cvo_state.R_gpu, cvo_state.T_gpu ); 
      end = std::chrono::system_clock::now();
      t_transform_pcd += (end - start); 


      // update the inner product matrix
      start = chrono::system_clock::now();
      se_kernel(params_gpu, cvo_state.cloud_x_gpu, cvo_state.cloud_y_gpu ,
                cvo_state.ell, 
                &cvo_state.A_host, cvo_state.A);
      cudaDeviceSynchronize();
      auto end = chrono::system_clock::now();
      if (debug_print ) {
        std::cout<<"nonzeros in A "<<nonzeros(&cvo_state.A_host)<<std::endl;
        std::cout<<"time for se_kernel is "<<std::chrono::duration_cast<std::chrono::milliseconds>((end- start)).count()<<std::endl;
        std::cout<<"A rows is "<<cvo_state.A_host.rows<<", A cols is "<<cvo_state.A_host.cols<<std::endl;
        
      }

      // compute omega and v
      start = chrono::system_clock::now();
      compute_flow(&cvo_state, params_gpu, &omega, &v);
      if (debug_print) std::cout<<"iter "<<k<< "omega: \n"<<omega.transpose()<<"\nv: \n"<<v.transpose()<<std::endl;
      if (k == 0) printf("nonzeros in A is %d\n", cvo_state.A_host.nonzero_sum);
      end = std::chrono::system_clock::now();
      t_compute_flow += (end - start);

      // compute indicator and change lenthscale if needed
      //float indicator = (double)cvo_state.A_host.nonzero_sum / (double)source_points.num_points() / (double) target_points.num_points();

      start = chrono::system_clock::now();
      compute_step_size(&cvo_state, &params);
      end = std::chrono::system_clock::now();
      t_compute_step += (end -start);
      
      // stop if the step size is too small
      if (debug_print) printf("copy gradient to cpu...");
      cudaMemcpy(omega.data(), cvo_state.omega->data(), sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost);
      cudaMemcpy(v.data(), cvo_state.v->data(), sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost);
      if(omega.cast<double>().norm()<params.eps && v.cast<double>().norm()<params.eps){
        iter = k;
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
      Eigen::AngleAxisd R_angle(R.cast<double>() * dR);
      R = R_angle.toRotationMatrix().cast<float>(); // re-orthogonalization

      // reduce ell, if the se3 distance is smaller than eps2, break
      float dist_this_iter = dist_se3(dR.cast<float>(),dT.cast<float>());
      if (debug_print)  std::cout<<"dist: "<<dist_this_iter <<std::endl<<"check bounds....\n";
      float ip_curr = (float)((double)cvo_state.A_host.nonzero_sum / (double)source_points.num_points() / (double) target_points.num_points());
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
     
      if(dist_this_iter<params.eps_2){
        iter = k;
        std::cout<<"break: dist: "<<dist_this_iter<<std::endl;
        break;
      }

      if (k>params.ell_decay_start && need_decay_ell  ) {
        cvo_state.ell = cvo_state.ell * params.ell_decay_rate;
        if (cvo_state.ell < params.ell_min)
          cvo_state.ell = params.ell_min;
      }
      
      if(debug_print) printf("end of iteration \n\n\n");

    }
    auto end_all = chrono::system_clock::now();
    chrono::duration<double> t_all = chrono::duration<double>::zero();
    t_all = end_all - start_all;
    std::cout<<"cvo # of iterations is "<<iter<<std::endl;
    std::cout<<"t_transform_pcd is "<<t_transform_pcd.count()<<"\n";
    std::cout<<"t_compute_flow is "<<t_compute_flow.count()<<"\n";
    std::cout<<"t_compute_step is "<<t_compute_step.count()<<"\n"<<std::flush;
    std::cout<<"t_all is "<<t_all.count()<<"\n"<<std::flush;
    std::cout<<"non adaptive cvo ends. final ell is "<<cvo_state.ell<<", final iteration is "<<k
             <<"MAX_ITER is "<<params.MAX_ITER<<std::endl;
    if (registration_seconds)
      *registration_seconds = t_all.count();

    update_tf(R, T, &cvo_state, transform);

    //if (is_logging) {
    ell_file.close();
    dist_change_file.close();
    transform_file.close();
    step_file.close();
    inner_product_file.close();
    //}
    
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
     const float l = params.ell_min;
    // const float l = params.ell_init;

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
              // ck = 1;
#ifdef IS_USING_SEMANTICS              
              sk = params.s_sigma*params.s_sigma*exp(-d2_semantic/(2.0*params.s_ell*params.s_ell));
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
    //return float(A_mat.sum())/float(A_mat.nonZeros());
    return A_mat.sum()/fixed_positions.size()*1e5/moving_positions.size() ;
    //return float(A_mat.nonZeros()) / float(fixed_positions.size()) / float(moving_positions.size() ) * 100 ;
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