#include "cvo/CvoGPU_impl.cuh"
#include "cvo/SparseKernelMat.hpp"
#include "utils/PointSegmentedDistribution.hpp"
#include "cvo/Association.hpp"
#include "utils/CvoPoint.hpp"
#include "cvo/gpu_utils.cuh"
#include "cvo/CudaTypes.cuh"
#include "cupointcloud/point_types.h"
#include "cupointcloud/cupointcloud.h"
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


extern template struct pcl::PointSegmentedDistribution<FEATURE_DIMENSIONS, NUM_CLASSES>;

namespace cvo {


  struct transform_point_R_T : public thrust::unary_function<CvoPoint,CvoPoint>
  {
    const Mat33f * R;
    const Vec3f * T;
    const bool update_normal_and_cov;

    transform_point_R_T(const Mat33f * R_gpu, const Vec3f * T_gpu,
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


  struct transform_point_pose_vec : public thrust::unary_function<CvoPoint,CvoPoint>
  {
    float * T_12_row_gpu;
    //Eigen::Matrix<float, 3, 4, Eigen::RowMajor> T;
    //Eigen::MatrixXf T;
    const bool update_normal_and_cov;

    transform_point_pose_vec(float * pose_vec_row_gpu,
                             bool update_normal_and_covariance):
      //T(Eigen::Map<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>(pose_vec_row_gpu)),
      T_12_row_gpu(pose_vec_row_gpu),
      //T(3,4),
      update_normal_and_cov(update_normal_and_covariance){
      
      //auto p = pose_vec_row_gpu;
      /*
      T << pose_vec_row_gpu[0], pose_vec_row_gpu[1], pose_vec_row_gpu[2], pose_vec_row_gpu[3],
        pose_vec_row_gpu [4], pose_vec_row_gpu[5], pose_vec_row_gpu[6], pose_vec_row_gpu[7],
        pose_vec_row_gpu[8], pose_vec_row_gpu[9], pose_vec_row_gpu[10], pose_vec_row_gpu[11];
      */
      
      
    }
    
    __host__ __device__
    CvoPoint operator()(const CvoPoint & p_init)
    {
      CvoPoint result(p_init);

      Eigen::Vector4f input;
      input << p_init.x, p_init.y, p_init.z, 1.0;

      Eigen::Matrix<float, 3, 4, Eigen::RowMajor> T =
      //auto T =
        Eigen::Map<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>(T_12_row_gpu);
      //Eigen::Map<Eigen::Matrix<const float, 3, 4, Eigen::RowMajor>> T(T_12_row_gpu);



      Eigen::Vector3f trans = T * input;
      result.x = trans(0);
      result.y = trans(1);
      result.z = trans(2);
      //printf("transform point: T is \n %f %f %f %f\n %f %f %f %f\n%f %f %f %f\nbefore T: p=(%f,%f,%f), after T: p=(%f,%f,%f)\n",
      //       T(0,0), T(0,1), T(0,2), T(0,3), T(1,0), T(1,1), T(1,2), T(1,3), T(2,0), T(2,1), T(2,2), T(2,3),
      //       input(0), input(1), input(2), result.x, result.y, result.z);      
      
      if (update_normal_and_cov) {
        Eigen::Vector4f input_normal;
        input_normal << p_init.normal[0], p_init.normal[1], p_init.normal[2], 1.0;
        Eigen::Vector3f trans_normal = T * input_normal;
        result.normal[0] = trans_normal(0);
        result.normal[1] = trans_normal(1);
        result.normal[2] = trans_normal(2);



        Eigen::Matrix3f point_cov;
        point_cov << p_init.covariance[0], p_init.covariance[1], p_init.covariance[2],
          p_init.covariance[3], p_init.covariance[4], p_init.covariance[5],
          p_init.covariance[6], p_init.covariance[7], p_init.covariance[8];

        Eigen::Matrix<float,3,3,Eigen::RowMajor> R = T.block<3,3>(0,0);
      
        point_cov = ((R) * point_cov * (R.transpose())).eval();
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
                       transform_point_R_T(R_gpu, T_gpu, update_normal_and_cov));
  
  }

  void  transform_pointcloud_thrust(std::shared_ptr<CvoPointCloudGPU> init_cloud,
                                    std::shared_ptr<CvoPointCloudGPU> transformed_cloud,
                                    float * T12_row_gpu_,
                                    bool update_normal_and_cov
                                    ) {

    thrust::transform( init_cloud->begin(), init_cloud->end(),  transformed_cloud->begin(),
                       transform_point_pose_vec(T12_row_gpu_, update_normal_and_cov));
    // transform_point_R_T(R_gpu, T_gpu, update_normal_and_cov));
  
  }
  

  void  transform_pointcloud_thrust(thrust::device_vector<CvoPoint> & init_cloud,
                                    thrust::device_vector<CvoPoint> & transformed_cloud,
                                    float * T_12_row_gpu,
                                    bool update_normal_and_cov
                                    ) {

    thrust::transform( init_cloud.begin(),
                       init_cloud.end(),
                       transformed_cloud.begin(), 
                       transform_point_pose_vec(T_12_row_gpu, update_normal_and_cov));
  }
  

  
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

  void CvoPointCloud_to_gpu(const CvoPointCloud& cvo_cloud, thrust::device_vector<CvoPoint> & output ) {
    int num_points = cvo_cloud.num_points();
    // const ArrayVec3f & positions = cvo_cloud.positions();
//    const Eigen::Matrix<float, Eigen::Dynamic, FEATURE_DIMENSIONS> & features = cvo_cloud.features();
    //const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> & normals = cvo_cloud.normals();
    // const Eigen::Matrix<float, Eigen::Dynamic, 2> & types = cvo_cloud.types();
//    auto & labels = cvo_cloud.labels();
    // set basic informations for pcl_cloud
    thrust::host_vector<CvoPoint> host_cloud(cvo_cloud.get_points());
    std::cout << "Feature Dimemsions is " << FEATURE_DIMENSIONS << "\n";
    int actual_num = 0;
    for(int i=0; i<num_points; ++i){

      if (FEATURE_DIMENSIONS == 5
//          && features.rows() == num_points &&
//          features.cols() == FEATURE_DIMENSIONS
          ) {
        (host_cloud)[i].r = (uint8_t)std::min(255.0, (cvo_cloud.feature_at(i)[0] * 255.0));
        (host_cloud)[i].g = (uint8_t)std::min(255.0, (cvo_cloud.feature_at(i)[1] * 255.0));
        (host_cloud)[i].b = (uint8_t)std::min(255.0, (cvo_cloud.feature_at(i)[2] * 255.0));
      }
      
      if (cvo_cloud.num_classes() > 0) {
        cvo_cloud.label_at(i).maxCoeff(&host_cloud[i].label);
//        labels.row(i).maxCoeff(&host_cloud[i].label);
      }
      
      //if (normals.rows() > 0 && normals.cols()>0) {
      //  for (int j = 0; j < 3; j++)
      //    host_cloud[i].normal[j] = normals(i,j);
      //}

      //if (cvo_cloud.covariance().size() > 0 )
      //  memcpy(host_cloud[i].covariance, cvo_cloud.covariance().data()+ i*9, sizeof(float)*9  );
      //if (cvo_cloud.eigenvalues().size() > 0 )
      //  memcpy(host_cloud[i].cov_eigenvalues, cvo_cloud.eigenvalues().data() + i*3, sizeof(float)*3);
      /*
      if (i == 1) {
        printf("Total %d, Raw input from pcl at 1000th: \n", num_points);
        std::cout<<"Before conversion: "<<positions[i].transpose()<<"\n";
        print_point(host_cloud[i]);
        }*/
      actual_num ++;
    }
    output = host_cloud;
  }

  
   
  CvoPointCloudGPU::SharedPtr CvoPointCloud_to_gpu(const CvoPointCloud & cvo_cloud ) {

    CvoPointCloudGPU::SharedPtr gpu_cloud(new CvoPointCloudGPU);
    CvoPointCloud_to_gpu(cvo_cloud, gpu_cloud->points);
    // thrust::host_vector<CvoPoint> host_cloud(cvo_cloud.get_points());
    // gpu_cloud->points = host_cloud;

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

  void pcl_PointCloud_to_gpu(const pcl::PointCloud<CvoPoint> & cvo_cloud, thrust::device_vector<CvoPoint> & output ) {
    int num_points = cvo_cloud.size();
    thrust::host_vector<CvoPoint> host_cloud;
    host_cloud.resize(num_points);

    //int actual_num = 0;
    for(int i=0; i<num_points; ++i){
      memcpy(&host_cloud[i], &cvo_cloud[i], sizeof(CvoPoint));
      //std::copy(&)
      /*
      (host_cloud)[i].x = cvo_cloud[i].x;
      (host_cloud)[i].y = cvo_cloud[i].y;
      (host_cloud)[i].z = cvo_cloud[i].z;
      (host_cloud)[i].r = cvo_cloud[i].r;
      (host_cloud)[i].g = cvo_cloud[i].g;
      (host_cloud)[i].b = cvo_cloud[i].b;


      ///memcpy(host_cloud[i].features, features.row(i).data(), FEATURE_DIMENSIONS * sizeof(float));
      for (int j = 0; j < FEATURE_DIMENSIONS; j++)
        host_cloud[i].features[j] = features(i,j);

      if (cvo_cloud.num_classes() > 0) {
        labels.row(i).maxCoeff(&host_cloud[i].label);
        for (int j = 0; j < cvo_cloud.num_classes(); j++)
          host_cloud[i].label_distribution[j] = labels(i,j);
      }
      
      if (normals.rows() > 0 && normals.cols()>0) {
        for (int j = 0; j < 3; j++)
          host_cloud[i].normal[j] = normals(i,j);
      }

      if (cvo_cloud.covariance().size() > 0 )
        memcpy(host_cloud[i].covariance, cvo_cloud.covariance().data()+ i*9, sizeof(float)*9  );
      if (cvo_cloud.eigenvalues().size() > 0 )
        memcpy(host_cloud[i].cov_eigenvalues, cvo_cloud.eigenvalues().data() + i*3, sizeof(float)*3);

      actual_num ++;
      if (i == 1000) {
        printf("Total %d, Raw input from pcl at 1000th: \n", num_points);
        std::cout<<"pcl: \n";
        print_point(cvo_cloud[i]);
        std::cout<<"host_cloud\n";
        print_point(host_cloud[i]);
      }
      
      */
    }
    output = host_cloud;
    //gpu_cloud->points = host_cloud;
    
  }
  

  CvoPointCloudGPU::SharedPtr pcl_PointCloud_to_gpu(const pcl::PointCloud<CvoPoint> & cvo_cloud ) {

    
    CvoPointCloudGPU::SharedPtr gpu_cloud(new CvoPointCloudGPU);
    
    pcl_PointCloud_to_gpu(cvo_cloud, gpu_cloud->points );

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


  
  void gpu_association_to_cpu(const SparseKernelMat & association_gpu,
                              Association & association_cpu,
                              int num_source,
                              int num_target,
                              int num_neighbors) {
    int rows = association_gpu.rows;
    int cols = num_neighbors == -1? association_gpu.cols : num_neighbors;

    //association_cpu.source_inliers.resize(rows, false);
    //association_cpu.target_inliers.resize(num_target, false);

    if (association_gpu.nonzero_sum == 0)
      return;
    
    thrust::device_ptr<float> inner_product_ptr(thrust::raw_pointer_cast(association_gpu.mat));
    thrust::device_vector<float> inner_product_gpu(inner_product_ptr, inner_product_ptr + rows *cols);
    thrust::host_vector<float> inner_product = inner_product_gpu;
    
    thrust::device_ptr<int> ind_row2col_ptr(thrust::raw_pointer_cast(association_gpu.ind_row2col));
    thrust::device_vector<int> ind_row2col_gpu(ind_row2col_ptr, ind_row2col_ptr + rows *cols);
    thrust::host_vector<int> ind_row2col = ind_row2col_gpu;
    
    thrust::device_ptr<unsigned int> nonzeros_ptr ( thrust::raw_pointer_cast(association_gpu.nonzeros));
    thrust::device_vector<unsigned int> nonzeros_gpu(nonzeros_ptr, nonzeros_ptr + rows);
    thrust::host_vector<unsigned int> nonzeros = nonzeros_gpu;

    ///association_cpu.pairs.reserve(association_gpu.nonzero_sum);
    association_cpu.pairs.resize(rows,cols);

    //std::cout<<"Construct the CPU association:\n";
    //tbb::mutex mutex;
    //tbb::parallel_for(int(0), rows, [&]( int i ){
    #pragma omp parallel for    
    for (int i = 0; i < rows; i++) {

  
      if (nonzeros[i] > 0) {
        //association_cpu.source_inliers[i] = true;
        association_cpu.source_inliers.push_back(i);
        
        for (int j = 0 ; j < cols; j++) {
          if (ind_row2col[i * cols + j] == -1) {
            break;
          }
          
          #pragma omp critical
          {
            int ind_target = ind_row2col[i*cols+j];
            //association_cpu.target_inliers[ind_target] = true;
            association_cpu.target_inliers.push_back(ind_target);
            association_cpu.pairs.insert(i, ind_target) = inner_product[i*cols+j];
            //if (i == 2349)
            //  std::cout<<"Association: i=="<<i<<", j=="<<ind_target<<", a=="<<inner_product[i*cols+j]<<std::endl;
          }
          
        }
      }

    }
    association_cpu.pairs.makeCompressed();
  
  }


  void find_nearby_source_points_cukdtree(//const CvoParams *cvo_params,
                                          std::shared_ptr<CvoPointCloudGPU> cloud_x_gpu,
                                          CuKdTree & kdtree_cloud_y,
                                          const Eigen::Matrix4f & transform_cpu_yf2xf,
                                          int num_neighbors,
                                          // output
                                          std::shared_ptr<CvoPointCloudGPU> cloud_x_gpu_transformed_kdtree,
                                          thrust::device_vector<int> & cukdtree_inds_results
                                          ) {
    cudaEvent_t cuda_start, cuda_stop;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);
    cudaEventRecord(cuda_start, 0);

    Eigen::Matrix3f * R_gpu_yf2xf;
    Eigen::Vector3f * T_gpu_yf2xf;
    cudaMalloc(&R_gpu_yf2xf, sizeof(Eigen::Matrix3f));
    cudaMalloc(&T_gpu_yf2xf, sizeof(Eigen::Vector3f));
    Eigen::Matrix3f R_cpu_yf2xf = transform_cpu_yf2xf.block<3,3>(0,0);
    Eigen::Vector3f T_cpu_yf2xf = transform_cpu_yf2xf.block<3,1>(0,3);
    cudaMemcpy(R_gpu_yf2xf, &R_cpu_yf2xf, sizeof(decltype(R_cpu_yf2xf)), cudaMemcpyHostToDevice);
    cudaMemcpy(T_gpu_yf2xf, &T_cpu_yf2xf, sizeof(decltype(T_cpu_yf2xf)), cudaMemcpyHostToDevice);

    // cvo::CvoPoint xx = cloud_x_gpu->points[0];
    //std::cout<<"Before transform: "<<xx.x<<"\n";
    transform_pointcloud_thrust(cloud_x_gpu, cloud_x_gpu_transformed_kdtree,
                                R_gpu_yf2xf, T_gpu_yf2xf, false);
    //cvo::CvoPoint xx_y = cloud_x_gpu_transformed_kdtree->points[0];    
    //std::cout<<"After transform: "<<xx_y.x<<"\n";
    
    kdtree_cloud_y.NearestKSearch(cloud_x_gpu_transformed_kdtree, num_neighbors , cukdtree_inds_results);

    cudaFree(R_gpu_yf2xf);
    cudaFree(T_gpu_yf2xf);

    cudaEventRecord(cuda_stop, 0);
    cudaEventSynchronize(cuda_stop);
    float elapsedTime, totalTime;
    
    cudaEventElapsedTime(&elapsedTime, cuda_start, cuda_stop);
    cudaEventDestroy(cuda_start);
    cudaEventDestroy(cuda_stop);
    totalTime = elapsedTime/(1000);
    //if (debug_print)std::cout<<"Kdtree query time is "<<totalTime<<std::endl;
    
  }

  void find_nearby_source_points_cukdtree(//const CvoParams *cvo_params,
                                          std::shared_ptr<CvoPointCloudGPU> cloud_x_gpu,
                                          CuKdTree & kdtree_cloud_y,
                                          const Eigen::Matrix4f & transform_cpu_yf2xf,
                                          int num_neighbors,
                                          // output
                                          std::shared_ptr<CvoPointCloudGPU> cloud_x_gpu_transformed_kdtree,
                                          int * cukdtree_inds_results
                                          ) {
    cudaEvent_t cuda_start, cuda_stop;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);
    cudaEventRecord(cuda_start, 0);

    Eigen::Matrix3f * R_gpu_yf2xf;
    Eigen::Vector3f * T_gpu_yf2xf;
    cudaMalloc(&R_gpu_yf2xf, sizeof(Eigen::Matrix3f));
    cudaMalloc(&T_gpu_yf2xf, sizeof(Eigen::Vector3f));
    Eigen::Matrix3f R_cpu_yf2xf = transform_cpu_yf2xf.block<3,3>(0,0);
    Eigen::Vector3f T_cpu_yf2xf = transform_cpu_yf2xf.block<3,1>(0,3);
    cudaMemcpy(R_gpu_yf2xf, &R_cpu_yf2xf, sizeof(decltype(R_cpu_yf2xf)), cudaMemcpyHostToDevice);
    cudaMemcpy(T_gpu_yf2xf, &T_cpu_yf2xf, sizeof(decltype(T_cpu_yf2xf)), cudaMemcpyHostToDevice);
    
    //cvo::CvoPoint xx = cloud_x_gpu->points[0];
    //std::cout<<"Before transform: "<<xx.x<<"\n";    
    transform_pointcloud_thrust(cloud_x_gpu, cloud_x_gpu_transformed_kdtree,
                                R_gpu_yf2xf, T_gpu_yf2xf, false);
    //cvo::CvoPoint xx_y = cloud_x_gpu_transformed_kdtree->points[0];    
    //std::cout<<"After transform: "<<xx_y.x<<"\n";
    
    kdtree_cloud_y.NearestKSearch(cloud_x_gpu_transformed_kdtree, num_neighbors ,
                                  cukdtree_inds_results, num_neighbors * cloud_x_gpu->size());

    cudaFree(R_gpu_yf2xf);
    cudaFree(T_gpu_yf2xf);

    cudaEventRecord(cuda_stop, 0);
    cudaEventSynchronize(cuda_stop);
    float elapsedTime, totalTime;
    
    cudaEventElapsedTime(&elapsedTime, cuda_start, cuda_stop);
    cudaEventDestroy(cuda_start);
    cudaEventDestroy(cuda_stop);
    totalTime = elapsedTime/(1000);
    //if (debug_print)std::cout<<"Kdtree query time is "<<totalTime<<std::endl;
    
  }
  
  
  
  
  


}
