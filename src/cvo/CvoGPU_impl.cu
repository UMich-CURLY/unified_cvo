#include "cvo/CvoGPU_impl.cuh"
#include "cvo/SparseKernelMat.cuh"
#include "utils/PointSegmentedDistribution.hpp"
#include "cvo/Association.hpp"
#include "utils/PointSegmentedDistribution.hpp"
#include "cvo/gpu_utils.cuh"
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
    //const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> & normals = cvo_cloud.normals();
    // const Eigen::Matrix<float, Eigen::Dynamic, 2> & types = cvo_cloud.types();
    auto & labels = cvo_cloud.labels();
    // set basic informations for pcl_cloud
    thrust::host_vector<CvoPoint> host_cloud;
    host_cloud.resize(num_points);

    int actual_num = 0;
    for(int i=0; i<num_points; ++i){
      (host_cloud)[i].x = positions[i](0);
      (host_cloud)[i].y = positions[i](1);
      (host_cloud)[i].z = positions[i](2);
      if (FEATURE_DIMENSIONS == 5) {
        (host_cloud)[i].r = (uint8_t)std::min(255.0, (features(i,0) * 255.0));
        (host_cloud)[i].g = (uint8_t)std::min(255.0, (features(i,1) * 255.0));
        (host_cloud)[i].b = (uint8_t)std::min(255.0, (features(i,2) * 255.0));
      }

      ///memcpy(host_cloud[i].features, features.row(i).data(), FEATURE_DIMENSIONS * sizeof(float));
      for (int j = 0; j < FEATURE_DIMENSIONS; j++)
        host_cloud[i].features[j] = features(i,j);

      if (cvo_cloud.num_classes() > 0) {
        labels.row(i).maxCoeff(&host_cloud[i].label);
        for (int j = 0; j < cvo_cloud.num_classes(); j++)
          host_cloud[i].label_distribution[j] = labels(i,j);
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

  CvoPointCloudGPU::SharedPtr pcl_PointCloud_to_gpu(const pcl::PointCloud<CvoPoint> & cvo_cloud ) {
    int num_points = cvo_cloud.size();
    //const ArrayVec3f & positions = cvo_cloud.positions();
    //const Eigen::Matrix<float, Eigen::Dynamic, FEATURE_DIMENSIONS> & features = cvo_cloud.features();
    //const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> & normals = cvo_cloud.normals();
    // const Eigen::Matrix<float, Eigen::Dynamic, 2> & types = cvo_cloud.types();
    //auto & labels = cvo_cloud.labels();
    // set basic informations for pcl_cloud
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


  
  void gpu_association_to_cpu(const SparseKernelMat & association_gpu,
                              Association & association_cpu,
                              int num_source,
                              int num_target,
                              int num_neighbors) {
    int rows = association_gpu.rows;
    int cols = num_neighbors == -1? association_gpu.cols : num_neighbors;

    association_cpu.source_inliers.resize(rows, false);
    association_cpu.target_inliers.resize(num_target, false);

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

    //tbb::mutex mutex;
    //tbb::parallel_for(int(0), rows, [&]( int i ){
#pragma omp parallel for    
    for (int i = 0; i < rows; i++) {

  
      if (nonzeros[i] > 0) {
        association_cpu.source_inliers[i] = true;
        
        for (int j = 0 ; j < cols; j++) {
          if (ind_row2col[i * cols + j] == -1) {
            break;
          }
          
#pragma omp critical
          {
            int ind_target = ind_row2col[i*cols+j];
            association_cpu.target_inliers[ind_target] = true;
            association_cpu.pairs.insert(i, ind_target) = inner_product[i*cols+j];
          }
          
        }
      }

    }
    association_cpu.pairs.makeCompressed();
  
  }


}
