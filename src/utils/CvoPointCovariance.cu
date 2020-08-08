#include "utils/CvoPointCloud.hpp"
#include "utils/CvoPixelSelector.hpp"
#include "utils/LidarPointSelector.hpp"
#include "utils/LeGoLoamPointSelection.hpp"
#include "cupointcloud/point_types.h"
#include "cupointcloud/cupointcloud.h"
#include "cukdtree/cukdtree.h"
#include "cvo/gpu_utils.cuh"
#include <chrono>
#include <vector>
#include <memory>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace cvo {
  /*   helper functions  */
  static
  __global__
  void init_covariance(perl_registration::cuPointXYZ * points, // mutate
                       int num_points,
                       int * neighbors,
                       int num_neighbors_each_point,
                       // outputs
                       float  * covariance_out,
                       float * eigenvalues_out,
                       bool * is_cov_degenerate
                       ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > num_points - 1) {
      return;
    }
    if (i == 0)
      printf("i==0 start\n");
    perl_registration::cuPointXYZ & curr_p = points[i];
    if (i == 1000)
      printf("i==1000: p is %f,%f,%f\n", curr_p.x, curr_p.y, curr_p.z);
    
    Eigen::Vector3f curr_p_vec (curr_p.x, curr_p.y, curr_p.z);
    int * indices = neighbors + i * num_neighbors_each_point;
    
    Eigen::Vector3f mean(0, 0, 0);
    int num_neighbors_in_range = 0;
    for (int j = 0; j < num_neighbors_each_point; j++) {
      auto & neighbor = points[indices[j]]; 
      if (squared_dist(neighbor, curr_p) > 0.65 * 0.65) {
        indices[j] = -1;
        continue;
      }
     
      Eigen::Vector3f neighbor_vec(neighbor.x, neighbor.y, neighbor.z);
      mean = (mean + neighbor_vec).eval();
      num_neighbors_in_range += 1;
    }

    if (num_neighbors_in_range < 10) {
      is_cov_degenerate[i] = true;
      return;
    } else
      is_cov_degenerate[i] = false;
    
    mean = mean  / static_cast<float>(num_neighbors_in_range);

    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (int j = 0; j < num_neighbors_each_point; j++) {
      auto & neighbor = points[indices[j]];
      if (indices[j] == -1) continue;
      Eigen::Vector3f neighbor_vec(neighbor.x, neighbor.y, neighbor.z);
      Eigen::Vector3f x_minis_mean = neighbor_vec - mean;
      Eigen::Matrix3f temp_cov = x_minis_mean * x_minis_mean.transpose();
      covariance = covariance + temp_cov;
      if (i == 1000) {
        //printf("i==0: x_minis_mean is %.3f,%.3f,%.3f\n", x_minis_mean(0),
        //       x_minis_mean(1), x_minis_mean(2));
        printf("i==1000: neighbor (%f, %f, %f)\n", neighbor.x, neighbor.y, neighbor.z);
      }
    }
    covariance = (covariance / (float)(num_neighbors_in_range-1)).eval();
    float * cov_curr = covariance_out + i * 9;
    for (int j = 0; j < 9; j++)
      cov_curr[j] = covariance(j/3, j%3);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(3);
    //es.compute(covariance);
    if (i == 1000) {
      printf("i=%d, covariance is\n %f %f %f\n %f %f %f\n %f %f %f\n",i,
             covariance(0,0), covariance(0,1), covariance(0,2),
             covariance(1,0), covariance(1,1), covariance(1,2),
             covariance(2,0), covariance(2,1), covariance(2,2));

    }
    /* 
    //auto e_values = es.eigenvalues();
    for (int j = 0; j < 3; j++) {
      eigenvalues_out[j+3*i] = e_values(j);
      
      if (i == 0) {
        printf("i==0: cov_eigenvalue is %.3f,%.3f,%.3f\n", eigenvalues_out[3*i],
               eigenvalues_out[1+3*i], eigenvalues_out[2+3*i]);
      }
      }*/

    /* 
    //PCA in GICP 
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(3);
    es.computeDirect(covariance);
    // Eigen values are sorted
    Eigen::Matrix3f eigen_value_replacement = Eigen::Matrix3f::Zero();
    eigen_value_replacement(0, 0) = 1e-3;
    eizgen_value_replacement(1, 1) = 1.0;
    eigen_value_replacement(2, 2) = 1.0;
    covariances.data[pos] = es.eigenvectors() * eigen_value_replacement *
    es.eigenvectors().transpose();
    covariance = covariances.data[pos];
    */
  }

  static
  void fill_in_pointcloud_covariance(perl_registration::cuPointCloud<perl_registration::cuPointXYZ>::SharedPtr pointcloud_gpu, thrust::device_vector<float> & covariance, thrust::device_vector<float> & eigenvalues, thrust::device_vector<bool> & is_cov_degenerate ) {
    auto start = std::chrono::system_clock::now();
    perl_registration::cuKdTree<perl_registration::cuPointXYZ> kdtree;
    kdtree.SetInputCloud(pointcloud_gpu);
    const int num_wanted_points = KDTREE_K_SIZE;
    thrust::device_vector<int> indices;

    kdtree.NearestKSearch(pointcloud_gpu, num_wanted_points, indices );
    cudaDeviceSynchronize();

    std::cout<<"Init cov, point size is "<<pointcloud_gpu->size()<<std::endl<<std::flush;

    /*
      thrust::host_vector<int> indices_host = indices;
      for(int i = 0; i < num_wanted_points; i++) {
      std::cout<<indices_host[i+num_wanted_points * (pointcloud_gpu->size()-1)]<<",";
      
      }
      std::cout<<"\n";
    */

    covariance.resize(pointcloud_gpu->size()*9);
    eigenvalues.resize(pointcloud_gpu->size()*3);
    is_cov_degenerate.resize(pointcloud_gpu->size());
    init_covariance<<<(pointcloud_gpu->size() / 512 + 1), 512>>>(
                                                                 thrust::raw_pointer_cast(pointcloud_gpu->points.data()), // mutate
                                                                 pointcloud_gpu->size(),
                                                                 thrust::raw_pointer_cast(indices.data() ),
                                                                 num_wanted_points,
                                                                 thrust::raw_pointer_cast(covariance.data()),
                                                                 thrust::raw_pointer_cast(eigenvalues.data()),
                                                                 thrust::raw_pointer_cast(is_cov_degenerate.data()));
    cudaDeviceSynchronize();
    
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> t_kdtree_search = end-start;
    std::cout<<"kdtree construction and  nn search time is "<<t_kdtree_search.count()<<std::endl;
  }

  static
  void compute_covariance(const pcl::PointCloud<pcl::PointXYZI> & pc_raw,
                          thrust::device_vector<float> & covariance_all,
                          thrust::device_vector<float> & eigenvalues_all,
                          thrust::device_vector<bool> & is_cov_degenerate
                          )   {


    //int num_points = selected_indexes.size();
    int num_points = pc_raw.size();

    // set basic informations for pcl_cloud
    thrust::host_vector<perl_registration::cuPointXYZ> host_cloud;
    //pcl::PointCloud<perl_registration::cuPointXYZ> host_cloud;
    host_cloud.resize(num_points);

    for(int i=0; i<num_points; ++i){
      //auto & curr_p = pc_raw[selected_indexes[i]];
      auto & curr_p = pc_raw[i];
      (host_cloud)[i].x = curr_p.x;
      (host_cloud)[i].y = curr_p.y;
      (host_cloud)[i].z = curr_p.z;
    }
   
    //gpu_cloud->points = host_cloud;
    //auto pc_gpu = std::make_shared<perl_registration::cuPointCloud<perl_registration::cuPointXYZ>>(new perl_registration::cuPointCloud<perl_registration::cuPointXYZ>>);
    perl_registration::cuPointCloud<perl_registration::cuPointXYZ>::SharedPtr pc_gpu (new perl_registration::cuPointCloud<perl_registration::cuPointXYZ>);
    pc_gpu->points = host_cloud;

    fill_in_pointcloud_covariance(pc_gpu, covariance_all, eigenvalues_all, is_cov_degenerate);

    return;
  }


  
  CvoPointCloud::CvoPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pc,  int beam_num) {
    double intensity_bound = 0.4;
    double depth_bound = 4.0;
    double distance_bound = 75.0;
    //pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out (new pcl::PointCloud<pcl::PointXYZI>);
    //std::unique_ptr<pcl::PointCloud<pcl::PointXYZI>> pc_out = std::make_unique<pcl::PointCloud<pcl::PointXYZI>>();
    //pcl::PointCloud<pcl::PointXYZI> pc_out;
    std::vector <double> output_depth_grad;
    std::vector <double> output_intenstity_grad;
    std::vector <int> selected_indexes;

    
    int expected_points = 5000;

    /*
    std::vector <float> edge_or_surface;
    LidarPointSelector lps(expected_points, intensity_bound, depth_bound, distance_bound, beam_num);
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_edge (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_surface (new pcl::PointCloud<pcl::PointXYZI>);
    lps.edge_detection(pc, pc_out_edge, output_depth_grad, output_intenstity_grad, selected_indexes);    
    lps.legoloam_point_selector(pc, pc_out_surface, edge_or_surface, selected_indexes);    
    *pc_out += *pc_out_edge;
    *pc_out += *pc_out_surface;
    */

    
    random_surface_with_edges(pc, expected_points, intensity_bound, depth_bound, distance_bound, beam_num,
                              output_depth_grad, output_intenstity_grad, selected_indexes);
    std::cout<<"compute covariance\n";
    thrust::device_vector<float> cov_all, eig_all;
    thrust::device_vector<bool> is_cov_degenerate_gpu;
    compute_covariance(*pc, cov_all, eig_all, is_cov_degenerate_gpu);
    std::unique_ptr<thrust::host_vector<float>> cov(new thrust::host_vector<float>(cov_all));
    std::unique_ptr<thrust::host_vector<float>> eig(new thrust::host_vector<float>(eig_all));
    std::unique_ptr<thrust::host_vector<bool>> is_cov_degenerate_host(new thrust::host_vector<bool>(is_cov_degenerate_gpu));

    num_points_ = 0;
    for (int j = 0; j < selected_indexes.size(); j++){
      if((*is_cov_degenerate_host)[selected_indexes[j]] == false)
        num_points_+=1;
    }

    // fill in class members
    num_classes_ = 0;
    
    // features_ = Eigen::MatrixXf::Zero(num_points_, 1);
    feature_dimensions_ = 1;
    features_.resize(num_points_, feature_dimensions_);
    normals_.resize(num_points_,3);
    covariance_.resize(num_points_ * 9);
    eigenvalues_.resize(num_points_*3);
    //eigenvalues_.resize(num_points_ * 3);
    //types_.resize(num_points_, 2);


    std::ofstream e_value_max("e_value_max.txt");
    std::ofstream e_value_min("e_value_min.txt");

    int actual_i = 0;
    for (int i = 0; i < selected_indexes.size() ; i++) {
      if ((*is_cov_degenerate_host)[selected_indexes[i]]) continue;
      
      int id_pc_in = selected_indexes[i];
      Vec3f xyz;
      //xyz << pc_out->points[i].x, pc_out->points[i].y, pc_out->points[i].z;
      xyz << pc->points[id_pc_in].x, pc->points[id_pc_in].y, pc->points[id_pc_in].z;
      positions_.push_back(xyz);
      features_(actual_i, 0) = pc->points[id_pc_in].intensity;
     
      memcpy(covariance_.data() + actual_i * 9, cov->data() + id_pc_in * 9, sizeof(float)*9);
      Eigen::Map<Eigen::Matrix3f> cov_curr(&covariance_.data()[actual_i*9]);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> ev(cov_curr);
      Eigen::Vector3f e_values = ev.eigenvalues();
      for (int j = 0; j < 3; j++) {
        eigenvalues_[j+actual_i*3] = sqrt(e_values(j));
      }
      //std::cout<<"i=="<<i<<":e_values are "<<e_values.transpose()<<", e_vectors are \n"<<ev.eigenvectors()<<std::endl;
      
      Eigen::Matrix3f tmp = cov_curr;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(3);
      es.computeDirect(cov_curr);
      Eigen::Matrix3f eigen_value_replacement = Eigen::Matrix3f::Zero();
      eigen_value_replacement(0, 0) = e_values(0) < 0.001 ? e_values(0) : 0.001;
      eigen_value_replacement(1, 1) = e_values(1) < 1 ? e_values(1) : 1;
      eigen_value_replacement(2, 2) = e_values(2) < 1  ? e_values(2) : 1;
      cov_curr = es.eigenvectors() * eigen_value_replacement *
        es.eigenvectors().transpose();
      //covariance = covariances.data[pos];
      eigenvalues_[actual_i*3]= sqrt(eigen_value_replacement(0,0));
      eigenvalues_[actual_i*3+1]=sqrt(eigen_value_replacement(1,1));
      eigenvalues_[actual_i*3+2]=sqrt(eigen_value_replacement(2,2));
      if (actual_i == 1) {
        std::cout<<"selected_indexes[i] is "<<id_pc_in<<", actual_i is "<<actual_i<<", cov_before is "<<tmp<<", cov_after is "<<cov_curr<<"andd\n"<<covariance_[actual_i*9]<<covariance_[actual_i*9+1]<<std::endl<<" eigenvalues_[actual_i] is "<<eigenvalues_[actual_i*3]<<", "<<eigenvalues_[actual_i*3+1]<<std::endl;
        
      } 
      

      // e_value_max << e_values(2) << std::endl;
      //e_value_min << e_values(0) << std::endl;
      
      actual_i++;
      //memcpy(eigenvalues_.data() + i * 3, eig->data() + id_pc_in * 3, sizeof(float)*3);
    }

    e_value_min.close();
    e_value_max.close();
    
    std::cout<<"Construct Cvo PointCloud, num of points is "<<num_points_<<" from "<<pc->size()<<" input points "<<std::endl;    
    //write_to_intensity_pcd("kitti_lidar.pcd");

  }

  

}
