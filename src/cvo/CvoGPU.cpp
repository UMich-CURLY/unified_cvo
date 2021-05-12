#include "utils/CvoPointCloud.hpp"
#include "utils/PointSegmentedDistribution.hpp"
#include "utils/CvoPoint.hpp"
#include "cvo/nanoflann.hpp"
#include "cvo/CvoParams.hpp"
#include "cvo/CvoGPU.hpp"
#include <tbb/tbb.h>
#include "cvo/KDTreeVectorOfVectorsAdaptor.h"
#include <pcl/point_cloud.h>
#include <Eigen/Dense>
#include <cstdlib>


namespace cvo {
  typedef Eigen::Triplet<float> Trip_t;
  
  void CvoPointCloud_to_pcl(const CvoPointCloud & cvo_cloud,
                            pcl::PointCloud<CvoPoint> &pcl_cloud
                            ) {
    int num_points = cvo_cloud.num_points();
    const ArrayVec3f & positions = cvo_cloud.positions();
    const Eigen::Matrix<float, Eigen::Dynamic, FEATURE_DIMENSIONS> & features = cvo_cloud.features();
    //const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> & normals = cvo_cloud.normals();
    // const Eigen::Matrix<float, Eigen::Dynamic, 2> & types = cvo_cloud.types();
    auto & labels = cvo_cloud.labels();
    // set basic informations for pcl_cloud
    pcl_cloud.resize(num_points);

    //int actual_num = 0;
    for(int i=0; i<num_points; ++i){
      //memcpy(&host_cloud[i], &cvo_cloud[i], sizeof(CvoPoint));
      (pcl_cloud)[i].x = positions[i](0);
      (pcl_cloud)[i].y = positions[i](1);
      (pcl_cloud)[i].z = positions[i](2);
      if (FEATURE_DIMENSIONS >= 3) {
        (pcl_cloud)[i].r = (uint8_t)std::min(255.0, (features(i,0) * 255.0));
        (pcl_cloud)[i].g = (uint8_t)std::min(255.0, (features(i,1) * 255.0));
        (pcl_cloud)[i].b = (uint8_t)std::min(255.0, (features(i,2) * 255.0));
      }

      for (int j = 0; j < FEATURE_DIMENSIONS; j++)
        pcl_cloud[i].features[j] = features(i,j);

      if (cvo_cloud.num_classes() > 0) {
        labels.row(i).maxCoeff(&pcl_cloud[i].label);
        for (int j = 0; j < cvo_cloud.num_classes(); j++)
          pcl_cloud[i].label_distribution[j] = labels(i,j);
      }
      
      //if (normals.rows() > 0 && normals.cols()>0) {
      //  for (int j = 0; j < 3; j++)
      //    pcl_cloud[i].normal[j] = normals(i,j);
      //}

      //if (cvo_cloud.covariance().size() > 0 )
      //  memcpy(pcl_cloud[i].covariance, cvo_cloud.covariance().data()+ i*9, sizeof(float)*9  );
      //if (cvo_cloud.eigenvalues().size() > 0 )
      //  memcpy(pcl_cloud[i].cov_eigenvalues, cvo_cloud.eigenvalues().data() + i*3, sizeof(float)*3);

      //if (i == 1000) {
      //  printf("Total %d, Raw input from pcl at 1000th: \n", num_points);
      //  print_point(pcl_cloud[i]);
      //}
      
    }
    //gpu_cloud->points = host_cloud;

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
    return;
  }

  
  static
  void se_kernel_init_ell_cpu(const CvoPointCloud* cloud_a, const CvoPointCloud* cloud_b, \
                              cloud_t* cloud_a_pos, cloud_t* cloud_b_pos, \
                              Eigen::SparseMatrix<float,Eigen::RowMajor>& A_temp,
                              tbb::concurrent_vector<Trip_t> & A_trip_concur_,
                              const CvoParams & params
                              ) {
    bool debug_print = false;
    A_trip_concur_.clear();
    const float s2= params.sigma*params.sigma;

    const float l = params.ell_min;

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
        Eigen::VectorXf label_a;
        if (params.is_using_semantics)
          label_a = cloud_a->labels().row(i);
        
        // for(int j=0; j<num_moving; j++){
        for(size_t j=0; j<nMatches; ++j){
          int idx = ret_matches[j].first;
          float d2 = ret_matches[j].second;
          // d2 = (x-y)^2
          float k = 1;
          float ck = 1;
          float sk = 1;
          float d2_color = 0;
          float d2_semantic = 0;
          float a = 1;

          Eigen::VectorXf label_b;
          if (params.is_using_semantics) {
            label_b = cloud_b->labels().row(idx);
            d2_semantic = ((label_a-label_b).squaredNorm());            
            sk = params.s_sigma*params.s_sigma*exp(-d2_semantic/(2.0*params.s_ell*params.s_ell));
          }

          if (params.is_using_geometry) {
            k = s2*exp(-d2/(2.0*l*l));            
          } 

          if (params.is_using_intensity) {
            Eigen::Matrix<float,Eigen::Dynamic,1> feature_b = cloud_b->features().row(idx).transpose();            
            d2_color = ((feature_a-feature_b).squaredNorm());
            ck = params.c_sigma*params.c_sigma*exp(-d2_color/(2.0*params.c_ell*params.c_ell));
          }
          a = ck*k*sk;
          if (a > params.sp_thres){
            A_trip_concur_.push_back(Trip_t(i,idx,a));
          }
        }
      });

    A_temp.setFromTriplets(A_trip_concur_.begin(), A_trip_concur_.end());
    A_temp.makeCompressed();
  }



    
  float CvoGPU::inner_product_cpu(const CvoPointCloud& source_points,
                                  const CvoPointCloud& target_points,
                                  const Eigen::Matrix4f & t2s_frame_transform
                                  ) const {
    if (source_points.num_points() == 0 || target_points.num_points() == 0) {
      return 0;
    }
    ArrayVec3f fixed_positions = source_points.positions();
    ArrayVec3f moving_positions = target_points.positions();
    
    Eigen::Matrix3f rot = t2s_frame_transform.block<3,3>(0,0) ;
    Eigen::Vector3f trans = t2s_frame_transform.block<3,1>(0,3) ;
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

    return A_mat.sum();
  }


}
