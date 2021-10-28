#include "cvo/IRLS_State_CPU.hpp"
#include "cvo/IRLS_Cost_CPU.hpp"
#include "cvo/local_parameterization_se3.hpp"
#include "utils/data_type.hpp"
#include "cvo/KDTreeVectorOfVectorsAdaptor.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace cvo {


  static
  void transform_pcd(const Eigen::Matrix<double, 3, 4, Eigen::RowMajor> & transform,
                     const std::vector<Eigen::Vector3d,
                       Eigen::aligned_allocator<Eigen::Vector3d>> & cloud_y_init,
                     // output
                     std::vector<Eigen::Vector3d,
                       Eigen::aligned_allocator<Eigen::Vector3d> > & cloud_y
                     )  {
    int num_pts = cloud_y_init.size();
    #pragma omp parallel for
    for (int i = 0; i < num_pts; i++ ){
      (cloud_y)[i] = transform.block<3,3>(0,0)*cloud_y_init[i]+transform.block<3,1>(0,3);
    }
  }
  static
  void transform_pcd(const Eigen::Matrix<double, 3, 4, Eigen::RowMajor> & transform,
                     const std::vector<Eigen::Vector3f,
                       Eigen::aligned_allocator<Eigen::Vector3f>> & cloud_y_init,
                     // output
                     std::vector<Eigen::Vector3d,
                       Eigen::aligned_allocator<Eigen::Vector3d> > & cloud_y
                     )  {
    int num_pts = cloud_y_init.size();
    if (cloud_y.size() < num_pts)
      cloud_y.resize(num_pts);
    #pragma omp parallel for
    for (int i = 0; i < num_pts; i++ ){
      (cloud_y)[i] = transform.block<3,3>(0,0)*cloud_y_init[i].cast<double>()+transform.block<3,1>(0,3);
    }
  }

  static
  void transform_pcd(const Eigen::Matrix<double, 3, 4, Eigen::RowMajor> & transform,
                     const std::vector<Eigen::Vector3f,
                       Eigen::aligned_allocator<Eigen::Vector3f>> & cloud_y_init,
                     // output
                     std::vector<Eigen::Vector3f,
                       Eigen::aligned_allocator<Eigen::Vector3f> > & cloud_y
                     )  {
    int num_pts = cloud_y_init.size();
    if (cloud_y.size() < num_pts)
      cloud_y.resize(num_pts);
    #pragma omp parallel for
    for (int i = 0; i < num_pts; i++ ){
      (cloud_y)[i] = transform.block<3,3>(0,0).cast<float>()*cloud_y_init[i]+transform.block<3,1>(0,3).cast<float>();
    }
  }

  
  void BinaryStateCPU::add_residual_to_problem(ceres::Problem & problem) {

    auto & pc1 = frame1->points->positions();
    auto & pc2 = frame2->points->positions();
    
    for (int k=0; k<ip_mat_.outerSize(); ++k)
    {
      for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(ip_mat_,k); it; ++it) {

        int idx1 = it.row();   // row index
        int idx2 = it.col();   // col index (here it is equal to k)
        double color_ip = it.value();
        /*
          ceres::CostFunction * cost_per_point =
          new PairwiseCostFunction(&pc1[idx1], &pc2[idx2], color_ip,  ell, 0.1);
        */
        
        ceres::CostFunction* cost_per_point
          = new ceres::AutoDiffCostFunction<PairwiseAutoDiffFunctor, 1, 12, 12>(new PairwiseAutoDiffFunctor(pc1[idx1],
                                                                                                            pc2[idx2],
                                                                                                            color_ip,
                                                                                                            ell_,
                                                                                                            params_->sigma));    
        problem.AddResidualBlock(cost_per_point, nullptr, frame1->pose_vec, frame2->pose_vec);
      }
    }


  }


  
  BinaryStateCPU::BinaryStateCPU(CvoFrame::Ptr pc1,
                                 CvoFrame::Ptr pc2,
                                 const CvoParams * params
                                 ) : frame1(pc1), frame2(pc2),
                                     params_(params),
                                     ip_mat_(pc1->points->size(),
                                             pc2->points->size()) {
    //pc2_curr_.resize(pc2->points->size());
    //pc1_curr_.resize(pc1->points->size());
    //Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();
    //Mat34d_row identity_34 = identity.block<3,4>(0,0);
    //transform_pcd(identity_34, pc1->points->positions(), pc1_curr_);

    ell_ = params->ell_init;
    iter_ = 0;

    pc1_kdtree_.reset(new Kdtree(3 /*dim*/, frame1->points->positions(), 10 /* max leaf */ ));
    pc1_kdtree_->index->buildIndex();
  }

  static
  void se_kernel(// input
                 const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > &pc1,
                 const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > &pc2,
                 const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& color_pc1,
                 const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& color_pc2,
                 std::shared_ptr<BinaryStateCPU::Kdtree> mat_index,
                 float ell,
                 float sp_thresh,
                 float sigma,
                 int iter,
                 // output
                 Eigen::SparseMatrix<double, Eigen::RowMajor> & ip_mat_gradient_prefix
                 //, mstd::vector<Eigen::Triplet<double>> & nonzero_list
                 ) {


    
    const float d2_thresh = -2 * ell * ell * log(sp_thresh);
    const float d2_c_thresh = -2.0*0.05*0.05*log(sp_thresh);    
    const float s2 = sigma * sigma;
    
    // typedef KDTreeVectorOfVectorsAdaptor<std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >, double>  kd_tree_t;
    //kd_tree_t mat_index(3 /*dim*/, pc2, 20 /* max leaf */ );
    //mat_index.index->buildIndex();

    #pragma omp parallel for
    for (int idx = 0; idx < pc2.size(); idx++) {

      const float search_radius = d2_thresh;
        
      std::vector<std::pair<size_t,float>>  ret_matches;
      nanoflann::SearchParams params_flann;
      const size_t nMatches = mat_index->index->radiusSearch(pc2[idx].data(), search_radius, ret_matches, params_flann);



      for(size_t j=0; j<nMatches; ++j){
        int i = ret_matches[j].first;
        float d2 = ret_matches[j].second;

        float k = 1;
        float ck = 1;
        float a = 1;
        if(d2<d2_thresh){
          Eigen::VectorXf feature_b = color_pc2.col(idx);          
          Eigen::VectorXf feature_a = color_pc1.col(i);
          float d2_color = (feature_a-feature_b).squaredNorm();
            
          if(d2_color<d2_c_thresh){
            k = s2*exp(-d2/(2.0*ell*ell));
            
            ck = exp(-d2_color/(2.0*0.05*0.05));
            a = ck*k;

            if (a > sp_thresh){
              //double a_residual = ck * k * d2 ; // least square
              double a_gradient = (double)ck * k ;
              #pragma omp critical
              {
                //nonzero_list.push_back(Eigen::Triplet<double>(i, idx, a));
                ip_mat_gradient_prefix.insert(i, idx) = a_gradient;
                //*sum_residual += a_residual;
              }
            }
          /*if ( i == 10) {
            std::cout<<"Inside se_kernel: i=10, j="<<idx<<", d2="<<d2<<", k="<<k<<
            ", ck="<<ck
            <<", the point_a is ("<<pc1[i].transpose()
            <<", the point_b is ("<<pc2[idx].transpose()<<std::endl;
            std::cout<<"feature_a "<<feature_a.transpose()<<", feature_b "<<feature_b.transpose()<<std::endl;
                
            }*/
          
          }
        }
      }
    }
    ip_mat_gradient_prefix.makeCompressed();
  }

  void BinaryStateCPU::update_ell() {
    if (ell_ >(double) params_->ell_min)
      ell_ = ell_ * (double) params_->ell_decay_rate;     
  }


  void BinaryStateCPU::update_inner_product() {

    /*
    if (iter_ && iter_ % 10 == 0 ) {
      ell_ =  ell_ * params_->ell_decay_rate;
    }
    */
    Mat4d_row pose1_full = Mat4d_row::Identity();
    Mat4d_row pose2_full = Mat4d_row::Identity();
    pose1_full.block<3,4>(0,0) = Eigen::Map<Mat34d_row>(frame1->pose_vec);
    pose2_full.block<3,4>(0,0) = Eigen::Map<Mat34d_row>(frame2->pose_vec);;
    Mat34d_row f1_to_f2 = (pose1_full.inverse() * pose2_full ).block<3,4>(0,0);

    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > pc2_curr_;
    transform_pcd(f1_to_f2, frame2->points->positions(), pc2_curr_);

    se_kernel(frame1->points->positions(), pc2_curr_,
              frame1->points->features(), frame2->points->features(),
              pc1_kdtree_,
              (float)ell_, (float)params_->sp_thres,(float) params_->sigma, iter_,
              ip_mat_
              );
    
    
  }

  /*
  void BinaryStateCPU::update_jacobians(double ** jacobians) {
    //Eigen::Matrix<double, 3, 12> DT_all = Eigen::Matrix<double, 3, 12>::Zero();
    int counter = 0;


    if (!jacobians || !jacobians[0]) return;

    Mat34d_row T1 = Eigen::Map<Mat34d_row>(frame1->pose_vec);
    //Eigen::Mat44d_row T1 = Eigen::Mat4f::Identity();
    //T1.block<3,4>(0,0) = T1_rt;
    transform_pcd(T1, frame1->points->positions(), pc1_curr_);

    Mat34d_row T2 = Eigen::Map<Mat34d_row>(frame2->pose_vec);
    //Eigen::Mat44d_row T2 = Eigen::Mat4f::Identity();
    //T2.block<3,4>(0,0) = T2_rt;
    transform_pcd(T2, frame2->points->positions(), pc2_curr_);
    
    
    #pragma omp parallel for
    for (int k=0; k<ip_mat_gradient_prefix_.outerSize(); ++k)
    {
      for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(ip_mat_gradient_prefix_,k); it; ++it) {
        double grad_prefix = it.value();
        int idx1 = it.row();   // row index
        int idx2 = it.col();   // col index (here it is equal to k)
        
        auto Tp1_sub_Tp2 = ( pc1_curr_[idx1] - pc2_curr_[idx2]).transpose();
        
        Vec4d_row pt2_homo, pt1_homo;
        pt2_homo << pc2_curr_[idx2][0], pc2_curr_[idx2][1], pc2_curr_[idx2][2],1;
        pt1_homo << pc1_curr_[idx2][0], pc1_curr_[idx2][1], pc1_curr_[idx2][2],1;

        Eigen::Matrix<double, 3, 12> DT2 = Eigen::Matrix<double, 3, 12, Eigen::RowMajor>::Zero();
        DT2.block<1,4>(0,0) = pt2_homo;
        DT2.block<1,4>(1,4) = pt2_homo;
        DT2.block<1,4>(2,8) = pt2_homo;
        Eigen::Matrix<double, 3, 12> DT1 = Eigen::Matrix<double, 3, 12, Eigen::RowMajor>::Zero();
        DT1.block<1,4>(0,0) = pt1_homo;
        DT1.block<1,4>(1,4) = pt1_homo;
        DT1.block<1,4>(2,8) = pt1_homo;
        
        #pragma omp critical
        {
          Eigen::Map<Vec12d_row> jacob1(&jacobians[0][0]);
          jacob1 =  (jacob1 + Tp1_sub_Tp2 * DT1).eval();
          Eigen::Map<Vec12d_row> jacob2(&jacobians[1][0]);
          jacob2 =  (jacob2 - Tp1_sub_Tp2 * DT2).eval();
          
          if (counter == 0) {
            //std::cout<<"Compute Jacobian: \nDT is \n"<<DT<<",\n p1_sub_Tp2 is \n"<<p1_sub_Tp2<<std::endl<<"jacob is \n"<<jacob<<std::endl;
          }
          counter++;

        }
      
      
      }

    }
    
  }
  */
  
}

