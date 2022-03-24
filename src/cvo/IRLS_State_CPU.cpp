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

    std::cout<<"Nonzeros is "<<ip_mat_.nonZeros()<<std::endl;
    
    for (int k=0; k<ip_mat_.outerSize(); ++k)
    {
      for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(ip_mat_,k); it; ++it) {

        int idx1 = it.row();   // row index
        int idx2 = it.col();   // col index (here it is equal to k)
        double color_ip = it.value();


        
        //ceres::CostFunction * cost_per_point =
        //  new PairwiseAnalyticalDiffFunctor(pc1[idx1], pc2[idx2], color_ip,  ell_, params_->sigma);
        

        
        ceres::CostFunction* cost_per_point
          = new ceres::AutoDiffCostFunction<PairwiseAutoDiffFunctor, 1, 12, 12>(new PairwiseAutoDiffFunctor(pc1[idx1],
                                                                                                            pc2[idx2],
                                                                                                            color_ip,
                                                                                                            ell_,
                                                                                                            params_->sigma));  
        
        
        //ceres::LossFunctionWrapper* loss_function(new ceres::HuberLoss(1.0), ceres::TAKE_OWNERSHIP);
        problem.AddResidualBlock(cost_per_point, nullptr , frame1->pose_vec, frame2->pose_vec);
      }
    }


  }

  BinaryStateCPU::BinaryStateCPU(CvoFrame::Ptr pc1,
                                 CvoFrame::Ptr pc2,
                                 const CvoParams * params,
                                 int num_kdtree_neighbor,
                                 double init_ell
                                 ) : frame1(pc1), frame2(pc2),
                                     params_(params),
                                     ip_mat_(pc1->points->size(),
                                             pc2->points->size()),
                                     ell_(init_ell){
    //pc2_curr_.resize(pc2->points->size());
    //pc1_curr_.resize(pc1->points->size());
    //Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();
    //Mat34d_row identity_34 = identity.block<3,4>(0,0);
    //transform_pcd(identity_34, pc1->points->positions(), pc1_curr_);
    std::cout<<"Construct BinaryStateCPU: ell is "<<ell_<<"\n";

    iter_ = 0;

    pc1_kdtree_.reset(new Kdtree(3 /*dim*/,
                                 frame1->points->positions(),
                                 num_kdtree_neighbor /* max leaf */ ));
    pc1_kdtree_->index->buildIndex();
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

    ell_ = params->multiframe_ell_init;
    std::cout<<"Construct BinaryStateCPU: ell is "<<ell_<<"\n";    
    iter_ = 0;

    pc1_kdtree_.reset(new Kdtree(3 /*dim*/, frame1->points->positions(), 20 /* max leaf */ ));
    pc1_kdtree_->index->buildIndex();
  }
  /*
  static
  float compute_geometric_type_ip(const float * geo_type_a,
                                  const float * geo_type_b,
                                  int size
                                  ) {
    float norm2_a = square_norm(geo_type_a, size);
    float norm2_b = square_norm(geo_type_b, size);
    float dot_ab = dot(geo_type_a, geo_type_b, size);
    //printf("norm2_a=%f, norm2_b=%f, dot_ab=%f\n", norm2_a, norm2_b, dot_ab);
    float geo_sim = dot_ab / sqrt(norm2_a * norm2_b);
    
    return geo_sim;
  }
  */

  static
  void se_kernel(// input
                 //const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > &pc1,
                 //const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > &pc2,
                 //const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& color_pc1,
                 //const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& color_pc2,
                 const CvoPointCloud & pc1_cvo,
                 const CvoPointCloud & pc2_cvo,
                 const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > & pc2_curr,                 
                 const CvoParams * cvo_params,                 
                 std::shared_ptr<BinaryStateCPU::Kdtree> mat_index,
                 float ell,
                 int iter,
                 // output
                 Eigen::SparseMatrix<double, Eigen::RowMajor> & ip_mat_gradient_prefix
                 //, mstd::vector<Eigen::Triplet<double>> & nonzero_list
                 ) {

    ip_mat_gradient_prefix.setZero();

    auto & pc1 = pc1_cvo.positions();
    auto & pc2 = pc2_cvo.positions();
    auto & labels1 = pc1_cvo.labels();
    auto & labels2 = pc2_cvo.labels();
    auto & features1 = pc1_cvo.features();
    auto & features2 = pc2_cvo.features();
    auto & geometry1 = pc1_cvo.geometric_types();
    auto & geometry2 = pc2_cvo.geometric_types();

    const float sigma2 = cvo_params->sigma * cvo_params->sigma;
    const float c_ell2 = cvo_params->c_ell * cvo_params->c_ell;
    const float s_ell2 = cvo_params->s_ell * cvo_params->s_ell;
    const float c_sigma2 = cvo_params->c_sigma*cvo_params->c_sigma;
    const float s_sigma2 = cvo_params->s_sigma*cvo_params->s_sigma;
    
    
    float d2_thresh=1, d2_c_thresh=1, d2_s_thresh=1;
    //d2_thresh = -2 * ell * ell * log(sp_thresh);
    if (cvo_params->is_using_geometry)
      d2_thresh = -2.0*ell*ell*log(cvo_params->sp_thres/sigma2);
    if (cvo_params->is_using_intensity)
      d2_c_thresh = -2.0*c_ell2*log(cvo_params->sp_thres/c_sigma2);
    if (cvo_params->is_using_semantics)
      d2_s_thresh = -2.0*s_ell2*log(cvo_params->sp_thres/s_sigma2);
    
    
    // typedef KDTreeVectorOfVectorsAdaptor<std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >, double>  kd_tree_t;
    //kd_tree_t mat_index(3 /*dim*/, pc2, 20 /* max leaf */ );
    //mat_index.index->buildIndex();

    #pragma omp parallel for num_threads(24)
    for (int idx = 0; idx < pc2_curr.size(); idx++) {

      const float search_radius = d2_thresh;
        
      std::vector<std::pair<size_t,float>>  ret_matches;
      nanoflann::SearchParams params_flann;
      const size_t num_matches = mat_index->index->radiusSearch(pc2_curr[idx].data(), search_radius, ret_matches, params_flann);

      for(size_t j=0; j<num_matches; ++j){
        int i = ret_matches[j].first;
        float d2 = ret_matches[j].second;

        float a = 1, sk=1, ck=1, k=1, geo_sim=1;
        if (cvo_params->is_using_geometric_type) {
          //geo_sim = compute_geometric_type_ip(p_a->geometric_type,
          //                                    p_b->geometric_type,
          //                                    2);
          const Eigen::Vector2f g_a = Eigen::Map<const Eigen::Vector2f>(geometry1.data()+i*2);
          const Eigen::Vector2f g_b = Eigen::Map<const Eigen::Vector2f>(geometry2.data()+idx*2);
          geo_sim = g_a.dot(g_b) / g_a.norm() / g_b.norm();
          if(geo_sim < 0.01) {
            //         std::cout<<"i="<<i<<", j="<<idx<<", geo_sim="<<geo_sim<<", skipped\n";
            continue;
          }
        }
      
        if (cvo_params->is_using_geometry) {
          //float d2 = (squared_dist( *p_b ,*p_a ));
          if (d2 < d2_thresh)
            k= sigma2*exp(-d2/(2.0*ell*ell));
          else continue;
        }
        
        Eigen::VectorXf feature_b = features2.row(idx);          
        Eigen::VectorXf feature_a = features1.row(i);
        if (cvo_params->is_using_intensity) {
          float d2_color = (feature_a-feature_b).squaredNorm();
          if (d2_color < d2_c_thresh)
            ck = c_sigma2 * exp(-d2_color/(2.0*c_ell2));
          else
            continue;
        }
        if (cvo_params->is_using_semantics) {
          Eigen::VectorXf label_a = labels1.row(i);
          Eigen::VectorXf label_b = labels2.row(idx);
          float d2_semantic = (label_a - label_b).squaredNorm();  //squared_dist<float>(p_a->label_distribution, p_b->label_distribution, NUM_CLASSES);
          if (d2_semantic < d2_s_thresh )
            sk = s_sigma2*exp(-d2_semantic/(2.0*s_ell2));
          else
            continue;
        }
        a = ck*k*sk*geo_sim;
        if (a > cvo_params->sp_thres){
          //double a_residual = ck * k * d2 ; // least square
          double a_gradient = (double)ck * k ;
#pragma omp critical
          {
            //nonzero_list.push_back(Eigen::Triplet<double>(i, idx, a));
            ip_mat_gradient_prefix.insert(i, idx) = a_gradient;
            //*sum_residual += a_residual;
          }
          /*
          if ( idx == 0) {
            std::cout<<"Inside se_kernel: i="<<i<<"j="<<idx<<", d2="<<d2<<", k="<<k<<
            ", ck="<<ck
            <<", the point_a is ("<<pc1[i].transpose()
            <<", the point_b is ("<<pc2[idx].transpose()<<std::endl
                     << ", geo_sim is "<<geo_sim
                     <<"feature_a "<<feature_a.transpose()<<", feature_b "<<feature_b.transpose()<<std::endl;
                
                     }*/
          
          //}
        }
      }
    }
    ip_mat_gradient_prefix.makeCompressed();
  }

  void BinaryStateCPU::update_ell() {
    if (ell_ >(double) params_->multiframe_ell_min)
      ell_ = ell_ * (double) params_->multiframe_ell_decay_rate;     
  }


  int BinaryStateCPU::update_inner_product() {

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

    se_kernel(*frame1->points, *frame2->points,
              pc2_curr_,
              params_,
              pc1_kdtree_,
              (float)ell_,
              iter_,
              ip_mat_
              );

    int nonzeros = ip_mat_.nonZeros();
    return nonzeros;
    //if (ip_mat_.nonZeros() < 100) {
    //  std::cout<<"too sparse inner product mat "<<ip_mat_.nonZeros()<<std::endl;
    //  return -1;
    //} else
    //  return 0;
    
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

