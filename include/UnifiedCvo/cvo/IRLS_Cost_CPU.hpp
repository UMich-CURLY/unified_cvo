#include <ceres/ceres.h>
//#include "local_parameterization_se3.hpp"
#include <Eigen/Dense>

namespace cvo {
  
  class PairwiseAutoDiffFunctor {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    PairwiseAutoDiffFunctor(const Eigen::Vector3f & pt1,
                            const Eigen::Vector3f & pt2,
                            double label_ip,
                            double ell,
                            double sigma
                            ) {
      pt1_.head<3>() = pt1.cast<double>();
      pt1_[3] = 1.0;
      pt2_.head<3>() = pt2.cast<double>();
      pt2_[3] = 1.0;
      ell2_ = ell*ell;
      sigma2_ = sigma*sigma;
      label_ip_ = label_ip;
    
    }

    ~PairwiseAutoDiffFunctor() {}

    template <typename T>
    bool operator()(const T* pose1_vec,
                    const T* pose2_vec,
                    T* residuals) const {

      Eigen::Map<
        Eigen::Matrix<T, 3, 4, Eigen::RowMajor> const> T1(pose1_vec);
      Eigen::Map<
        Eigen::Matrix<T, 3, 4, Eigen::RowMajor> const> T2(pose2_vec);

      Eigen::Matrix<T, 3, 1> pt1_transformed = T1 * pt1_.cast<T>();
      Eigen::Matrix<T, 3, 1> pt2_transformed = T2 * pt2_.cast<T>();

      // auto p1_sub_Tp2 = (- pt1_.head<3>().cast<T>() + pt2_transformed).transpose();
      auto p1_sub_Tp2 = (- pt1_transformed + pt2_transformed);
      T d2 = p1_sub_Tp2.squaredNorm();
    
      // residuals[0] = color_ip_ * sigma2_ * exp(-d2/(2*ell2_));
      residuals[0] = label_ip_ * d2;

      //if (residuals) {
      //std::cout<<"\n\nnew iteration, T is \n"<<transformation<<std::endl;      
      //  residuals[0] =  (pt1_.cast<T>().head(3)-pt2_transformed).norm();
      //std::cout<<"residuals is "<<residuals[0]<<std::endl;
      //}
      //residuals[0] = (pt1.block<3,1>(0,0) - pt2_transformed).norm();

    
      return true;
    }

  private:
    //double pt1_[4];
    //double pt2_[4];
    Eigen::Vector4d pt1_;
    Eigen::Vector4d pt2_;
    double ell2_;
    double sigma2_;
    double label_ip_;
  
  };

  
  class PairwisePoseEllAutoDiffFunctor {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    PairwisePoseEllAutoDiffFunctor(const Eigen::Vector3f & pt1,
                                   const Eigen::Vector3f & pt2,
                                   double label_ip,
                                   double sigma
                                   ) {
      pt1_.head<3>() = pt1.cast<double>();
      pt1_[3] = 1.0;
      pt2_.head<3>() = pt2.cast<double>();
      pt2_[3] = 1.0;
      sigma2_ = sigma*sigma;
      label_ip_ = label_ip;
    
    }

    ~PairwisePoseEllAutoDiffFunctor() {}

    template <typename T>
    bool operator()(const T* pose1_vec,
                    const T* pose2_vec,
                    const T * ell,
                    T* residuals) const {

      Eigen::Map<
        Eigen::Matrix<T, 3, 4, Eigen::RowMajor> const> T1(pose1_vec);
      Eigen::Map<
        Eigen::Matrix<T, 3, 4, Eigen::RowMajor> const> T2(pose2_vec);

      Eigen::Matrix<T, 3, 1> pt1_transformed = T1 * pt1_.cast<T>();
      Eigen::Matrix<T, 3, 1> pt2_transformed = T2 * pt2_.cast<T>();

      // auto p1_sub_Tp2 = (- pt1_.head<3>().cast<T>() + pt2_transformed).transpose();
      T ell2 = *ell * (*ell);
      auto p1_sub_Tp2 = ( pt1_transformed - pt2_transformed);
      T d2 =  p1_sub_Tp2.squaredNorm() * (-0.5 / ell2 );
    
      //residuals[0] =  -2 * sigma2_ * exp(-d2);
      residuals[0] = (-2) * label_ip_ * d2;
      //residuals[0] = (-2.0) *  d2;

      //if (residuals) {
      //std::cout<<"\n\nnew iteration, T is \n"<<transformation<<std::endl;      
      //  residuals[0] =  (pt1_.cast<T>().head(3)-pt2_transformed).norm();
      //std::cout<<"residuals is "<<residuals[0]<<std::endl;
      //}
      //residuals[0] = (pt1.block<3,1>(0,0) - pt2_transformed).norm();

    
      return true;
    }

  private:
    //double pt1_[4];
    //double pt2_[4];
    Eigen::Vector4d pt1_;
    Eigen::Vector4d pt2_;
    //double ell2_;
    double sigma2_;
    double label_ip_;
  
  };

  class SelfPoseEllAutoDiffFunctor {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    SelfPoseEllAutoDiffFunctor(const Eigen::Vector3f & pt1,
                               const Eigen::Vector3f & pt2,
                               double label_ip,
                               double sigma
                               ) {

      pt1_sub_pt2_square_ = (pt1.cast<double>() - pt2.cast<double>()).squaredNorm();
      sigma2_ = sigma*sigma;
      label_ip_ = label_ip;
    
    }

    ~SelfPoseEllAutoDiffFunctor() {}

    template <typename T>
    bool operator()(const T * ell,
                    T* residuals) const {

      // auto p1_sub_Tp2 = (- pt1_.head<3>().cast<T>() + pt2_transformed).transpose();
      T ell2 = *ell * (*ell);
      T d2 =  pt1_sub_pt2_square_ * (-0.5 / ell2 );
    
      // residuals[0] = color_ip_ * sigma2_ * exp(-d2/(2*ell2_));
       residuals[0] = label_ip_ * exp(d2);
      //residuals[0] =  d2;

      //if (residuals) {
      //std::cout<<"\n\nnew iteration, T is \n"<<transformation<<std::endl;      
      //  residuals[0] =  (pt1_.cast<T>().head(3)-pt2_transformed).norm();
      //std::cout<<"residuals is "<<residuals[0]<<std::endl;
      //}
      //residuals[0] =es (pt1.block<3,1>(0,0) - pt2_transformed).norm();

    
      return true;
    }

  private:
    //double pt1_[4];
    //double pt2_[4];
    double pt1_sub_pt2_square_;
    //Eigen::Vector4d pt2_;
    // Eigen
    //double ell2_;
    double sigma2_;
    double label_ip_;
  
  };
  
  

  class
//#ifdef __CUDACC__
//  __align__(16)
//#else
//    alignas(16)
//#endif  
  PairwiseAnalyticalDiffFunctor : public ceres::SizedCostFunction <1, 12, 12>  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    PairwiseAnalyticalDiffFunctor(const Eigen::Vector3f & pt1,
                                  const Eigen::Vector3f & pt2,
                                  double label_ip,
                                  double ell,
                                  double sigma=0.1
                                  //,int num_pts_1,
                                  //int num_pts_2
                                  ) {
      pt1_.head<3>() = pt1.cast<double>();
      pt1_[3] = 1.0;
      pt2_.head<3>() = pt2.cast<double>();
      pt2_[3] = 1.0;
      ell2_ = ell*ell;
      sigma2_ = sigma*sigma;
      label_ip_ = label_ip;

      DT1 = Eigen::Matrix<double, 3, 12>::Zero();
      auto pt1_homo = pt1_.transpose();
      DT1.block<1,4>(0,0) = pt1_homo;
      DT1.block<1,4>(1,4) = pt1_homo;
      DT1.block<1,4>(2,8) = pt1_homo;

      DT2 = Eigen::Matrix<double, 3, 12>::Zero();      
      Eigen::Matrix<double, 1, 4> pt2_homo;
      pt2_homo.head<3>() =  pt2_.transpose().head<3>();
      pt2_homo(3) = 1;
      DT2.block<1,4>(0,0) = pt2_homo;
      DT2.block<1,4>(1,4) = pt2_homo;
      DT2.block<1,4>(2,8) = pt2_homo;

    
    }

    ~PairwiseAnalyticalDiffFunctor() {}

    bool Evaluate(const double * const * pose_vecs,
                  double * residuals,
                  double ** jacobians) const {

      Eigen::Map<
        Eigen::Matrix<double, 3, 4, Eigen::RowMajor> const> T1(pose_vecs[0]);
      Eigen::Map<
        Eigen::Matrix<double, 3, 4, Eigen::RowMajor> const> T2(pose_vecs[1]);

      Eigen::Matrix<double, 3, 1> pt1_transformed = T1 * pt1_;
      Eigen::Matrix<double, 3, 1> pt2_transformed = T2 * pt2_;

      //auto p1_sub_Tp2 = (- pt1_.head<3>().cast<T>() + pt2_transformed).transpose();
      auto p1_sub_Tp2 = ( pt1_transformed - pt2_transformed).transpose();
      double d2 = p1_sub_Tp2.squaredNorm();
    
      // residuals[0] = color_ip_ * sigma2_ * exp(-d2/(2*ell2_));
      if (residuals) {
        residuals[0] = label_ip_ * d2;
      } else
        return true;

      //if (residuals) {
      //std::cout<<"\n\nnew iteration, T is \n"<<transformation<<std::endl;      
      //  residuals[0] =  (pt1_.cast<T>().head(3)-pt2_transformed).norm();

      //}
      //residuals[0] = (pt1.block<3,1>(0,0) - pt2_transformed).norm();
      //std::cout<<"jacobians==null is "<<(jacobians==nullptr)<<std::endl<<std::flush;

      //if (jacobians)
      //  std::cout<<"jacobians[0]==null is "<<(jacobians[0]==nullptr)<<std::endl;
      if (!jacobians) return true;

      if (jacobians && jacobians[0] ) {
        
        Eigen::Map<Vec12d_row> jacob1(jacobians[0]);
        jacob1 =  (p1_sub_Tp2 * DT1);
        
      }

      if (jacobians && jacobians[1] ) {
        
        Eigen::Map<Vec12d_row> jacob2(jacobians[1]); 
        jacob2 = - p1_sub_Tp2 * DT2;
        
      }
      


    
      return true;
    }

  private:
    //double pt1_[4];
    //double pt2_[4];
    Eigen::Vector4d pt1_;
    Eigen::Vector4d pt2_;
    Eigen::Matrix<double, 3, 12> DT1;
    Eigen::Matrix<double, 3, 12> DT2;
    double ell2_;
    double sigma2_;
    double label_ip_;
    //    int pc1_size_;
    //int pc2_size_;
  };
  

}
