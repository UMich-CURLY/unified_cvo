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

}
