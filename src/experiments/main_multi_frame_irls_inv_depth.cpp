
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include "utils/def_assert.hpp"
#include "utils/CvoPointCloud.hpp"
#include "utils/ImageRGBD.hpp"
#include "utils/ImageStereo.hpp"
#include "utils/CvoPoint.hpp"
#include <cmath>
#include <cstdint>
#include <memory>
#include <opencv2/opencv.hpp>
#include "cvo/local_parameterization_se3.hpp"
#include <omp.h>
#include "utils/PoseLoader.hpp"
#include <ceres/local_parameterization.h>
#include <ceres/cubic_interpolation.h>
#include "sophus/se3.hpp"
#include <pcl/point_cloud.h>
#include <string>
#include <utility>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <cstdio>

#include "utils/Calibration.hpp"
#include <omp.h>
#include "dataset_handler/DataHandler.hpp"
#include "dataset_handler/TartanAirHandler.hpp"
#include "dataset_handler/KittiHandler.hpp"
#include "dataset_handler/TumHandler.hpp"
#include <Eigen/SparseCore>
#include <Eigen/Sparse>
//#include "cvo/nanoflann.hpp"
//#include "cvo/KDTreeVectorOfVectorsAdaptor.h"
#include <pcl/filters/voxel_grid.h>
#include <opencv2/core/eigen.hpp>
//#include "ceres/gradient_checker.h"
using namespace cvo;
enum class PoseFormat { TARTAN, KITTI, TUM };

const int NUM_CHANNELS = 3;

const std::vector<std::vector<double>> pixel_pattern = {
  {0, 0}, {-1, 0}, {-1, -1}, {-1, 1}, {0, 1}, {0, -1}, {1, 1}, {1, 0}};

typedef Eigen::Matrix<double, 4, 1, Eigen::RowMajor> Vec4d_row;
typedef Eigen::Matrix<double, 3, 1, Eigen::RowMajor> Vec3d_row;
typedef Eigen::Matrix<double, 3, 1> Vec3d;
typedef Eigen::Matrix<double, 1, 12> Vec12d_row;
typedef Eigen::Matrix<double, 3, 4, Eigen::RowMajor> Mat34d_row;
typedef Eigen::Matrix<float, 3, 4, Eigen::RowMajor> Mat34f_row;
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Mat44f_row;
typedef Eigen::Matrix<double, 3, 4, Eigen::ColMajor> Mat34d;
typedef Eigen::Matrix<double, 3, 3, Eigen::ColMajor> Mat33d;
typedef Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Mat33d_row;
typedef Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Mat44d_row;

/*typedef KDTreeVectorOfVectorsAdaptor<
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>,
  float>
  kd_tree_eigen;
*/
/*
  template <typename T>
  T dist2(T*a, T* b, int dim) {
  T dist = 0;
  for (int i = 0; i < dim; i++){
  dist += (a[i] - b[i]) * (a[i] - b[i]);
  }
  return dist;
  }V
*/

class Frame;
struct Pixel {
private:

  double u_;
  double v_;
  double inv_depth_;
  const Frame * frame_;
  //double features_[3];
public:
  Pixel(double u, double v, double inv_depth,
        const Frame * frame
        ) : frame_(frame) {
    u_ = u;
    v_ = v;
    inv_depth_ = static_cast<double>(inv_depth);
    //for (int i = 0 ; i < 3; i++) features_[i] = static_cast<double>(feature[i]);
  }
  double & inv_depth() { return inv_depth_;}
  double u() const { return u_; }
  double v() const { return v_; }
  Eigen::Vector3d uv() const { Eigen::Vector3d uv; uv << u_, v_, 1.0; return uv;}
  const Frame * host() const {return frame_;}
  //double * features() { return features_; }
};

// template <typename DType>

struct ImageColorEigen {
private:
  
  //Eigen::MatrixXd R_;
  //Eigen::MatrixXd G_;
  //Eigen::MatrixXd B_;
  //std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> color_;
  std::vector<double> color_;

  const cv::Mat raw_;

  std::unique_ptr<ceres::Grid2D<double, 3> > grid_;
  std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double, 3> > > interpolate_pixel_;
  
public:
  const unsigned int channels;
  const unsigned int rows;
  const unsigned int cols;

  
  ImageColorEigen(const cv::Mat & img_uchar) :
    raw_(img_uchar),
    rows(img_uchar.rows), cols(img_uchar.cols), channels(img_uchar.channels()){

    cv::Mat img;
    img_uchar.convertTo(img, CV_64FC3);
    img = img / 255.0;

    color_.resize(rows * cols * channels);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        cv::Vec3d pixel_cv = img.at<cv::Vec3d>(i, j);
        //cv::cv2eigen(pixel_cv, color_[i * cols + j]);
        color_[(i*cols+j)*channels] = pixel_cv(0);
        color_[(i*cols+j)*channels+1] = pixel_cv(1);
        color_[(i*cols+j)*channels+2] = pixel_cv(2);
      }
    }
    /*
      cv::Mat r, g, b;
      cv::extractChannel(img, b, 0); // extract specific channel
      cv::extractChannel(img, g, 1); // extract specific channel
      cv::extractChannel(img, r, 2); // extract specific channel
      cv::cv2eigen(r, R_);
      cv::cv2eigen(g, G_);
      cv::cv2eigen(b, B_);
    */
    grid_.reset(new ceres::Grid2D<double, 3>(
                                             &color_[0],
                                             0, img.rows,
                                             0, img.cols
                                             ));
    interpolate_pixel_.reset(
                             new ceres::BiCubicInterpolator<ceres::Grid2D<double, 3> >(*grid_));
  }
  //const Eigen::MatrixXd * r() const { return &R_; }
  double r(unsigned int row, unsigned int col) const { return color_[this->channels*(row*cols + col)+2]; }  
  
  // const Eigen::MatrixXd * g() { return &G_; }
  double g(unsigned int row, unsigned int col) const { return color_[this->channels*(row*cols + col)+1]; }
  
  //const Eigen::MatrixXd * b() { return &B_; }
  double b(unsigned int row, unsigned int col) const { return color_[this->channels*(row*cols + col)+0]; }

  Eigen::Vector3d  at(unsigned int row, unsigned int col) const {
    Eigen::Vector3d pixel;
    pixel << this->r(row, col), this->g(row, col), this->b(row, col);
    return pixel;
  }

  template <typename T>
  void interpolate(const T & row, const T & col, T * pixel) const {
    this->interpolate_pixel_->Evaluate(row, col, pixel);
  }

  //template <typename Dtype>
  //Eigen::Vector3d rgb(Dtype r, Dtype c) const {
  //  
  //}

  template <typename T>
  bool is_in_bound(const T& r,const T & c) const {
    for (int i = 0; i < pixel_pattern.size(); i++) {
      if ((int)(r+pixel_pattern[i][1]) < 1 ||
          (int)(r+pixel_pattern[i][1]) >= this->rows-1 ||
          (int)(c+pixel_pattern[i][0]) < 1 ||
          (int)(c+pixel_pattern[i][0]) >= this->cols-1 )
        return false;
    }
    return true;
  }

  const cv::Mat & get_raw() const { return raw_; }

  ~ImageColorEigen() {}
};

template <unsigned int NChannels>
struct Image {
private:
  
  //Eigen::MatrixXd R_;
  //Eigen::MatrixXd G_;
  //Eigen::MatrixXd B_;
  //std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> color_;
  std::vector<double> data_;

  const cv::Mat raw_;
  //const std::vector<float> gt_depth_;

  std::unique_ptr<ceres::Grid2D<double, NChannels> > grid_;
  std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double, NChannels> > > interpolate_pixel_;
  
public:
  //  const unsigned int channels;
  const unsigned int rows;
  const unsigned int cols;

  
  Image(const cv::Mat & img_uchar) :
    raw_(img_uchar),
    rows(img_uchar.rows),
    cols(img_uchar.cols) { 

    cv::Mat img;
    if (img_uchar.channels() == 3 && NChannels == 1) {
      cv::cvtColor(img_uchar, img, CV_BGR2GRAY);
      img.convertTo(img, CV_64FC(NChannels));
    } else {
      img_uchar.convertTo(img, CV_64FC(NChannels));
    }
    img = img / 255.0;
    if (img.isContinuous() == false) {
      img = img.clone();
    }

    data_.resize(rows*cols*NChannels);
    std::memcpy(data_.data(), img.data, rows*cols*NChannels*sizeof(double));

    grid_.reset(new ceres::Grid2D<double, NChannels>(
                                             &data_[0],
                                             0, rows,
                                             0, cols
                                             ));
    interpolate_pixel_.reset(
                             new ceres::BiCubicInterpolator<ceres::Grid2D<double, NChannels> >(*grid_));
  }

  Eigen::Matrix<double, NChannels, 1>  at(unsigned int row, unsigned int col) const {
    Eigen::Matrix<double, NChannels, 1>  pixel =
      Eigen::Map<Eigen::Matrix<double, NChannels, 1>>(&(data_[(row*cols+col)*NChannels]));
    return pixel;
  }

  double at(unsigned int row, unsigned int col, unsigned int channel) const {
    return data_[(row*cols+col)*NChannels+channel];
  }

  template <typename T>
  void interpolate(const T & row, const T & col, T * pixel) const {
    this->interpolate_pixel_->Evaluate(row, col, pixel);
  }

  //template <typename Dtype>
  //Eigen::Vector3d rgb(Dtype r, Dtype c) const {
  //  
  //}

  template <typename T>
  bool is_in_bound(const T& r,const T & c) const {
    for (int i = 0; i < pixel_pattern.size(); i++) {
      if ((int)(r+pixel_pattern[i][1]) < 1 ||
          (int)(r+pixel_pattern[i][1]) >= this->rows-1 ||
          (int)(c+pixel_pattern[i][0]) < 1 ||
          (int)(c+pixel_pattern[i][0]) >= this->cols-1 )
        return false;
    }
    return true;
  }

  const cv::Mat & get_raw() const { return raw_; }

  ~Image() {}
};

/*
struct Frame {
private:
  std::vector<Pixel> pixels_;
  //std::vector<double> inv_depths_;
  double pose_vec_[12];
  const ImageColorEigen img_;
  const unsigned int index;
public:
  void push_back(double u, double v, double inv_depth
                 //const double * feat
                 ) {
    pixels_.push_back({u, v, inv_depth, this});
    //inv_depths_.push_back(inv_depth);
  }
  Frame(const unsigned int new_index,
        const cv::Mat & new_img):
    index(new_index), img_(new_img){
  }
  //Frame() {}
  ~Frame() {}
  void set_pose(const double * pose_in) {
    std::memcpy(pose_vec_, pose_in, sizeof(double)*12);
  }
  size_t size() const {return pixels_.size(); }
  double & depth(unsigned int i) {
    assert(i >=0 && i < this->size());
    return pixels_[i].inv_depth();
  };
  Pixel & pixel(unsigned int i) {
    assert(i>=0 && i < this->size());
    return pixels_[i];
  }
  double * pose_vec(){ return pose_vec_; }
  const double * pose_vec() const { return pose_vec_; }
  const ImageColorEigen * img() const { return &img_; }
};
*/

struct Frame {
private:
  std::vector<Pixel> pixels_;
  //std::vector<double> inv_depths_;
  double pose_vec_[12];
  const Image<NUM_CHANNELS> img_;

public:
  const unsigned int index;  
  void push_back(double u, double v, double inv_depth
                 //const double * feat
                 ) {
    pixels_.push_back({u, v, inv_depth, this});
    //inv_depths_.push_back(inv_depth);
  }
  Frame(const unsigned int new_index,
        const cv::Mat & new_img):
    index(new_index), img_(new_img){
  }
  //Frame() {}
  ~Frame() {}
  void set_pose(const double * pose_in) {
    std::memcpy(pose_vec_, pose_in, sizeof(double)*12);
  }
  size_t size() const {return pixels_.size(); }
  double & depth(unsigned int i) {
    assert(i >=0 && i < this->size());
    return pixels_[i].inv_depth();
  };
  Pixel & pixel(unsigned int i) {
    assert(i>=0 && i < this->size());
    return pixels_[i];
  }
  double * pose_vec(){ return pose_vec_; }
  const double * pose_vec() const { return pose_vec_; }
  const Image<NUM_CHANNELS> * img() const { return &img_; }
};


template <typename T, unsigned int RC_MAJOR>
Eigen::Matrix<T, 3, 1> projection(const Eigen::Matrix<T, 3, 4, RC_MAJOR> & T12,
                                  const Eigen::Matrix<T, 3, 1> & uv2_homo,
                                  T inv_depth_2,
                                  const Mat33d_row & K,
                                  const Mat33d_row & inv_K) {

  Eigen::Matrix<T, 3, 1, RC_MAJOR> p2_in_2 = (inv_K * uv2_homo) . template cast<T>() / inv_depth_2;
  //Eigen::Matrix<T, 4, 1> p2_in_2_hom;
  //p2_in<< p2_in_2(0), p2_in_2(1), p2_in_2(2), static_cast<T>(1.0);
  typename Eigen::Matrix<T,3,1> uv_in_1 = K * (T12. template block<3,3>(0,0) * p2_in_2 + T12. template block<3,1>(0,3));
  uv_in_1 = (uv_in_1 / uv_in_1(2)).eval();
  return uv_in_1;
}

template <typename T, unsigned int RC_MAJOR>
Eigen::Matrix<T, 3, 1> projection(const Eigen::Matrix<T, 3, 4, RC_MAJOR> & T1,
                                  const Eigen::Matrix<T, 3, 4, RC_MAJOR> & T2,
                                  const Eigen::Matrix<T, 3, 1> & uv2_homo,
                                  T inv_depth_2,
                                  const Mat33d_row & K,
                                  const Mat33d_row & inv_K) {

  typename Eigen::Matrix<T, 3, 1> p2_in_2 = (inv_K * uv2_homo) . template cast<T>() / inv_depth_2;
  
  typename Eigen::Matrix<T, 4, 4, RC_MAJOR> T1_44 = Eigen::Matrix<T, 4, 4, RC_MAJOR>::Identity();
  T1_44. template block<3,4>(0,0) = T1;
  typename Eigen::Matrix<T, 4, 4, RC_MAJOR> T2_44 = Eigen::Matrix<T, 4, 4, RC_MAJOR>::Identity();
  T2_44. template block<3,4>(0,0) = T2;
  typename Eigen::Matrix<T, 4, 4, RC_MAJOR> T_1_w = T1_44. template inverse();
  typename Eigen::Matrix<T, 4, 4, RC_MAJOR> T12 = T_1_w * T2_44;
  
  //Eigen::Matrix<T, 4, 1> p2_in_2_hom;
  //p2_in<< p2_in_2(0), p2_in_2(1), p2_in_2(2), static_cast<T>(1.0);
  typename Eigen::Matrix<T,3,1> uv_in_1 = K * (T12. template block<3,3>(0,0) * p2_in_2 + T12. template block<3,1>(0,3));
  uv_in_1 = (uv_in_1 / uv_in_1(2)).eval();
  return uv_in_1;
}


/*
  class ClosestPointAutoDiffFunctor {
  public:
  ClosestPointAutoDiffFunctor(Point * pt1,
  Point * pt2,
  double weight,
  double ell,
  double sigma,
  const Eigen::Matrix<double,3,3, Eigen::RowMajor> * inv_intrinsics
                              
  )  : inv_K (inv_intrinsics) 
  uv1_(0) =(double) pt1->u;
  uv1_(1) =(double) pt1->v;
  uv1_(2) = 1.0;
  inv_depth_1_ = pt1->init_inv_depth;
  uv2_(0) =(double) pt2->u;
  uv2_(1) =(double) pt2->v;
  uv2_(2) = 1.0;
  inv_depth_2_ = pt2->init_inv_depth;
  ell2_ = ell*ell;
  sigma2_ = sigma*sigma;
  weight_ = weight;

    
  }

  template <typename T>
  bool operator()(T const * const * parameters,
  //const T* pose_vec_1,
  T* residuals) const {

  //Eigen::Map<const Eigen::Matrix<T, 3, 1>> pt1(pt1_);
  //Eigen::Map<const Eigen::Matrix<T, 4, 1>> pt2(pt2_);
  const T * pose_vec1 = parameters[0];
  const T* pose_vec2  = parameters[1];
  //const T* inv_depth_1 = parameters[2];
  //const T* inv_depth_2 = parameters[3];

  Eigen::Map< typename
  Eigen::Matrix<T, 3, 4, Eigen::RowMajor> const> transformation1(pose_vec1);
  typename Eigen::Matrix<T, 4, 4, Eigen::RowMajor> T1 = Eigen::Matrix<T, 4, 4, Eigen::RowMajor>::Identity();
  T1. template block<3,4>(0,0) = transformation1;
    
  Eigen::Map< typename
  Eigen::Matrix<T, 3, 4, Eigen::RowMajor> const> transformation2(pose_vec2);
  typename Eigen::Matrix<T, 4, 4, Eigen::RowMajor> T2 = Eigen::Matrix<T, 4, 4, Eigen::RowMajor>::Identity();
  T2. template block<3,4>(0,0) = transformation2;

  typename Eigen::Matrix<T, 4, 4, Eigen::RowMajor> T12 = T1.inverse() * T2;

  Eigen::Matrix<T, 3, 1, Eigen::ColMajor> p_in_2 = ((*inv_K) * uv2_) . template cast<T>() / (*inv_depth_2);
  Eigen::Matrix<T, 4, 1> p_in_2_hom;
  p_in_2_hom << p_in_2(0), p_in_2(1), p_in_2(2), T(1);

  typename Eigen::Matrix<T,3,1> XYZ2 = (T12 * p_in_2_hom).head(3);
  typename Eigen::Matrix<T,3,1> XYZ1 =  (*inv_K). template cast<T>() *
  (( uv1_.template cast<T> ()/ (*inv_depth_1) )) ;
    
  T d2  = (- XYZ1 + XYZ2). template squaredNorm() ;
  T regularizer = (*lambda_) *  ( static_cast<T>(inv_depth_1_) - (*inv_depth_1) ) * ( static_cast<T>(inv_depth_1_) - (*inv_depth_1) )
  + (*lambda_) *  ( static_cast<T>(inv_depth_2_) - (*inv_depth_2) ) *  ( static_cast<T>(inv_depth_2_) - (*inv_depth_2) );
  // T regularizer = (*lambda_) * std::abs( static_cast<T>(inv_depth_1_) - (*inv_depth_1) )
  //  + (*lambda_) * std::abs ( static_cast<T>(inv_depth_2_) - (*inv_depth_2) ) ;
    
  residuals[0] = weight_ * d2  + regularizer;
    

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
  //Eigen::Matrix<double, 3, 1, Eigen::RowMajor> uv1_;
  Eigen::Vector3d uv1_;
  double inv_depth_1_;
  //Eigen::Matrix<double, 3, 1, Eigen::RowMajor> uv2_;
  Eigen::Vector3d uv2_;
  double inv_depth_2_;
  double ell2_;
  double sigma2_;
  double weight_;
  const double * lambda_;
  const Mat33d_row * inv_K;
  const cv::Mat * img1;
  const cv::Mat * img2;
  };
*/




template <unsigned int NChannels>
class InvDepthAutoDiffFunctor {
public:
  InvDepthAutoDiffFunctor(Pixel * pt2,
                          double weight,
                          const Image<NChannels> * img1,
                          const Image<NChannels> * img2,
                          //const double * lambda,
                          const Eigen::Matrix<double,3,3, Eigen::RowMajor> * inv_intrinsics,
                          const Eigen::Matrix<double,3,3, Eigen::RowMajor> * intrinsics
                          ) : inv_K (inv_intrinsics),
                              K(intrinsics),
                              img1_(img1),
                              img2_(img2){

    uv2_ << (double) pt2->u(), (double) pt2->v(), 1.0;

    //inv_depth_1_ = pt1->init_inv_depth;
    //ell2_ = ell*ell;
    //sigma2_ = sigma*sigma;
    weight_ = weight;
    //colors_.resize(pixel_pattern.size());
    //for (int i = 0; i < pixel_pattern.size(); i++) {

    //   int c2 = (int)(uv2_(0)+0.5 + pixel_pattern[i][0]);
    //  int r2 = (int)(uv2_(1)+0.5 + pixel_pattern[i][1]);

      
      
      //colors_[i] = img2_->at(r2, c2);
    //}
  }

  template <typename T>
  bool operator()(T const * const * parameters,
                  //const T* pose_vec_1,
                  T* residuals) const {

    //std::cout<<"Processing img "<<img1_<<" and "<<img2_<<"\n";

    //Eigen::Map<const Eigen::Matrix<T, 3, 1>> pt1(pt1_);
    //Eigen::Map<const Eigen::Matrix<T, 4, 1>> pt2(pt2_);
    const T * pose_vec1 = parameters[0];
    const T* pose_vec2  = parameters[1];
    const T* inv_depth_2 = parameters[2];
    //const T* inv_depth_2 = parameters[3];

    Eigen::Map< typename
                Eigen::Matrix<T, 3, 4, Eigen::RowMajor> const> transformation1(pose_vec1);
    typename Eigen::Matrix<T, 4, 4, Eigen::RowMajor> T1 = Eigen::Matrix<T, 4, 4, Eigen::RowMajor>::Identity();
    T1. template block<3,4>(0,0) = transformation1;
    
    Eigen::Map< typename
                Eigen::Matrix<T, 3, 4, Eigen::RowMajor> const> transformation2(pose_vec2);
    typename Eigen::Matrix<T, 4, 4, Eigen::RowMajor> T2 = Eigen::Matrix<T, 4, 4, Eigen::RowMajor>::Identity();
    T2. template block<3,4>(0,0) = transformation2;

    typename Eigen::Matrix<T, 4, 4, Eigen::RowMajor> T12 = T1.inverse() * T2;

    Eigen::Matrix<T, 3, 1, Eigen::ColMajor> p2_in_2 = ((*inv_K) * uv2_) . template cast<T>() / (*inv_depth_2);
    Eigen::Matrix<T, 4, 1> p2_in_2_hom;
    p2_in_2_hom << p2_in_2(0), p2_in_2(1), p2_in_2(2), static_cast<T>(1.0);
    typename Eigen::Matrix<T,3,1> uv_in_1 = (*K) * (T12 * p2_in_2_hom).head(3);
    uv_in_1 = (uv_in_1 / uv_in_1(2)).eval();


    std::vector<T> depth_residuals(pixel_pattern.size());
    for (int i = 0; i < pixel_pattern.size(); i++) {
      //T d2  = //(- XYZ1 + XYZ2). template squaredNorm() ;
      double offset_c = pixel_pattern[i][0];
      double offset_r = pixel_pattern[i][1];
      
      //auto c1 = uv_in_1(0) + (offset_c). template cast<T>();
      auto c1 = uv_in_1(0) + static_cast<T>(offset_c);//. template cast<T>();
      auto r1 = uv_in_1(1) + static_cast<T>(offset_r);//. template cast<T>();
      std::vector<T> color1(NChannels);
      img1_->interpolate(r1, c1, color1.data());
      if (r1 < 0 || r1 >= this->img1_->rows || c1 < 0 || c1 > this->img1_->cols  ) {
        //std::cout<<"Evaluate depth residual failed because of ofb, r1="<<r1<<", c1="<<c1<<"\n";
        residuals[0] = static_cast<T>(9.0 * pixel_pattern.size())*weight_;
        return true;
      }

      auto c2 = static_cast<T>(uv2_(0) + pixel_pattern[i][0]);///.template cast<T>();
      auto r2 = static_cast<T>(uv2_(1) + pixel_pattern[i][1]);//.template cast<T>();
      std::vector<T> color2(NChannels);
      img2_->interpolate(r2, c2,  color2.data());

      Eigen::Matrix<T, NChannels, 1> dist = Eigen::Map< Eigen::Matrix<T, NChannels, 1>>(color1.data()) - Eigen::Map< Eigen::Matrix<T, NChannels, 1>>(color2.data());
      //depth_residuals[i] = weight_ * ((color1 - colors_[i])).squaredNorm();
      depth_residuals[i] = weight_ * dist.squaredNorm(); //dist2(color1, color2, 3);  //.squaredNorm();
      //std::cout<<"Evaluate depth residual of pattern i: "<<depth_residuals[i]<<", inv depth is "<<*inv_depth_2<<", ";
      //std::cout<<"r1="<<r1<<", c1="<<c1<<"\n";
        
    }


    
    //T regularizer = (*lambda_) *  ( static_cast<T>(inv_depth_1_) - (*inv_depth_1) ) * ( static_cast<T>(inv_depth_1_) - (*inv_depth_1) )
    //   + (*lambda_) *  ( static_cast<T>(inv_depth_2_) - (*inv_depth_2) ) *  ( static_cast<T>(inv_depth_2_) - (*inv_depth_2) );
    residuals[0] = static_cast<T>(0.0);
    //std::cout<<"Residuals: "<<residuals[0]<<". ";    
    for (int i = 0; i < depth_residuals.size(); i++){
      //std::cout<<"residual[0]+="<<depth_residuals[i]<<" from "<<i<<"-th residual\n";
      residuals[0] += depth_residuals[i];
    }
    //std::cout<<" updated to: "<<residuals[0]<<". ";    
    //residuals[0] = weight_ * depth_residual;
    

    //if (residuals) {
    //std::cout<<"\n\nnew iteration, T is \n"<<transformation<<std::endl;      
    //  residuals[0] =  (pt1_.cast<T>().head(3)-pt2_transformed).norm();
    //std::cout<<"residuals is "<<residuals[0]<<std::endl;
    //}
    //residuals[0] = (pt1.block<3,1>(0,0) - pt2_transformed).norm();

    //std::cout<<"End evaluate jacob\n";
    return true;
  }

private:
  //double pt1_[4];
  //double pt2_[4];
  //Eigen::Matrix<double, 3, 1, Eigen::RowMajor> uv1_;
  //Eigen::Vector3d uv1_;
  // double inv_depth_1_;
  //Eigen::Matrix<double, 3, 1, Eigen::RowMajor> uv2_;
  Eigen::Vector3d uv2_;
  double inv_depth_2_;
  //double ell2_;
  //double sigma2_;
  double weight_;
  //const double * lambda_;
  const Mat33d_row * inv_K;
  const Mat33d_row * K;
  //const ImageColorEigen * img1_;
  //const ImageColorEigen * img2_;
  //std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> colors_;
  const Image<NChannels> * img1_;
  const Image<NChannels> * img2_;
  //std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> colors_;
};


void write_transformed_pc(std::map<int, std::shared_ptr<Frame>> & frames,
                          const Mat33d_row * inv_K,
                          std::string & fname,
                          bool is_filter_outlier=true,
                          std::map<int, std::vector<bool>> * inlier_map=nullptr) {
  pcl::PointCloud<pcl::PointXYZRGB> pc_all;
  //for (int i = 0 ; i < frames.size(); i++) {
  for (auto && [i, frame]: frames) {

    Mat34d_row pose = Mat34d_row::Identity();
    pose.block<3,4>(0,0) = Eigen::Map<Mat34d_row>(frames[i]->pose_vec());
    
    Mat34f_row pose_f = pose.cast<float>();
    //std::cout<<__func__<<" pose: "<<pose_f<<"\n";
    
    Mat44f_row T_f = Mat44f_row::Identity();
    
    T_f.block<3,4>(0,0) = pose_f;
    T_f = T_f.inverse().eval();
    for (int j = 0; j < frame->size(); j++ ){

      double u = frames[i]->pixel(j).u();
      double v = frames[i]->pixel(j).v();
      double d = 1 / frames[i]->pixel(j).inv_depth();
      Eigen::Vector3d uvd;
      uvd << u*d, v*d, d;
      Eigen::Vector3d local_p = *inv_K * uvd;
      
      auto p_curr = pose_f.block<3,3>(0,0) * local_p.cast<float>() + pose_f.block<3,1>(0,3);


      if (is_filter_outlier && p_curr.norm() > 80 )
        continue;

      if (inlier_map && (*inlier_map)[i][j] == false)
        continue;
      
      pcl::PointXYZRGB p;
      p.getVector3fMap() = p_curr.cast<float>();

      int r = (int)(v+0.5);
      int c = (int)(u+0.5);
      cv::Vec3b bgr = frame->img()->get_raw().at<cv::Vec3b>(r, c);
      p.b = bgr(0);
      p.g = bgr(1);
      p.r = bgr(2);
      //p.b = (uint8_t)(frame->img()->b(r, c) * 255.0);
      //p.g = (uint8_t)(frame->img()->g(r, c) * 255.0);
      //p.r = (uint8_t)(frame->img()->r(r, c) * 255.0);
      pc_all.push_back(p);
    }

  }
  pcl::io::savePCDFileASCII(fname, pc_all);
}


cv::Rect decide_patch_boundary(int rows, int cols,
                               int u, int v,
                               int patch_size) {

  int up, down,
    left, right;
  if (u - patch_size < 0) {
    left = 0;
    right = patch_size * 2;
  } else if (u + patch_size >= cols) {
    right = cols - 1;
    left = cols - patch_size * 2 - 1;
  } else {
    left = u - patch_size;
    right = u + patch_size;
  }

  if (v - patch_size < 0) {
    up = 0;
    down = patch_size * 2;
  } else if (v + patch_size >= rows) {
    down = rows - 1;
    up = rows - patch_size * 2 - 1;
  } else {
    up = v - patch_size;
    down = v + patch_size;
  }
  return cv::Rect(left, up, patch_size * 2+1, patch_size * 2 + 1);
}

void visualize_pixel(Pixel * pixel,
                     const std::map<int, std::shared_ptr<Frame>> * frames,
                     const Mat33d_row * K,
                     const Mat33d_row * inv_K,
                     const std::string & window_name,

                     int patch_size_half=50,
                     double sigma=0.1,
                     std::string comments=""                     
                     ) {

  //std::map<int, cv::Mat> vis;

  if (pixel == nullptr || frames == nullptr || K == nullptr || inv_K == nullptr)
    return;
  std::vector<cv::Mat> vis;
  const Frame * host = pixel->host();

  int rows = frames->begin()->second->img()->rows;
  int cols = frames->begin()->second->img()->cols;
    
  for (auto && [ind, frame]: *frames) {
    if (frame.get() == pixel->host()) {
      cv::Mat curr = frame->img()->get_raw().clone();
      cv::circle( curr,
                  cv::Point(pixel->u(), pixel->v()),
                  2.0, //curr.rows / 32,
                  cv::Scalar( 0, 0, 255 ),
                  cv::FILLED,
                  cv::LINE_8 );
      //vis.push_back( curr);
      cv::Rect rect = decide_patch_boundary(rows, cols,
                                            pixel->u(), pixel->v(),
                                            patch_size_half);
      curr = curr(rect).clone();
      cv::putText(curr, std::string("p: (")+
                  std::to_string((int)(pixel->u())) + ","+
                  std::to_string((int)(pixel->v()))+")" ,
                  cv::Point(10,10), CV_FONT_HERSHEY_PLAIN, 1.0 ,CV_RGB(0,0,250));
      
      vis.push_back( curr);
      //pixel_centers
    } else {
      Mat34d_row pose =  Eigen::Map<Mat34d_row>(frame->pose_vec());
      Mat34d_row pose_host =  Eigen::Map<const Mat34d_row>(host->pose_vec());
      Eigen::Vector3d curr_p_in_1 = projection<double, Eigen::RowMajor>(pose,
                                                                        pose_host,
                                                                        pixel->uv(),
                                                                        pixel->inv_depth(),
                                                                        *K,
                                                                        *inv_K
                                                                        );
      
      cv::Mat curr = frame->img()->get_raw().clone();
      cv::circle( curr,
                  cv::Point(curr_p_in_1(0), curr_p_in_1(1)),
                  2.0, //curr.rows / 32,
                  cv::Scalar( 0, 0, 255 ),
                  cv::FILLED,
                  cv::LINE_8 );

      Eigen::Vector3d line_start_in_1 = projection<double, Eigen::RowMajor>(pose,
                                                                            pose_host,
                                                                            pixel->uv(),
                                                                            pixel->inv_depth()+sigma,
                                                                            *K,
                                                                            *inv_K
                                                                            );

      Eigen::Vector3d line_end_in_1 = projection<double, Eigen::RowMajor>(pose,
                                                                          pose_host,
                                                                          pixel->uv(),
                                                                          pixel->inv_depth()-sigma,
                                                                          *K,
                                                                          *inv_K
                                                                          );
 
      
 
      cv::line( curr,
                cv::Point(line_start_in_1(0), line_start_in_1(1)),
                cv::Point(line_end_in_1(0), line_end_in_1(1)),
                cv::Scalar( 0, 255, 0 ),
                1,
                cv::LINE_8);
      cv::Rect rect = decide_patch_boundary(rows, cols,
                                            curr_p_in_1(0), curr_p_in_1(1),
                                            patch_size_half);
      curr = curr(rect).clone();
      cv::putText(curr, std::string("p: (")+
                  std::to_string((int)curr_p_in_1(0)) + ","+ std::to_string((int)curr_p_in_1(1))+")" ,
                  cv::Point(10,10), CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,250));
      
      vis.push_back( curr);
    }
  }

  cv::Mat total(patch_size_half*2+1,//pixel->host()->img()->rows * (frames->size()),
                (patch_size_half*2+1) * frames->size(), //pixel->host()->img()->cols,
                CV_8UC3);
  for (int i = 0; i < vis.size(); i++) {
    vis[i]
      .copyTo(total(cv::Rect(i*(patch_size_half*2+1),0,
                             patch_size_half*2+1, patch_size_half*2+1)));
                             //pixel->host()->img()->cols,
                             //pixel->host()->img()->rows)));
  }
  if (comments.size())
    cv::putText(total, comments,
                cv::Point(50,patch_size_half*2), CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,250));
  
  cv::resize(total, total, cv::Size(), 2.0, 2.0);
  cv::imshow(window_name,total);
  cv::waitKey(0);
  
  
  
}

class LoggingCallback : public ceres::IterationCallback {
public:
  explicit LoggingCallback(std::map<int, std::shared_ptr<Frame>> * frames,
                           Pixel * pixel,
                           const Mat33d_row * K,
                           const Mat33d_row * inv_K,
                           double weight,
                           bool is_vis=false
                           )
    : frames_(frames), pixel_(pixel), K(K), inv_K(inv_K), is_vis_(is_vis), weight_(weight) {}

  ~LoggingCallback() {}

  ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) {
    //const char* kReportRowFormat =
    //  "% 4d: f:% 8e d:% 3.2e g:% 3.2e h:% 3.2e "
    //  "rho:% 3.2e mu:% 3.2e eta:% 3.2e li:% 3d";
    std::cout<<"Iter: "<<summary.iteration
             <<", cost: "<<summary.cost
             <<", cost_change: "<<summary.cost_change
             <<", param: "<<pixel_->inv_depth()
             <<"\n";
    /*
      std::string output = StringPrintf(kReportRowFormat,
      summary.iteration,
      summary.cost,
      summary.cost_change,
      summary.gradient_max_norm,
      summary.step_norm,
      summary.relative_decrease,
      summary.trust_region_radius,
      summary.eta,
      summary.linear_solver_iterations);
      std::cout << output << std::endl;
    */
    if (is_vis_) {
      if (summary.iteration == 0)
        cv::destroyAllWindows();

      std::string window_name = std::to_string((uintptr_t)(pixel_));// +
      //  std::string(": Iter ") + std::to_string(summary.iteration) +
      //  ", cost: "+ std::to_string(summary.cost_change);
      
      visualize_pixel(pixel_, frames_, K, inv_K, window_name, 50, sqrt(1/weight_));
    }
    std::string after_name(std::to_string(summary.iteration)+"_iter.pcd");
    write_transformed_pc(*frames_, inv_K, after_name, true);//, &inlier_filter);
    
    
    
    return ceres::SOLVER_CONTINUE;
  }

private:
  std::map<int, std::shared_ptr<Frame>> * frames_;
  Pixel * pixel_;
  const Mat33d_row * K;
  const Mat33d_row * inv_K;
  bool is_vis_;
  double weight_;
   
};





void pcl_to_eigen(const pcl::PointCloud<pcl::PointXYZRGB> & pc1_pcl,
                  // output
                  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &pc1,
                  Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> color_pc1
                  )  {
  pc1.resize(pc1_pcl.size());

#pragma omp parallel for num_threads(24)
  for (int i = 0; i < pc1_pcl.size(); i++){
    auto & pt = pc1_pcl[i];
    pc1[i] << pt.x, pt.y, pt.z;
    //if (i = 100)
    //  std::cout<<"rgb is "<<(int)pt.r<<", "<<(int)pt.g<<", "<<pt.b<<std::endl;
    color_pc1.col(i) << (double)pt.r / 256, (double)pt.g / 256, (double)pt.b / 256;
  }
}

void transform_pcd(const Eigen::Matrix<double, 3, 4, Eigen::RowMajor> & transform,
                   const std::vector<Eigen::Vector3d,
                   Eigen::aligned_allocator<Eigen::Vector3d>> & cloud_y_init,
                   // output
                   std::vector<Eigen::Vector3d,
                   Eigen::aligned_allocator<Eigen::Vector3d> > & cloud_y
                   )  {
  int num_pts = cloud_y_init.size();
#pragma omp parallel for num_threads(24)
  for (int i = 0; i < num_pts; i++ ){
    (cloud_y)[i] = transform.block<3,3>(0,0)*cloud_y_init[i]+transform.block<3,1>(0,3);
  }
}
/*
  void se_kernel(// input
  //const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &pc1,
  //const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &pc2,
  const pcl::PointCloud<pcl::PointXYZRGB>  & pc1,
  //const pcl::PointCloud<pcl::PointXYZ> & pc2,
  const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &pc2,
  const std::vector<cvo::CvoPoint> & color_pc1,
  const std::vector<cvo::CvoPoint> & color_pc2,
  const kd_tree_eigen * mat_index,
  double ell,
  double sp_thresh,
  double sigma2,
  // output
  Eigen::SparseMatrix<double, Eigen::RowMajor> & ip_mat
  //,std::unordered_map<std::unordered_map<Vec12d_row>> & jacob
               
  //double * sum_residual
  //, mstd::vector<Eigen::Triplet<double>> & nonzero_list
  ) {

  const double c_ell = 15.0 / 255.0;    
  const double d2_thresh = -2 * ell * ell * log(sp_thresh);
  const double d2_c_thresh = -2.0*c_ell*c_ell*log(sp_thresh);    
  const double s2 = sigma2;
     

  unsigned int num_moving = pc2.size();
  unsigned int num_fixed = pc1.size();

  int counter = 0;
  #pragma omp parallel for num_threads(24)
  for (int idx = 0; idx < num_moving; idx++) {

  const float search_radius = d2_thresh;
  std::vector<std::pair<size_t,float>>  ret_matches;
  nanoflann::SearchParams params;
  Eigen::Vector3f pc2_f;
  pc2_f(0) = pc2[idx](0);
  pc2_f(1) = pc2[idx](1);
  pc2_f(2) = pc2[idx](2);
    
    
  const size_t nMatches = mat_index->index->radiusSearch(pc2_f.data(), search_radius, ret_matches, params);

  for(size_t j=0; j<nMatches; ++j){
  int i = ret_matches[j].first;
  double d2 =(double) ret_matches[j].second;
  double k = 1;
  double ck = 1;
  double a = 1;
  if(d2<d2_thresh){
  const Eigen::Vector3d feature_a = Eigen::Map<const Eigen::Vector3d>(color_pc1[i].feature);
  const Eigen::Vector3d feature_b = Eigen::Map<const Eigen::Vector3d>(color_pc2[idx].feature);
  double d2_color = (feature_a-feature_b).squaredNorm();
            
  if(d2_color<d2_c_thresh){
  k = s2*exp(-d2/(2.0*ell*ell));
  //k = exp(-d2/(2.0*ell*ell));
            
  ck = exp(-d2_color/(2.0*c_ell*c_ell));
  a = ck*k;

  if (a > sp_thresh){
  double a_residual = ck * k * d2 ; // least square residual
  double a_gradient = ck * k ; // weight of the least square
  #pragma omp critical
  {
  //nonzero_list.push_back(Eigen::Triplet<double>(i, idx, a));
  //ip_mat_gradient_prefix.insert(i, idx) = a_gradient;
  ip_mat.insert(i, idx) = a_gradient;
  //ip_mat.insert(i, idx) = ck;

  auto p1_sub_Tp2 = (- pc1[i] + pc2[idx]).transpose();
                
  Vec4d_row pt2_homo;
  pt2_homo << pc2[idx][0], pc2[idx][1], pc2[idx][2],1;
                
  Eigen::Matrix<double, 3, 12, Eigen::RowMajor> DT = Eigen::Matrix<double, 3, 12, Eigen::RowMajor>::Zero();
  DT.block<1,4>(0,0) = pt2_homo;
  DT.block<1,4>(1,4) = pt2_homo;
  DT.block<1,4>(2,8) = pt2_homo;
                
  Vec12d_row jacob_this = p1_sub_Tp2 * DT;
  if (jacob.find(i) == jacob.end()) 
  jacob[i] = std::unordered_map<Vec12d_row>();
  jacob[i][idx] = jacob_this;



              
  }
  }
          
  }
  }
  }
  }
  ip_mat.makeCompressed();
  
  }
*/  
/*
  void pcl_to_pixel_with_invdepth (const Mat33d_row * K,
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> & pcs,
  // output
  std::vector<std::vector<Point>> & pixels,
  std::vector<std::vector<double>> & inv_depths
  ) {

  pixels.resize(pcs.size());
  inv_depths.resize(pcs.size());

  for (int i = 0 ; i < pcs.size(); i++) {
  auto & pc = *pcs[i];
  for (int j = 0; j < pc.size(); j++ ) {
  Eigen::Vector3f xyz = pc[j].getVector3fMap();
  Eigen::Vector3d uvd = *K * xyz.cast<double>();
  Point x;
  x.u = uvd(0) / uvd(2);
  x.v = uvd(1) / uvd(2);
  std::uint32_t rgb = *reinterpret_cast<int*>(&pc[j].rgb);
  std::uint8_t r = (rgb >> 16) & 0x0000ff;
  std::uint8_t g = (rgb >> 8)  & 0x0000ff;
  std::uint8_t b = (rgb)       & 0x0000ff;
  x.feature[0] = ((double) r) / 255,0;
  x.feature[1] = ((double) g) / 255.0;
  x.feature[2] = ((double) b) / 255.0;
  x.init_inv_depth = 1.0 / uvd(2);
  inv_depths[i].push_back(1.0/uvd(2));
  pixels[i].push_back(x);
  }
  }
  
  
  }
*/


void pose_snapshot(const std::vector<std::vector<double>> & poses,
                   std::vector<Sophus::SE3d> & poses_curr ) {
    
  poses_curr.resize(poses.size());
  for (int i = 0; i < poses.size(); i++) {
    const Mat34d_row pose_eigen = Eigen::Map<const Mat34d_row>(poses[i].data());
    Sophus::SE3d pose(pose_eigen.block<3,3>(0,0), pose_eigen.block<3,1>(0,3));
    poses_curr[i] = pose;
  }
}    



double change_of_all_poses(std::vector<Sophus::SE3d> & poses_old,
                           std::vector<Sophus::SE3d> & poses_new) {
  
  double change = 0;
  
  for (int i = 0; i < poses_old.size(); i++) {
    
    change += (poses_old[i].inverse() * poses_new[i]).log().norm();
    
  }
  
  return change;
}

void epipolar_tracing(const std::map<int, std::shared_ptr<Frame>> & frames,
                      const Mat33d_row * K,
                      const Mat33d_row * inv_K,
                      Pixel * target,
                      double sigma,
                      double step_size,
                      std::vector<double> & loss,
                      bool is_log=false
                      ) {
  std::vector<double> inv_deps;
  for (double d_inv = target->inv_depth() - sigma; d_inv < target->inv_depth()+sigma; d_inv+=step_size) {
    double loss_curr = 0;    
    for (auto && [ind, frame] : frames) {
      if (ind == target->host()->index) continue;
      
      const Mat34d_row T1 = Eigen::Map<Mat34d_row>(frame->pose_vec());
      const Mat34d_row T2 = Eigen::Map<const Mat34d_row>(target->host()->pose_vec());
      Eigen::Vector3d uv1 = projection<double, Eigen::RowMajor>(T1, T2, target->uv(), d_inv, *K, *inv_K);
      Eigen::Vector3d uv2 = target->uv();

      for (int i = 0; i < pixel_pattern.size(); i++) {
        
        auto c2 = (uv2(0) + pixel_pattern[i][0]);///.template cast<T>();
        auto r2 = (uv2(1) + pixel_pattern[i][1]);//.template cast<T>()
        Eigen::VectorXd feat2(target->host()->img()->get_raw().channels());
        target->host()->img()->interpolate(r2, c2, feat2.data());
        
        auto c1 = (uv1(0) + pixel_pattern[i][0]);///.template cast<T>();
        auto r1 = (uv1(1) + pixel_pattern[i][1]);//.template cast<T>();
        Eigen::VectorXd feat1(target->host()->img()->get_raw().channels());
        frame->img()->interpolate(r1, c1,  feat1.data());
        loss_curr += (feat2 - feat1).squaredNorm();
      }
    }
    loss.push_back(loss_curr);
    inv_deps.push_back(d_inv);
  }
  if (is_log) {
    std::ofstream loss_f(std::string("inv_depth_loss.txt"));
    for (int i = 0; i < loss.size(); i++) {
      loss_f << inv_deps[i]<<", "<< loss[i] <<"\n";
    }
    loss_f.close();
  }
  
  
}

void visualize_depth_pixel(const std::map<int, std::shared_ptr<Frame>> & frames,
                           std::unordered_map<Pixel *, float> & err_before,
                           std::unordered_map<Pixel *, float> & err_after,
                           const Mat33d_row * K,
                           const Mat33d_row * inv_K                           
                           ) {
  std::map<int, cv::Mat> plots;
  for (auto && [ind, frame]: frames ) {
    plots[ind] = (frame->img()->get_raw()).clone();
  }
  for (auto && [pix, err1] : err_before) {
    float err2 = err_after[pix];
    cv::Scalar color(0,0,0);
    if (err1 < err2) {
      color(2) = 255;
      visualize_pixel(pix,
                      &frames, K, inv_K, "err matches",
                      50, 0.1,
                      std::to_string(err1) + "  vs. "+std::to_string(err2));


    } else 
      color(1) = 255;
    cv::circle( plots[pix->host()->index],
                cv::Point(pix->u(), pix->v()),
                1.0, //curr.rows / 32,
                color,
                cv::FILLED,
                cv::LINE_8 );
  }
  for (auto && [ind, img] : plots) {
    cv::imshow(std::to_string(ind), img);
    cv::waitKey(0);
  }
}

void  evaluate_depth(const std::map<int, std::shared_ptr<Frame>> & frames,
                     std::map<int, std::vector<float>> & gt_depth,
                     std::map<int, std::vector<bool>> & inlier_filter,
                     std::unordered_map<Pixel *, float> & err_all,                      std::string & fname) {

  if (gt_depth.size() == 0)
    return;

  if (inlier_filter.size() == 0)
    return;

  for (auto && [ind, frame] : frames) {
    std::vector<float> err;
    //err_all.insert(std::make_pair(ind, std::vector<float>()));
    std::ofstream err_f(std::to_string(ind)+fname);
    auto & gt_depth_curr = gt_depth[ind];
    for (int i = 0; i < frame->size(); i++) {
      if (inlier_filter[ind][i] == false)
        continue;
      float pred_depth =static_cast<float> (1 / frame->pixel(i).inv_depth());
      int c = static_cast<int>(frame->pixel(i).u()+0.5);
      int r = static_cast<int>(frame->pixel(i).v()+0.5);
      float gt = gt_depth_curr[r * frame->img()->cols + c];
      err.push_back(abs((gt-pred_depth)));
      err_f << err.back()<<"\n";
      err_all.insert(std::make_pair(&frame->pixel(i),err.back()));
      //std::cout<<"pred: "<<pred_depth<<", gt: "<<gt<<"\n";
    }
    err_f.close();
    float sum = std::accumulate(std::begin(err), std::end(err), 0.0);
    float m =  sum / err.size();
    float accum = 0.0;
    std::for_each (std::begin(err), std::end(err), [&](const float d) {
      accum += (d - m) * (d - m);
    });
    float stdev = sqrt(accum / (err.size()-1));

    std::sort(err.begin(), err.end());
    std::cout<<"Eval frame "<<ind<<" err: mean="<<m<<", std="<<stdev<<", medium="<<err[err.size()/2]<<"\n";
  }
  
}


int mapping(std::map<int, std::shared_ptr<Frame>> frames,
            //std::vector<std::pair<int, int>> & edges,
            const Mat33d_row * K,
            double weight,
            std::map<int, std::vector<bool>> &inlier_filter,
            std::map<int, std::vector<float>> & gt_depth,            
            bool is_vis=false
            ) {

  const Mat33d_row inv_K = (*K).inverse();

  ceres::Problem problem;
  
  LocalParameterizationSE3 * se3_parameterization =
    new LocalParameterizationSE3();
  for (auto && [ind2, frame2] : frames) {
    problem.AddParameterBlock(frame2->pose_vec(), 12, se3_parameterization);
    problem.SetParameterBlockConstant(frame2->pose_vec());

    std::cout<<"Add frame "<<ind2<<" block, image ptr is "<<frame2->img()<<"\n";
    
    for (int k = 0; k < frame2->size(); k++) {
      problem.AddParameterBlock(&(frame2->pixel(k).inv_depth()), 1);
    }
  }

  Pixel * target = nullptr;
  for (auto && [ind2, frame2] : frames) {
    inlier_filter.insert(std::make_pair(ind2, std::vector<bool>(frame2->size())));
    for (int i = 0; i < frame2->size(); i++) {
      /*
      ceres::Problem problem;

      LocalParameterizationSE3 * se3_parameterization =
        new LocalParameterizationSE3();
      for (auto && [ind1, frame1] : frames) {
        problem.AddParameterBlock(frame1->pose_vec(), 12, se3_parameterization);
        problem.SetParameterBlockConstant(frame1->pose_vec());
        std::cout<<"Add frame "<<ind1<<" block, image ptr is "<<frame1->img()<<"\n";
      }
      problem.AddParameterBlock(&(frame2->pixel(i).inv_depth()), 1);      
      */
      bool is_seen_by_all = true;
      for (auto && [ind1, frame1] : frames) {
        if ( ind2 == ind1 ) continue;


        Eigen::Vector3d curr_p_in_1 = projection<double, Eigen::RowMajor>(Eigen::Map<Mat34d_row>(frame1->pose_vec()),
                                                                          Eigen::Map<Mat34d_row>(frame2->pose_vec()),
                                                                          frame2->pixel(i).uv(),
                                                                          frame2->pixel(i).inv_depth(),
                                                                          *K,
                                                                          inv_K
                                                                          );
        //std::cout<<"Create residuals for frame"<<ind2<<"'s pixel "<<frame2->pixel(i).uv().transpose()<<", inv_depth="<<frame2->pixel(i).inv_depth()<<". ";
        //std::cout<<"Projected to frame "<<ind1<<", at uv="<<curr_p_in_1.transpose()<< "\n";
        
        if (frame1->img()->is_in_bound(curr_p_in_1(0), curr_p_in_1(1)) == false) {
          is_seen_by_all = false;
          break;            
        }
      }
      if (is_seen_by_all == false) {
        //std::cout<<"Skipeed point "<<i<<" from frame "<<ind2<<" because ofb init\n";
        inlier_filter[ind2][i] = false;
        continue;
      } else
        inlier_filter[ind2][i] = true;
    }
  }
  
  std::cout<<"Evaluate before ba:\n";
  std::string before_err_fname("before_ba.err.txt");
  std::unordered_map<Pixel *, float>  err_before;  
  evaluate_depth(frames, gt_depth, inlier_filter, err_before, before_err_fname );

  
  for (auto && [ind2, frame2] : frames) {
    //inlier_filter.insert(std::make_pair(ind2, std::vector<bool>(frame2->size())));
    for (int i = 0; i < frame2->size(); i++) {
      for (auto && [ind1, frame1] : frames) {
        if ( ind2 == ind1 ) continue;
        if (inlier_filter[ind2][i] == false)
          continue;
        

        ceres::DynamicAutoDiffCostFunction<InvDepthAutoDiffFunctor<NUM_CHANNELS>, 8>* cost_per_point =
          new ceres::DynamicAutoDiffCostFunction<InvDepthAutoDiffFunctor<NUM_CHANNELS>, 8>(
                                                                             new InvDepthAutoDiffFunctor<NUM_CHANNELS>(&(frame2->pixel(i)),
                                                                                                         weight,
                                                                                                             
                                                                                                         frame1->img(),
                                                                                                         frame2->img(),
                                                                                                         &inv_K, K
                                                                                                         ));
        cost_per_point->SetNumResiduals(1);
        cost_per_point->AddParameterBlock( 12);
        cost_per_point->AddParameterBlock( 12);
        cost_per_point->AddParameterBlock( 1);
        //cost_per_point->AddParameterBlock( 1);
  
        ceres::LossFunctionWrapper* loss_function = new ceres::LossFunctionWrapper(new ceres::HuberLoss(1), ceres::TAKE_OWNERSHIP);
        problem.AddResidualBlock(cost_per_point, loss_function,
                                 frame1->pose_vec(), frame2->pose_vec(),
                                 &(frame2->pixel(i).inv_depth()));

        if (target == nullptr || rand() % 300 == 0)
          target = &(frame2->pixel(i));

        //std::cout<<"Just added\n";

          

        //break;
      }

      /*

      //break;
      ceres::Solver::Options options;
      options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
      options.linear_solver_type = ceres::SPARSE_SCHUR;
      options.preconditioner_type = ceres::JACOBI;
      options.visibility_clustering_type = ceres::CANONICAL_VIEWS;
      options.function_tolerance = 1e-6;
      options.gradient_tolerance = 1e-6;
      options.parameter_tolerance = 1e-6;
      //options.line_search_direction_type = ceres::BFGS;
      options.num_threads = 24;
      options.max_num_iterations = 100;
  
      //if (i == 618) {
      options.update_state_every_iteration = true;
      LoggingCallback callback(&frames, target, K , &inv_K, weight, is_vis);
      options.callbacks.push_back(&callback);

      //}
  
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
      std::cout << summary.FullReport() << std::endl<<", just fin report for pixel "<<target<<"\n";
      if (summary.final_cost <= summary.initial_cost) {
        inlier_filter[ind2][i] = true;
      }
      */
    }

    //break;
  }
  std::vector<double> loss;
  epipolar_tracing(frames, K, &inv_K, target,
                   sqrt(1 / weight), 0.01,
                   loss,  true);



  

  
  ceres::Solver::Options options;
  options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.preconditioner_type = ceres::JACOBI;
  options.visibility_clustering_type = ceres::CANONICAL_VIEWS;
  options.function_tolerance = 1e-6;
  options.gradient_tolerance = 1e-6;
  options.parameter_tolerance = 1e-6;
  //options.line_search_direction_type = ceres::BFGS;
  options.num_threads = 24;
  options.max_num_iterations = 10;
  
  //if (i == 618) {
  options.update_state_every_iteration = true;
  LoggingCallback callback(&frames, target, K , &inv_K, weight, is_vis);
  options.callbacks.push_back(&callback);

    //}
  visualize_pixel(target, &frames,
                  K, &inv_K,
                  "first");
  
  
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << std::endl<<", just fin report for pixel "<<target<<"\n";


  std::cout<<"Evaluate results after ba: \n";
  std::string after_err_fname("after_ba.err.txt");
  std::unordered_map<Pixel *, float>  err_after;
  evaluate_depth(frames, gt_depth,  inlier_filter, err_after, after_err_fname);
  visualize_depth_pixel(frames,
                        err_before,
                        err_after,
                        K,
                        &inv_K
                        );
  

  std::cout<<"Write results:\n";
  std::string after_name("after_depth_ba.pcd");
  write_transformed_pc(frames, &inv_K, after_name, true);//, &inlier_filter);

  //std::cout<<" pose vec is ";
  //for (int k = 0; k < 12; k++) std::cout<<pose_vec[k]<<", "; std::cout<<"\n";

          
  return 0;
}
/*
  int explicit_loop(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> & pcs,
  std::vector<std::pair<int, int>> & edges,
  std::vector<std::vector<double>> & poses,
  std::vector<std::vector<Point>> & pixels,
  std::vector<std::vector<double>> & inv_depths,
  const Mat33d_row * K,
  const Mat33d_row * inv_K,
  const double * lambda ,
  double ell,
  double sp_thresh
  ) {




  std::vector<std::vector<Eigen::Vector3f,
  Eigen::aligned_allocator<Eigen::Vector3f>>> pcs_xyz(pcs.size()); for (int i =
  0 ; i < pcs.size(); i++) { auto & pc = *pcs[i];
  pcs_xyz[i].resize(pcs[i]->size());
  for (int j = 0; j < pc.size(); j++) {
  pcs_xyz[i][j](0) = pc[j].x;
  pcs_xyz[i][j](1) = pc[j].y;
  pcs_xyz[i][j](2) = pc[j].z;
  }
  }



  std::vector<std::unique_ptr<kd_tree_eigen>> mat_indices(edges.size());
  for (int i = 0; i < pcs.size(); i++) {

  mat_indices[i].reset(new kd_tree_eigen(3 , pcs_xyz[i], 20  ));
  mat_indices[i]->index->buildIndex();
  }


  int iter_ = 0;
  int last_nonzeros = 0;
  bool converged = false;
  while (!converged) {
  std::cout<<"\n\n==============================================================\n";
  std::cout << "Iter "<<iter_<< ": Solved Ceres problem, ell is "
  <<ell<<std::endl<<std::flush;

  ceres::Problem problem;

  LocalParameterizationSE3 * se3_parameterization =
  new LocalParameterizationSE3();

  std::cout<<"All poses are \n";
  for (int i = 0; i < pcs.size(); i++) {
  std::cout<<"frame "<<i<<std::endl;
  std::cout<<Eigen::Map<Mat34d_row>(poses[i].data())<<std::endl;
  problem.AddParameterBlock(poses[i].data(), 12, se3_parameterization);
  if (i == 0)
  problem.SetParameterBlockConstant(poses[i].data());

  }
  std::vector<Sophus::SE3d> poses_old(pcs.size());
  //pose_snapshot(poses, poses_old);

  //std::vector<double> invdepth_old(pcs.size());
  //invdepth_snapshot(pixels, invdepth_old);

  //for (int i = 0; i < pcs.size(); i++) {
  //  problem.AddParameterBlock(inv_depths[i].data(), inv_depths[i].size() );
  // }


  int curr_nonzeros = 0;
  for (int e = 0; e < edges.size(); e++) {
  std::pair edge = edges[e];
  int id1 = edge.first;
  int id2 = edge.second;

  Eigen::Map<
  Eigen::Matrix<double, 3, 4, Eigen::RowMajor> > T1_Rt(poses[id1].data());
  Eigen::Map<
  Eigen::Matrix<double, 3, 4, Eigen::RowMajor> > T2_Rt(poses[id2].data());

  Mat44d_row T1 = Mat44d_row::Identity();
  T1.block<3,4>(0,0) = T1_Rt;
  Mat44d_row T2 = Mat44d_row::Identity();
  T2.block<3,4>(0,0) = T2_Rt;
  Mat44d_row T = T1.inverse() * T2;
  Mat33d last_R_eigen = T.block<3,3>(0,0);
  Vec3d last_t_eigen = T.block<3,1>(0,3);
  //for (int k = 0; k < 12; k++) std::cout<<pose_vec[k]<<", "; std::cout<<"\n";
  //Sophus::SE3d last_T_sophus(last_R_eigen, last_t_eigen);

  //pcl::PointCloud<pcl::PointXYZ> pc2_curr_(pcs[id2].size());
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
  pc2_curr_(pcs[id2]->size()); #pragma omp parallel for for (int i = 0; i <
  pcs[id2]->size(); i++ ){ Eigen::Vector3f p2 = (*pcs[id2])[i].getVector3fMap();
  //pc2_curr_[i] = last_T_sophus * p2.cast<double>();
  pc2_curr_[i] = last_R_eigen * p2.cast<double> () + last_t_eigen;
  }

  Eigen::SparseMatrix<double, Eigen::RowMajor> ip_mat(pcs[id1]->size(),
  pcs[id2]->size());
  //std::unordered_map<std::unordered_map<Vec12d_row>> & jacob;
  se_kernel(*pcs[id1], pc2_curr_, pixels[id1], pixels[id2],
  mat_indices[id1].get(), ell, sp_thresh, 0.1 * 0.1, ip_mat
  //,jacob
  );
  int nonzeros = ip_mat.nonZeros();
  std::cout<<"Nonzeros is "<<nonzeros<<std::endl;
  if (nonzeros == 0)
  break;
  curr_nonzeros += nonzeros;
  //if (num_zeros > count_nonzeros)
  //      count_nonzeros = num_zeros;


  for (int k=0; k<ip_mat.outerSize(); ++k)
  {
  for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(ip_mat,k);
  it; ++it) {

  int idx1 = it.row();   // row index
  int idx2 = it.col();   // col index (here it is equal to k)
  double weight = it.value();

  ceres::DynamicAutoDiffCostFunction<ClosestPointAutoDiffFunctor, 512>*
  cost_per_point = new
  ceres::DynamicAutoDiffCostFunction<ClosestPointAutoDiffFunctor, 512>( new
  ClosestPointAutoDiffFunctor(&pixels[id1][idx1], &pixels[id2][idx2], weight,
  ell,
  0.1,
  lambda,
  inv_K
  ));
  cost_per_point->SetNumResiduals(1);
  cost_per_point->AddParameterBlock( 12);
  cost_per_point->AddParameterBlock( 12);
  cost_per_point->AddParameterBlock( 1);
  cost_per_point->AddParameterBlock( 1);
  //ceres::LossFunctionWrapper* loss_function(new ceres::HuberLoss(1.0),
  ceres::TAKE_OWNERSHIP); ceres::LossFunctionWrapper* loss_function = new
  ceres::LossFunctionWrapper(new ceres::HuberLoss(1.0), ceres::TAKE_OWNERSHIP);
  problem.AddResidualBlock(cost_per_point, loss_function,
  poses[id1].data(), poses[id2].data(),
  &inv_depths[id1][idx1], &inv_depths[id2][idx2]);
  //problem.SetParameterBlockConstant(  &inv_depths[id1][idx1]);
  //problem.SetParameterBlockConstant(  &inv_depths[id2][idx2]);
  }
  }
  }
  //problem.SetParameterBlockConstant(poses[0].data());
  ceres::Solver::Options options;
  options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.preconditioner_type = ceres::JACOBI;
  options.visibility_clustering_type = ceres::CANONICAL_VIEWS;
  options.function_tolerance = 1e-8;
  options.gradient_tolerance = 1e-8;
  options.parameter_tolerance = 1e-8;
  //options.line_search_direction_type = ceres::BFGS;
  options.num_threads = 24;

  options.max_num_iterations = 200;

  ceres::Solver::Summary summary;

  ceres::Solve(options, &problem, &summary);


  std::cout << summary.FullReport() << std::endl;
  //std::cout<<" pose vec is ";
  //for (int k = 0; k < 12; k++) std::cout<<pose_vec[k]<<", "; std::cout<<"\n";


  std::vector<Sophus::SE3d> poses_new(pcs.size());
  //pose_snapshot(poses, poses_new);
  //double param_change = change_of_all_poses(poses_old, poses_new);
  //std::cout<<"Update of pose is "<<param_change<<std::endl;


  //if (curr_nonzeros <= last_nonzeros) {
  //if (param_change < 1e-5 ) {
  if (iter_ && iter_ % 10 == 0) {
  if (ell >= 0.05) {
  ell = ell *  0.7;
  std::cout<<"Reduce ell to "<<ell<<std::endl;
  last_nonzeros = 0;
  } else {
  converged = true;
  // std::cout<<"End: pose change is "<<param_change<<std::endl;
  }
  } else
  last_nonzeros = curr_nonzeros;
  iter_++;
  curr_nonzeros=0;
  }



  return 0;
  }




  /*void write_traj_file(std::string & fname,
  PoseFormat tum_or_kitti,
  std::vector<std::string> & timestamps,
  std::vector<std::vector<double>> & poses ) {
  std::ofstream outfile(fname);
  for (int i = 0; i< poses.size(); i++) {
  if (tum_or_kitti == PoseFormat::TUM) {
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  pose.block<3,4>(0,0) = Eigen::Map<cvo::Mat34d_row>(poses[i].data());
  Sophus::SO3d q(pose.block<3,3>(0,0));
  auto q_eigen = q.unit_quaternion().coeffs();
  Eigen::Vector3d t(pose.block<3,1>(0,3));
  outfile <<timestamps[i]<<" "<< t(0) <<" "<< t(1)<<" "<<t(2)<<" "
  <<q_eigen[0]<<" "<<q_eigen[1]<<" "<<q_eigen[2]<<" "<<q_eigen[3]<<std::endl;
  } else {
  for (int j = 0 ; j < 12; j++)
  outfile << poses[i][j]<<" ";
  outfile << "\n";
  }
  }
  outfile.close();
  }
*/


int main(int argc, char **argv) {

  srand(time(nullptr));


  std::string dataset_folder(argv[1]); 
  int tum_or_kitti_arg = std::stoi(std::string(argv[2])); 
  std::string graph_file_name(argv[3]);
  std::string gt_pose_fname(argv[4]);
  double weight = std::stod(std::string(argv[5]));
  bool is_vis = (std::stoi(std::string(argv[6])) > 0);
  //std::string tracking_subset_poses_fname(argv[3]);
  //double ell = std::stod(std::string(argv[4]));
  //double sp_thresh = std::stod(std::string(argv[5]));
  //double lambda = std::stod(std::string(argv[6]));

  
  std::unique_ptr<cvo::DatasetHandler> dataset;
  std::unique_ptr<cvo::Calibration> calib;
  PoseFormat pose_format;
  if (tum_or_kitti_arg == (int)PoseFormat::KITTI) {
    pose_format = PoseFormat::KITTI;
    dataset.reset(new cvo::KittiHandler(dataset_folder, cvo::KittiHandler::DataType::STEREO));
    //dynamic_cast<cvo::TartanAirHandler*>(dataset.get())->set_depth_folder_name("deep_depth");
    std::string calib_file = dataset_folder + "/cvo_calib.txt";
    calib.reset(new cvo::Calibration(calib_file, cvo::Calibration::STEREO));
    
  } else if (tum_or_kitti_arg == (int)PoseFormat::TARTAN) {
    pose_format = PoseFormat::TARTAN;
    dataset.reset(new cvo::TartanAirHandler(dataset_folder));
    dynamic_cast<cvo::TartanAirHandler*>(dataset.get())->set_depth_folder_name("deep_depth");
    std::string calib_file = dataset_folder + "/cvo_calib_deep_depth.txt";
    calib.reset(new cvo::Calibration(calib_file, cvo::Calibration::RGBD));
  } else {
    pose_format = PoseFormat::TUM;
    dataset.reset(new cvo::TumHandler(dataset_folder));
    //dynamic_cast<cvo::TumHandler*>(dataset.get())->set_depth_folder_name("deep_depth");
    std::string calib_file = dataset_folder + "/cvo_calib.txt";
    calib.reset(new cvo::Calibration(calib_file, cvo::Calibration::RGBD));
  }
  std::cout<<"create dataset\n";
  //  PoseFormat tum_or_kitti = static_cast<PoseFormat>(std::stoi(std::string(argv[7])));

  Mat33d_row K, inv_K;
  K = calib->intrinsic().cast<double>();
  inv_K = K.inverse().eval();

  std::cout<<"K: \n"<<K<<"\n";
  
  
  //std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pcs;
  //std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pcs;
  std::map<int, std::shared_ptr<Frame>> frames;
  std::vector<int> frame_inds;  
  std::vector<std::pair<int, int>> edge_original_inds;
  std::vector<cvo::Mat34d_row, Eigen::aligned_allocator<cvo::Mat34d_row>> gt_poses_all, frame_poses;  
  cvo::read_graph_file<double, 3, 4, Eigen::RowMajor>(graph_file_name, frame_inds, edge_original_inds, frame_poses);
  std::cout<<"read graph file "<<graph_file_name<<"\n";
  //if (pose_format == KITTI) {
  //  
  //} else if (pose_format == TARTAN) {
  //  cvo::read_pose_file_tartan_format(gt_pose_fname,
  //                                    0,
  //                                    frame_inds.back(),
  //                                    gt_poses_all);
  //}

  // read point cloud
  //std::unordered_map<int, int> id_to_index;
  //std::vector<std::pair<int, int>> edges;
  //cvo::CvoPointCloud pc_all(FEATURE_DIMENSIONS, NUM_CLASSES);
  std::map<int, std::vector<float>> gt_depth;
  for (int i = 0; i<frame_inds.size(); i++) {
    //for (auto & [curr_frame_id, frame] : frames) {
    int curr_frame_id = frame_inds[i];
    
    //float leaf_size = cvo_align.get_params().multiframe_downsample_voxel_size;  
    //int curr_frame_id = frame_inds[i];
    dataset->set_start_index(curr_frame_id);
    std::shared_ptr<cvo::CvoPointCloud> pc_edge_raw;
    cv::Mat rgb;    
    if (pose_format == PoseFormat:: TARTAN) {
      std::vector<float> depth;
      dataset->read_next_rgbd(rgb, depth);
      std::shared_ptr<cvo::ImageRGBD<float>> raw(new cvo::ImageRGBD<float>(rgb, depth));
      pc_edge_raw.reset(new cvo::CvoPointCloud(*raw, *calib, cvo::CvoPointCloud::DSO_EDGES));

      dynamic_cast<cvo::TartanAirHandler*>(dataset.get())->set_depth_folder_name("depth_left");      
      std::vector<float> gt_depth_curr;
      gt_depth.insert(std::make_pair(curr_frame_id, gt_depth_curr));
      dataset->read_next_rgbd(rgb, gt_depth[curr_frame_id]);
      dynamic_cast<cvo::TartanAirHandler*>(dataset.get())->set_depth_folder_name("deep_depth");      
      

      
    } else if (pose_format == PoseFormat:: KITTI) {
      cv::Mat  right;
      dataset->read_next_stereo(rgb, right);
      std::shared_ptr<cvo::ImageStereo> raw(new cvo::ImageStereo(rgb, right));
      pc_edge_raw.reset(new cvo::CvoPointCloud(*raw, *calib, cvo::CvoPointCloud::DSO_EDGES));
    } else if (pose_format == PoseFormat::TUM) {
      std::vector<uint16_t> depth;
      dataset->read_next_rgbd(rgb, depth);
      std::shared_ptr<cvo::ImageRGBD<uint16_t>> raw(new cvo::ImageRGBD<uint16_t>(rgb, depth));
      pc_edge_raw.reset(new cvo::CvoPointCloud(*raw, *calib, cvo::CvoPointCloud::DSO_EDGES));
      
    }

    pc_edge_raw->write_to_color_pcd(std::to_string(i)+".pcd");

    std::shared_ptr<Frame> frame(new Frame(curr_frame_id, rgb));
    frame->set_pose(frame_poses[i].data());
    frames.insert(std::make_pair(frame_inds[i], frame));
    
    

    //  std::shared_ptr<cvo::CvoPointCloud> pc_edge_raw(
    for (int j = 0; j < pc_edge_raw->size(); j++) {
      const CvoPoint & pt = pc_edge_raw->point_at(j);
      Eigen::Vector3f xyz = K.cast<float>() * pt.getVector3fMap();
      double inv_depth = 1 / xyz(2);
      xyz = ( xyz / xyz(2)).eval();
      //std::vector<double> feat_double;
      //for (int k = 0; k < 3; k++) feat_double[k] = static_cast<double>(pt.features[k]);
      if (frame->img()->is_in_bound(xyz(0), xyz(1)))
        frame->push_back(xyz(0), xyz(1), inv_depth);
    }

    /*
      Eigen::Matrix4f pose_w = Eigen::Matrix4f::Identity();
      pose_w.block<3,4>(0,0) = frame_poses[i].cast<float>();
      cvo::CvoPointCloud transformed;
      cvo::CvoPointCloud::transform(pose_w, *pc_edge_raw, transformed);
      pc_all+= transformed;
    */
    
    //std::cout<<"is_edge_only is "<<is_edge_only<<"\n";
    std::cout<<"Load "<<curr_frame_id<<", "<<frame->size()<<" number of points\n";
    //pcs.push_back(pc);
    
    //id_to_index[curr_frame_id] = i;
  }

  omp_set_num_threads(24);

  //std::string before_pc("before_depth_pc.pcd");
  //pc_all.write_to_color_pcd(before_pc);
  
  std::string before_name("before_depth_ba.pcd");
  write_transformed_pc(frames, &inv_K, before_name);


  //// main loop
  //explicit_loop(pcs, edges, poses, pixels, inv_depths, &K, &inv_K, & lambda, ell, sp_thresh);
  std::map<int, std::vector<bool>> inlier_filter;  
  mapping(frames,  &K, weight, inlier_filter, gt_depth, is_vis);


  std::cout<<"Fin\n";

  //std::string traj_out_name("invdepth_traj.txt");
  //write_traj_file(traj_out_name, tum_or_kitti, timestamps, poses);
  return 0;
  
  
}

