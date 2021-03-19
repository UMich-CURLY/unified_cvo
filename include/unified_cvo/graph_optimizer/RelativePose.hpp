#pragma once
#include <Eigen/Dense>
#include <iostream>

namespace cvo{
  // the relative pose computed when running cvo
  class RelativePose {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    RelativePose(int curr_id):
      curr_frame_id_(curr_id) {
      ref_frame_id_ = curr_id;
      ref_frame_to_curr_frame_.setIdentity();
      cvo_inner_product_ = 0;
    }

    RelativePose(const RelativePose & input) :
      curr_frame_id_(input.curr_frame_id()) {
      ref_frame_id_ = input.ref_frame_id();
      ref_frame_to_curr_frame_ = input.ref_frame_to_curr_frame();
      cvo_inner_product_ = input.cvo_inner_product();
      
    }

    RelativePose& operator= (const RelativePose &input)
    {
    // do the copy
      ref_frame_id_ = input.ref_frame_id();
      ref_frame_to_curr_frame_ = input.ref_frame_to_curr_frame();
      cvo_inner_product_ = input.cvo_inner_product();

      return *this;
    }
    
    RelativePose(int curr_id, int ref_id, const Eigen::Affine3f & ref_to_curr) :
      curr_frame_id_(curr_id), ref_frame_id_(ref_id), ref_frame_to_curr_frame_(ref_to_curr),
      cvo_inner_product_(0){}

    void set_relative_transform(const RelativePose & input){
      ref_frame_id_ = input.ref_frame_id();
      ref_frame_to_curr_frame_ = input.ref_frame_to_curr_frame();
      cvo_inner_product_ = input.cvo_inner_product();
    }

    void set_relative_transform( int ref_id, const Eigen::Affine3f & ref_to_curr, float inner_prod) {
      ref_frame_id_ = ref_id;
      ref_frame_to_curr_frame_ = ref_to_curr;
      cvo_inner_product_ = inner_prod;
    }

    int curr_frame_id() const {return curr_frame_id_;}
    int ref_frame_id() const {return ref_frame_id_;}
    float cvo_inner_product() const {return cvo_inner_product_;}
    const Eigen::Affine3f & ref_frame_to_curr_frame() const {return ref_frame_to_curr_frame_;}
  private:
    
    const int curr_frame_id_;
    int ref_frame_id_;
    Eigen::Affine3f ref_frame_to_curr_frame_;
    float cvo_inner_product_;
  };
}
