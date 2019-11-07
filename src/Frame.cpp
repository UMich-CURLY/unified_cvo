
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <cassert>
#include "graph_optimizer/Frame.hpp"
#include "utils/StaticStereo.hpp"
#include "utils/Calibration.hpp"
#include "utils/data_type.hpp"
namespace cvo {

  Frame::Frame(int ind,
               const cv::Mat &left_image,
               const cv::Mat & right_image,
               const Calibration & calib
               )
    : id(ind) ,
      h(left_image.rows),
      w(left_image.cols),
      raw_image_(left_image),
      calib(calib),
      points_(raw_image_, right_image, calib),
      local_map_(nullptr),
      is_keyframe_(false),
      tracking_relative_transform_(ind){
      //is_map_centroids_latest_(false),
      //map_centroids_(nullptr){
    pose_in_graph_.setIdentity();

    Eigen::Affine3f eye = Eigen::Affine3f::Identity();
    tracking_relative_transform_.set_relative_transform(ind, eye, 1.0 );
  }


  Frame::Frame(int ind,
               const cv::Mat & left_image,
               const cv::Mat & right_image,
               int num_classes,
               const std::vector<float> & semantics,
               const Calibration & calib)
    : id(ind) ,
      h(left_image.rows),
      w(left_image.cols),
      calib(calib),
      raw_image_(left_image, num_classes, semantics), 
      points_(raw_image_, right_image, calib),
      is_keyframe_(false),
      tracking_relative_transform_(ind),
      local_map_(nullptr) {
      //is_map_centroids_latest_(false),
      //map_centroids_(nullptr){
    pose_in_graph_.setIdentity();
    Eigen::Affine3f eye = Eigen::Affine3f::Identity();
    tracking_relative_transform_.set_relative_transform(ind, eye,1.0 );
  }
  
  Frame::~Frame() {
  }

  void Frame::construct_map() {
    local_map_.reset(new semantic_bki::SemanticBKIOctoMap(0.1, 1, raw_image_.num_class()));

    semantic_bki::point3f origin;
    local_map_->insert_pointcloud_csm(points_, origin, -1, 100, -1);
    //is_map_centroids_latest = false;
  }

  std::unique_ptr<CvoPointCloud> Frame::export_points_from_map() const {
    if (!local_map_ )
      return nullptr;
    std::unique_ptr<CvoPointCloud> ret(new CvoPointCloud(*local_map_, raw_image_.num_class()));
    std::cout<<"export "<<ret->num_points()<<" points in "<<ret->num_classes()<<" from the map's centroids\n";
    return std::move(ret);
  }

  void Frame::add_points_to_map_from(const Frame & nonkeyframe) {
    if (nonkeyframe.tracking_relative_transform().ref_frame_id() != this->id) {
      printf("[add_points_to_map_from] input frame %d is not tracked relative to the current frame %d.Do nothing\n", nonkeyframe.id, this->id);
      return;
    }
    
    auto points_from_nonkeyframe = nonkeyframe.points();
    CvoPointCloud transformed_pc;
    auto tf_curr2input = nonkeyframe.tracking_relative_transform().ref_frame_to_curr_frame().matrix();
    CvoPointCloud::transform( tf_curr2input,
                              points_from_nonkeyframe,
                              transformed_pc);
    
    local_map_->insert_pointcloud_csm(transformed_pc,
                                      semantic_bki::point3f(tf_curr2input(0,3),tf_curr2input(1,3),tf_curr2input(2,3)),
                                      -1, 100, -1);
  }

  const Eigen::Affine3f Frame::pose_in_graph() const {
    if (is_keyframe_)
      return pose_in_graph_;
    else {
      std::cerr<<"ERR: pose_in_graph() only available for keyframes!\n";
      assert(0);
      
    }
      

    
  }
  
}
