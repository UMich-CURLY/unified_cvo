
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
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
      local_map_(nullptr) {
      //is_map_centroids_latest_(false),
      //map_centroids_(nullptr){
    pose_in_world_.setIdentity();
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
      local_map_(nullptr) {
      //is_map_centroids_latest_(false),
      //map_centroids_(nullptr){
    pose_in_world_.setIdentity();
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
    auto points_from_nonkeyframe = nonkeyframe.points();
    CvoPointCloud transformed_pc;
    Mat44f tf_curr2input = pose_in_world_.inverse() * nonkeyframe.pose_in_world().matrix();
    CvoPointCloud::transform( tf_curr2input,
                              points_from_nonkeyframe,
                              transformed_pc);
    
    local_map_->insert_pointcloud_csm(transformed_pc,
                                      semantic_bki::point3f(tf_curr2input(0,3),tf_curr2input(1,3),tf_curr2input(2,3)),
                                      -1, 100, -1);
  }
  
  
}
