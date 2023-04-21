#include "cvo/CvoFrame.hpp"
#include "utils/CvoPointCloud.hpp"
#include <cstdlib>

namespace cvo {

  CvoFrame::CvoFrame(const CvoPointCloud * pts,
                     const double poses[12],
                     bool is_using_kdtree) : points(pts)

                                               // points_transformed_(pts->num_features(), pts->num_classes()){
  {
    memcpy(pose_vec, poses, 12*sizeof(double));
  
  }

  const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > & CvoFrame::points_transformed() {
    return points_transformed_;
  }

  void CvoFrame::transform_pointcloud() {
    
  }

  Eigen::Matrix4d CvoFrame::pose_cpu() const {
    Eigen::Matrix4d pose;
    pose << pose_vec[0], pose_vec[1], pose_vec[2], pose_vec[3],
      pose_vec[4],pose_vec[5], pose_vec[6],pose_vec[7],
      pose_vec[8],pose_vec[9],pose_vec[10],pose_vec[11],
      0,0,0,1;
    return pose;
  }
      
      



}
