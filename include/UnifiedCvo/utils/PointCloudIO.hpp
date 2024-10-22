#include "utils/CvoPoint.hpp"
#include "cvo/CvoFrame.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <utility>
#include <string>

namespace cvo {
  
template<typename PointT>
void write_transformed_pc(std::map<int, cvo::CvoFrame::Ptr> & frames,
                          std::string & fname,
                          int start_frame_ind=0, int end_frame_ind=1000000){
  pcl::PointCloud<PointT> pc_all;
  pcl::PointCloud<cvo::CvoPoint> pc_xyz_all;
  for (auto & [i, ptr] : frames) {
  //for (int i = start_frame_ind; i <= std::min((int)frames.size(), end_frame_ind); i++) {
  //auto ptr = frames[i];

    cvo::CvoPointCloud new_pc;
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3,4>(0,0) = Eigen::Map<cvo::Mat34d_row>(ptr->pose_vec);
    
    Eigen::Matrix4f pose_f = pose.cast<float>();
    cvo::CvoPointCloud::transform(pose_f, *ptr->points, new_pc);

    pcl::PointCloud<PointT> pc_curr;
    pcl::PointCloud<cvo::CvoPoint> pc_xyz_curr;
    //new_pc.export_semantics_to_color_pcd(pc_curr);
    new_pc.export_to_pcd<PointT>(pc_curr);
    new_pc.export_to_pcd(pc_xyz_curr);

    pc_all += pc_curr;
    pc_xyz_all += pc_xyz_curr;

  }
  std::string fname_color = fname + ".semantic_color.pcd";
  pcl::io::savePCDFileASCII(fname_color, pc_all);
  pcl::io::savePCDFileASCII(fname, pc_xyz_all);
}

template<typename PointT>
void write_transformed_pc(const std::map<int, std::shared_ptr<cvo::CvoPointCloud>> & frames,
                          const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> poses,

                          std::string & fname,
                          int start_frame_ind=0, int end_frame_ind=1000000){
  pcl::PointCloud<PointT> pc_all;
  pcl::PointCloud<cvo::CvoPoint> pc_xyz_all;
  for (auto & [i, ptr] : frames) {
  //for (int i = start_frame_ind; i <= std::min((int)frames.size(), end_frame_ind); i++) {
  //auto ptr = frames[i];

    cvo::CvoPointCloud new_pc;
    Eigen::Matrix4d pose = poses[i]; // Eigen::Matrix4d::Identity();
    //pose.block<3,4>(0,0) = Eigen::Map<cvo::Mat34d_row>(ptr->pose_vec);
    
    Eigen::Matrix4f pose_f = pose.cast<float>();
    cvo::CvoPointCloud::transform(pose_f, *ptr, new_pc);

    pcl::PointCloud<PointT> pc_curr;
    pcl::PointCloud<cvo::CvoPoint> pc_xyz_curr;
    //new_pc.export_semantics_to_color_pcd(pc_curr);
    new_pc.export_to_pcd<PointT>(pc_curr);
    new_pc.export_to_pcd(pc_xyz_curr);

    pc_all += pc_curr;
    pc_xyz_all += pc_xyz_curr;

  }
  std::string fname_color = fname + ".semantic_color.pcd";
  pcl::io::savePCDFileASCII(fname_color, pc_all);
  pcl::io::savePCDFileASCII(fname, pc_xyz_all);
}
  
  
}
