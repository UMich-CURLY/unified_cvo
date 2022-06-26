#include <iostream>
#include <list>
#include <cmath>
#include <pcl/filters/voxel_grid.h>
#include <vector>
#include <utility>
#include <random>
#include <string>
#include <fstream>
#include <sstream>
#include <set>
#include <Eigen/Dense>
#include "cvo/CvoGPU.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoFrame.hpp"
#include "cvo/CvoFrameGPU.hpp"
#include "cvo/IRLS_State.hpp"
#include "cvo/IRLS_State_GPU.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


void write_transformed_pc(std::vector<cvo::CvoFrame::Ptr> & frames, std::string & fname) {
  pcl::PointCloud<pcl::PointXYZ> pc_all;
  for (auto ptr : frames) {
    cvo::CvoPointCloud new_pc(0,0);
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3,4>(0,0) = Eigen::Map<cvo::Mat34d_row>(ptr->pose_vec);
    
    Eigen::Matrix4f pose_f = pose.cast<float>();
    cvo::CvoPointCloud::transform(pose_f, *ptr->points, new_pc);

    pcl::PointCloud<pcl::PointXYZ> pc_curr;
    new_pc.export_to_pcd(pc_curr);

    pc_all += pc_curr;
  }
  pcl::io::savePCDFileASCII(fname, pc_all);
}

void gen_random_poses(std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> & poses, int num_poses) {
  poses.resize(num_poses);
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<> dist(0, 1);
  for (int i = 0; i < num_poses; i++) {

    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(dist(e2)*M_PI, Eigen::Vector3f::UnitX())
      * Eigen::AngleAxisf(dist(e2)*M_PI,  Eigen::Vector3f::UnitY())
      * Eigen::AngleAxisf(dist(e2)*M_PI, Eigen::Vector3f::UnitZ());

    poses[i] = Eigen::Matrix4f::Identity();
    poses[i].block<3,3>(0,0) = rot;
    poses[i](0,3) = dist(e2);
    poses[i](1,3) = dist(e2);
    poses[i](2,3) = dist(e2);

    std::cout<<"random pose "<<i<<" is \n"<<poses[i]<<"\n";
  }
}

int main(int argc, char** argv) {

  omp_set_num_threads(24);
  std::string in_pcd_fname(argv[1]);
  std::string cvo_param_file(argv[2]);
  int num_frames = std::stoi(argv[3]);
  
  cvo::CvoGPU cvo_align(cvo_param_file);

  pcl::PointCloud<pcl::PointXYZ>::Ptr raw_pcd(new pcl::PointCloud<pcl::PointXYZ>);

  pcl::io::loadPCDFile<pcl::PointXYZ> (in_pcd_fname, *raw_pcd);
  
  // Create the filtering object
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud (raw_pcd);
  sor.setLeafSize (0.025f, 0.025f, 0.025f);
  sor.filter (*raw_pcd);
  std::cout<<"after downsampling, num of points is "<<raw_pcd->size()<<std::endl;

  std::shared_ptr<cvo::CvoPointCloud> raw (new cvo::CvoPointCloud(*raw_pcd));  

  //std::vector<cvo::Mat34d_row, Eigen::aligned_allocator<cvo::Mat34d_row>> tracking_poses;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> tracking_poses;
  gen_random_poses(tracking_poses, num_frames);

  // read point cloud
  std::vector<cvo::CvoFrame::Ptr> frames;
  std::vector<std::shared_ptr<cvo::CvoPointCloud>> pcs;
  for (int i = 0; i<num_frames; i++) {

    std::shared_ptr<cvo::CvoPointCloud> pc(new cvo::CvoPointCloud(0,0));
    
    cvo::CvoPointCloud::transform(tracking_poses[i],
                                  *raw,
                                  *pc);
    cvo::Mat34d_row pose;
    Eigen::Matrix<double, 4,4, Eigen::RowMajor> pose44 = Eigen::Matrix<double, 4,4, Eigen::RowMajor>::Identity();
    pose = pose44.block<3,4>(0,0);
    

    cvo::CvoFrame::Ptr new_frame(new cvo::CvoFrameGPU(pc.get(), pose.data()));
    frames.push_back(new_frame);
    pcs.push_back(pc);
  }
  std::string f_name("before_BA.pcd");
  write_transformed_pc(frames, f_name);

  // std::list<std::pair<std::shared_ptr<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr>> edges;
  std::list<cvo::BinaryState::Ptr> edge_states;            
  for (int i = 0; i < num_frames; i++) {
    for (int j = i+1; j < num_frames; j++ ) {
      //std::pair<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr> p(frames[i], frames[j]);
      //edges.push_back(p);
      const cvo::CvoParams & params = cvo_align.get_params();
      cvo::BinaryStateGPU::Ptr edge_state(new cvo::BinaryStateGPU(std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[i]),
                                                                  std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[j]),
                                                                  &params,
                                                                  cvo_align.get_params_gpu(),
                                                                  params.multiframe_num_neighbors,
                                                                  params.multiframe_ell_init
                                                             //dist / 3
                                                             ));
      edge_states.push_back((edge_state));
      
    }
  }

  std::vector<bool> const_flags(frames.size(), false);
  const_flags[0] = true;
  
  cvo_align.align(frames, const_flags,
                  edge_states,  nullptr);

  //std::cout<<"Align ends. Total time is "<<time<<std::endl;
  f_name="after_BA.pcd";
  write_transformed_pc(frames, f_name);


  return 0;
}
