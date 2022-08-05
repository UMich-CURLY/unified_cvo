//#include <Eigen/src/Geometry/AngleAxis.h>
#include <Eigen/Geometry>
#include <iostream>
#include <list>
#include <cmath>
//#include <pcl-1.9/pcl/impl/point_types.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/impl/point_types.hpp>
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

void add_normal_noise(pcl::PointCloud<pcl::PointNormal> & input,
                      pcl::PointCloud<pcl::PointXYZ> & output,
                      float sigma) {
  
}

void gen_random_poses(std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> & poses, int num_poses,
                      float max_angle_axis=1.0 // max: [0, 1)
                      ) {
  poses.resize(num_poses);
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<> dist(0, max_angle_axis);
  std::uniform_real_distribution<> dist_trans(0,1);
  for (int i = 0; i < num_poses; i++) {

    Eigen::Matrix3f rot;
    Eigen::Vector3f axis;
    axis << (float)dist_trans(e2), (float)dist_trans(e2), (float)dist_trans(e2);
    axis = axis.normalized().eval();

    rot = Eigen::AngleAxisf(dist(e2)*M_PI, Eigen::Vector3f::UnitX())
       * Eigen::AngleAxisf(dist(e2)*M_PI,  Eigen::Vector3f::UnitY())
      * Eigen::AngleAxisf(dist(e2)*M_PI, Eigen::Vector3f::UnitZ());

    //rot = Eigen::AngleAxisf(dist(e2)*M_PI, axis );

    poses[i] = Eigen::Matrix4f::Identity();
    poses[i].block<3,3>(0,0) = rot;
    poses[i](0,3) = dist_trans(e2);
    poses[i](1,3) = dist_trans(e2);
    poses[i](2,3) = dist_trans(e2);

    std::cout<<"random pose "<<i<<" is \n"<<poses[i]<<"\n";
  }
}
template <typename T>
void pcd_rescale(typename pcl::PointCloud<T>::Ptr pcd ){
  Eigen::MatrixXf pcd_eigen(3, pcd->size());
  for (int j = 0; j < pcd->size(); j++) {
    pcd_eigen.col(j) = pcd->at(j).getVector3fMap();
  }

  float scale = (pcd_eigen.rowwise().maxCoeff() - pcd_eigen.rowwise().minCoeff()).norm();
  std::cout << "scale = " << scale << std::endl;
  pcd_eigen /= scale;

  for (int j = 0; j < pcd->size(); j++)
    pcd->at(j).getVector3fMap() = pcd_eigen.col(j);

}

int main(int argc, char** argv) {

  omp_set_num_threads(24);
  std::string in_pcd_fname(argv[1]);
  std::string cvo_param_file(argv[2]);
  int num_frames = std::stoi(argv[3]);
  float max_angle_per_axis = std::stof(argv[4]);
  
  cvo::CvoGPU cvo_align(cvo_param_file);

  pcl::PointCloud<pcl::PointXYZ>::Ptr raw_pcd(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointNormal>::Ptr raw_pcd_normal(new pcl::PointCloud<pcl::PointNormal>);
  pcl::io::loadPCDFile<pcl::PointNormal> (in_pcd_fname, *raw_pcd_normal);
  copyPointCloud(*raw_pcd_normal, *raw_pcd);
  pcd_rescale<pcl::PointXYZ>(raw_pcd);
  

  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> tracking_poses;
  gen_random_poses(tracking_poses, num_frames, 0.2);

  // read point cloud
  std::vector<cvo::CvoFrame::Ptr> frames;
  std::vector<std::shared_ptr<cvo::CvoPointCloud>> pcs;
  for (int i = 0; i<num_frames; i++) {
    // Create the filtering object
    pcl::PointCloud<pcl::PointXYZ>::Ptr raw_pcd_transformed(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud (*raw_pcd, *raw_pcd_transformed, tracking_poses[i]);
    
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud (raw_pcd_transformed);
    sor.setLeafSize (0.025f, 0.025f, 0.025f);
    sor.filter (*raw_pcd_transformed);
    std::cout<<"after downsampling, num of points is "<<raw_pcd_transformed->size()<<std::endl;

    std::shared_ptr<cvo::CvoPointCloud> pc (new cvo::CvoPointCloud(*raw_pcd_transformed));  
    //std::shared_ptr<cvo::CvoPointCloud> pc(new cvo::CvoPointCloud(0,0));
    
    // cvo::CvoPointCloud::transform(tracking_poses[i],
    //                               *raw,
    //                               *pc);
    
    cvo::Mat34d_row pose;
    Eigen::Matrix<double, 4,4, Eigen::RowMajor> pose44 = Eigen::Matrix<double, 4,4, Eigen::RowMajor>::Identity();
    pose = pose44.block<3,4>(0,0);
    

    cvo::CvoFrame::Ptr new_frame(new cvo::CvoFrameGPU(pc.get(), pose.data()));
    frames.push_back(new_frame);
    pcs.push_back(pc);
  }
  
  

  //std::vector<cvo::Mat34d_row, Eigen::aligned_allocator<cvo::Mat34d_row>> tracking_poses;
  std::cout<<"write to before_BA.pcd\n";
  std::string f_name("before_BA_bunny.pcd");
  write_transformed_pc(frames, f_name);

  // std::list<std::pair<std::shared_ptr<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr>> edges;
  std::cout<<"Start constructing cvo edges\n";
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

  std::cout<<"start align\n";
  cvo_align.align(frames, const_flags,
                  edge_states,  nullptr);

  //std::cout<<"Align ends. Total time is "<<time<<std::endl;
  f_name="after_BA_bunny.pcd";
  write_transformed_pc(frames, f_name);


  return 0;
}
