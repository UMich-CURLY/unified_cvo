#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <pcl/common/transforms.h>
#include <boost/filesystem.hpp>
//#include <opencv2/opencv.hpp>
#include "dataset_handler/TartanAirHandler.hpp"
//#include "graph_optimizer/Frame.hpp"
#include "utils/Calibration.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoGPU.hpp"
#include "cvo/Cvo.hpp"
#include "cvo/CvoParams.hpp"
#include "utils/ImageRGBD.hpp"
using namespace std;
using namespace boost::filesystem;
// read poses in tum format: [x y z qx qy qz qw]
Eigen::Quaterniond
euler2Quaternion( const double roll,
                  const double pitch,
                  const double yaw )
{
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

    Eigen::Quaterniond q = rollAngle * yawAngle;
    return q;
}
inline
void read_pose_file_tartan_format(const std::string & pose_fname,
                                  int start_frame,
                                  int last_frame,
                                  std::vector<Eigen::Matrix4d,
                                  Eigen::aligned_allocator<Eigen::Matrix4d>> & poses) {
  std::ifstream f(pose_fname);
  poses.clear();
  std::string line;
  int line_ind = 0;
  while (std::getline(f, line)) {
    
    if (line_ind < start_frame) {
      line_ind ++;
      continue;
    }
    if (line_ind == last_frame+1)
      break;
    
    std::stringstream line_stream(line);
    std::string timestamp;
    double xyz[3];
    double q[4]; // x y z w
    int pose_counter = 0;

    //line_stream >> timestamp;
    std::string xyz_str[3];
    line_stream >> xyz_str[0] >> xyz_str[1] >> xyz_str[2];
    xyz[0] = std::stod(xyz_str[0]);
    xyz[1] = std::stod(xyz_str[1]);
    xyz[2] = std::stod(xyz_str[2]);
    Eigen::Vector3d xyz_eigen = Eigen::Map<Eigen::Vector3d>(xyz);
    std::string q_str[4];
    line_stream >> q_str[0] >> q_str[1] >> q_str[2] >> q_str[3];
    q[0] = stod(q_str[0]);
    q[1] = stod(q_str[1]);
    q[2] = stod(q_str[2]);
    q[3] = stod(q_str[3]);
    Eigen::Quaterniond q_eigen(q[3], q[0], q[1], q[2]);
    Eigen::Matrix3d R_mat = q_eigen.normalized().toRotationMatrix();
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = R_mat;
    T.block<3,1>(0,3) = xyz_eigen;

    auto coordinateTransform = euler2Quaternion(-M_PI/2,0,-M_PI/2);
    Eigen::Matrix3d cR = coordinateTransform.normalized().toRotationMatrix();
    Eigen::Matrix4d cT = Eigen::Matrix4d::Identity();
    Eigen::Vector3d xyz_c(0,0,0);
    cT.block<3,3>(0,0) = cR;
    cT.block<3,1>(0,3) = xyz_c;

    poses.push_back(T*cT.inverse());
    line_ind ++;
  }
  f.close();
  std::cout<<"read "<<poses.size()<<" poses\n";
}

int main(int argc, char *argv[]) {
  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  cvo::TartanAirHandler tartan(argv[1]);
  int total_iters = tartan.get_total_number();
  //vector<string> vstrRGBName = tum.get_rgb_name_list();
  string cvo_param_file(argv[2]);
  string calib_file;
  calib_file = string(argv[1] ) +"/cvo_calib.txt"; 
  cvo::Calibration calib(calib_file, cvo::Calibration::RGBD);
  int start_frame = std::stoi(argv[3]);
  tartan.set_start_index(start_frame);
  string saveFolder(argv[4]);
  int first_frame = std::stoi(argv[5]);
  
  cvo::CvoGPU cvo_align(cvo_param_file );
  cvo::CvoParams & init_param = cvo_align.get_params();
  float ell_init = init_param.ell_init;
  float ell_decay_rate = init_param.ell_decay_rate;
  int ell_decay_start = init_param.ell_decay_start;
  init_param.ell_init = init_param.ell_init_first_frame;
  init_param.ell_decay_rate = init_param.ell_decay_rate_first_frame;
  init_param.ell_decay_start  = init_param.ell_decay_start_first_frame;
  cvo_align.write_params(&init_param);

  std::cout<<"write ell! ell init is "<<cvo_align.get_params().ell_init<<std::endl;

  cv::Mat source_rgb;
  vector<float> source_depth;
  int NUM_CLASS = 19;
  vector<float> source_semantics;
  tartan.read_next_rgbd(source_rgb, source_depth,NUM_CLASS,source_semantics);

 
  std::shared_ptr<cvo::ImageRGBD<float>> source_raw(new cvo::ImageRGBD<float>(source_rgb, source_depth,NUM_CLASS,source_semantics));
  std::shared_ptr<cvo::CvoPointCloud> source(new cvo::CvoPointCloud(*source_raw,
                                                                    calib
								    ,cvo::CvoPointCloud::DSO_EDGES
                                                                    )); 
  pcl::PointCloud<cvo::CvoPoint> target_cvo;
  cvo::CvoPointCloud_to_pcl(*source,target_cvo);
  std::cout<<"First point is "<<source->at(0).transpose()<<std::endl;
  std::string poseFile = std::string(argv[1]) + "/pose_left.txt";
  std::vector<Eigen::Matrix4d,Eigen::aligned_allocator<Eigen::Matrix4d>> poses;
  read_pose_file_tartan_format(poseFile,0,start_frame,poses);
  Eigen::Matrix4f pose_start = poses[first_frame].cast <float>();
  Eigen::Matrix4f pose_now = poses[start_frame].cast <float>();
  Eigen::Quaternionf quat(pose_now.topLeftCorner<3, 3>());
  std::cout << "current frame  " <<pose_now << std::endl;
  std::cout << "start frame is " << start_frame << std::endl;
  Eigen::Matrix4f pose_need = pose_start.inverse()*pose_now;
  pcl::PointCloud<cvo::CvoPoint>::Ptr raw_pcd_transformed(new pcl::PointCloud<cvo::CvoPoint>);
  pcl::transformPointCloud (target_cvo, *raw_pcd_transformed,pose_need);
   //19, semantics_source, 
  //                                                                    cvo::CvoPointCloud::CV_FAST));
  //Convert cvo Point cloud to normal pointcloud 
  pcl::PointCloud<pcl::PointXYZ> target_xyz;
  for (auto pt: target_cvo.points){
    pcl::PointXYZ newpt;
    newpt.x = pt.x;
    newpt.y = pt.y;
    newpt.z = pt.z;
    target_xyz.push_back(newpt);
  }
  pcl::io::savePCDFileASCII (saveFolder + "result_cvo" + to_string(start_frame) + ".pcd",*raw_pcd_transformed);

  pcl::io::savePCDFileASCII (saveFolder + "result_cvoxyz" + to_string(start_frame) + ".pcd", target_xyz);
  
  return 0;
}
