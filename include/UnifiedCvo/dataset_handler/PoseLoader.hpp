#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <Eigen/Dense>

namespace cvo {
  
  // read poses in tum format: [time x y z qx qy qz qw]
  inline
  void read_pose_file_tum_format(const std::string & pose_fname,
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

      line_stream >> timestamp;
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

      poses.push_back(T);
      line_ind ++;
    }
    f.close();
  }

  
  // read poses in tum format: [x y z qx qy qz qw]
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

      poses.push_back(T);
      line_ind ++;
    }
    f.close();
    std::cout<<"read "<<poses.size()<<" poses\n";
  }

  
  // read poses in kitti format
  inline
  void read_pose_file_kitti_format(const std::string & pose_fname,
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
      std::string substr;
      double pose_v[12];
      int pose_counter = 0;
      while (std::getline(line_stream,substr, ' ')) {
        pose_v[pose_counter] = std::stod(substr);
        pose_counter++;
      }
      Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> Tr(pose_v);
      Eigen::Matrix<double, 3, 4, Eigen::ColMajor> T34 = Tr;
      Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
      T.block<3,4>(0,0) = T34;
      poses.push_back(T);
      std::cout<<"Read pose from file\n"<<T<<std::endl;
      line_ind++;

    }
    f.close();
  }

  
}
