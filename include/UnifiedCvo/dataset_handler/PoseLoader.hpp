#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <Eigen/Dense>
#include <map>
#include <sophus/so3.hpp>
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

  void transform_vector_of_poses(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> & poses_in,
                                 //const Sophus::SE3d & pose_anchor,
                                 const Eigen::Matrix4d & pose_anchor_eigen,
                                 std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> & poses_out) {
    poses_out.resize(poses_in.size());
    
    for (int i = 1; i < poses_in.size(); i++) {
      Eigen::Matrix4d i_from_0 = poses_in[0].inverse() * poses_in[i];
      poses_out[i] = pose_anchor_eigen * i_from_0;
    }
    if (poses_out.size() > 0) {
      poses_out[0] = pose_anchor_eigen;
    }
    
  }


  //// read g2o format, copied from ceres example
  // Reads a single pose from the input and inserts it into the map. Returns false
  // if there is a duplicate entry.
  template <typename Pose, typename Allocator>
  bool ReadVertex(std::ifstream* infile,
                  std::map<int, Pose, std::less<int>, Allocator>* poses) {
    int id;
    Pose pose;
    *infile >> id >> pose;
    // Ensure we don't have duplicate poses.
    if (poses->find(id) != poses->end()) {
      std::cerr << "Duplicate vertex with ID: " << id <<"\n";
      return false;
    }
    (*poses)[id] = pose;
    return true;
  }
  // Reads the constraints between two vertices in the pose graph
  template <typename Constraint, typename Allocator>
  void ReadConstraint(std::ifstream* infile,
                      std::vector<Constraint, Allocator>* constraints) {
    Constraint constraint;
    *infile >> constraint;
    constraints->push_back(constraint);
  }
  // Reads a file in the g2o filename format that describes a pose graph
  // problem. The g2o format consists of two entries, vertices and constraints.
  //
  // In 2D, a vertex is defined as follows:
  //
  // VERTEX_SE2 ID x_meters y_meters yaw_radians
  //
  // A constraint is defined as follows:
  //
  // EDGE_SE2 ID_A ID_B A_x_B A_y_B A_yaw_B I_11 I_12 I_13 I_22 I_23 I_33
  //
  // where I_ij is the (i, j)-th entry of the information matrix for the
  // measurement.
  //
  //
  // In 3D, a vertex is defined as follows:
  //
  // VERTEX_SE3:QUAT ID x y z q_x q_y q_z q_w
  //
  // where the quaternion is in Hamilton form.
  // A constraint is defined as follows:
  //
  // EDGE_SE3:QUAT ID_a ID_b x_ab y_ab z_ab q_x_ab q_y_ab q_z_ab q_w_ab I_11 I_12 I_13 ... I_16 I_22 I_23 ... I_26 ... I_66 // NOLINT
  //
  // where I_ij is the (i, j)-th entry of the information matrix for the
  // measurement. Only the upper-triangular part is stored. The measurement order
  // is the delta position followed by the delta orientation.
  template <typename Pose,
            typename Constraint,
            typename MapAllocator,
            typename VectorAllocator>
  bool ReadG2oFile(const std::string& filename,
                   std::map<int, Pose, std::less<int>, MapAllocator>* poses,
                   std::vector<Constraint, VectorAllocator>* constraints) {
    CHECK(poses != nullptr);
    CHECK(constraints != nullptr);
    poses->clear();
    constraints->clear();
    std::ifstream infile(filename.c_str());
    if (!infile) {
      return false;
    }
    std::string data_type;
    while (infile.good()) {
      // Read whether the type is a node or a constraint.
      infile >> data_type;
      if (data_type == Pose::name()) {
        if (!ReadVertex(&infile, poses)) {
          return false;
        }
      } else if (data_type == Constraint::name()) {
        ReadConstraint(&infile, constraints);
      } else {
        std::cerr << "Unknown data type: " << data_type <<"\n";
        return false;
      }
      // Clear any trailing whitespace from the line.
      infile >> std::ws;
    }
    return true;
  }


  template <typename T, unsigned int major>
  void write_traj_file(std::string & fname,
                       std::vector<Eigen::Matrix<T, 4, 4, major>,
                       Eigen::aligned_allocator<Eigen::Matrix<T, 4, 4, major>>> &  poses) {

    std::ofstream outfile(fname);
    for (int i = 0; i< poses.size(); i++) {
      Eigen::Matrix<T, 4, 4, major> pose = poses[i];//Eigen::Matrix4f::Identity();
      Sophus::SO3<T> q(pose.block(0,0,3,3));
      auto q_eigen = q.unit_quaternion().coeffs();
      Eigen::Matrix<T, 3, 1, major> t(pose.block(0,3,3,1));
      outfile << t(0) <<" "<< t(1)<<" "<<t(2)<<" "
              <<q_eigen[0]<<" "<<q_eigen[1]<<" "<<q_eigen[2]<<" "<<q_eigen[3]<<std::endl;
    
    }
    outfile.close();
  }

  
  
}
