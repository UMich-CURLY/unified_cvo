#pragma once
#include <iostream>
#include <string>
#include "utils/data_type.hpp"
#include "graph_optimizer/PoseGraphOptimization.hpp"
#include <Eigen/Dense>


namespace cvo {
  
  // Reads a single pose from the input and inserts it into the map. Returns false
  // if there is a duplicate entry.
  inline
  bool ReadVertex(std::ifstream* infile,
                  cvo::pgo::MapOfPoses* poses) {
    int id;
    cvo::pgo::Pose3d pose;
    *infile >> id >> pose;
    // Ensure we don't have duplicate poses.
    if (poses->find(id) != poses->end()) {
      std::cerr << "Duplicate vertex with ID: " << id;
      return false;
    }
    std::cout<<"Read pose "<<id<<"\n";
    (*poses)[id] = pose;
    return true;
  }

  
  // Reads the contraints between two vertices in the pose graph
  inline
  void ReadConstraint(std::ifstream* infile,
                      cvo::pgo::VectorOfConstraints* constraints) {
    cvo::pgo::Constraint3d constraint;
    *infile >> constraint;
    std::cout<<"Read constrain: "<<constraint<<"\n";
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
  inline bool ReadG2oFile(const std::string& filename,
                          std::vector<std::pair<int, int>> & loop_closures,
                          std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>&  lc_poses
                          ){
  
    cvo::pgo::MapOfPoses  poses;
    cvo::pgo::VectorOfConstraints constraints;

    std::ifstream infile(filename.c_str());
    if (!infile) {
      return false;
    }
    std::string data_type;
    while (infile.good()) {
      // Read whether the type is a node or a constraint.
      infile >> data_type;
      if (data_type == cvo::pgo::Pose3d::name()) {
        if (!ReadVertex(&infile, &poses)) {
          return false;
        }
      } else if (data_type == cvo::pgo::Constraint3d::name()) {
        ReadConstraint(&infile, &constraints);
      } else {
        std::cerr << "Unknown data type: " << data_type;
        return false;
      }
      // Clear any trailing whitespace from the line.
      infile >> std::ws;
    }

  
    std::cout<<__func__<<"Read from file, # of lc constrains is  "<<constraints.size()<<"\n";
    for (auto && constrain: constraints ) {
      int id1 = constrain.id_begin;
      int id2 = constrain.id_end;
      loop_closures.push_back(std::make_pair(id1, id2));
      Eigen::Matrix4f pose = cvo::pgo::pose3d_to_eigen<float, Eigen::ColMajor>(constrain.t_be);
      lc_poses.push_back(pose);
      std::cout<<__func__<<"Read from file, loop closure constrain between "<<id1<<" and "<<id2<<" is \n"<<pose<<"\n";
    }  
    return true;
  }


}
