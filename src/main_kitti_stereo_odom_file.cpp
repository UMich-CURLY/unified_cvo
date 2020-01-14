#include <map>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>
#include <opencv2/opencv.hpp>


#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam_unstable/nonlinear/BatchFixedLagSmoother.h>
//#include <gtsam/gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/FastList.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>


#include "dataset_handler/KittiHandler.hpp"
#include "graph_optimizer/Frame.hpp"
#include "graph_optimizer/PoseGraph.hpp"
#include "utils/Calibration.hpp"
#include "utils/data_type.hpp"
#include "utils/conversions.hpp"

using namespace cvo;

gtsam::BetweenFactor<gtsam::Pose3> build_poes_factor(const Aff3f & aff, int id1, int id2,
                                                     float rot_noise=0.1, float trans_noise=0.1) {
  gtsam::Pose3 pose = affine3f_to_pose3(aff);
  gtsam::Vector6 prior_pose_noise;
  prior_pose_noise << gtsam::Vector3::Constant(rot_noise ), gtsam::Vector3::Constant(trans_noise );
  auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas( prior_pose_noise);

  return gtsam::BetweenFactor<gtsam::Pose3>(id1, id2,
                                            pose, pose_noise);
}



int main(int argc, char ** argv) {

  std::string odom_file_name(argv[1]);
  std::string outfile_name(argv[2]);
  float window_size = std::stof(argv[3]);
  float rot_noise = std::stof(argv[4]);
  float trans_noise = std::stof(argv[5]);
  float ip_thres = std::stof(argv[6]);
  gtsam::BatchFixedLagSmoother  smoother_( window_size );
  gtsam::NonlinearFactorGraph factor_graph_;
  gtsam::BatchFixedLagSmoother::KeyTimestampMap timesteps_new_;
  gtsam::Values graph_values_new_, graph_values_results_;
  //fill config values
  gtsam::Vector4 q_WtoC;
  q_WtoC << 0,0,0,1;
  gtsam::Vector3 t_WtoC;
  t_WtoC << 0,0,0;
  gtsam::Vector6 prior_pose_noise;
  prior_pose_noise << gtsam::Vector3::Constant(rot_noise ), gtsam::Vector3::Constant(trans_noise );
  
  // prior state and noise
  gtsam::Pose3 prior_state(gtsam::Quaternion(q_WtoC(3), q_WtoC(0), q_WtoC(1), q_WtoC(2)),
                           t_WtoC);
  auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas( prior_pose_noise);
    
  factor_graph_.add(gtsam::PriorFactor<gtsam::Pose3>(0, prior_state, pose_noise));
  graph_values_new_.insert((0 ), prior_state);
  timesteps_new_[ 0 ] = 0.0;

  std::vector<Aff3f, Eigen::aligned_allocator<Aff3f>> poses; //  pose in world
  std::vector<int> ref_ids;
  std::vector<Aff3f, Eigen::aligned_allocator<Aff3f>> relative_poses; // relative pose
  std::unordered_set<int> keyframe_ids;
  std::vector<int> keyframes;
  keyframe_ids.insert(0);
  ref_ids.push_back(0);
  relative_poses.push_back(Aff3f::Identity());
  poses.push_back(Aff3f::Identity());
  keyframes.push_back(0);


  bool initialized = false;
  int file_keyframe_id_start = 0;
  int file_from_id_start = 0;
  
  std::ifstream infile(odom_file_name);
  if (infile.is_open()) {
    std::string line;
    int newest_id = 0;
    int total_num_keyframes_ = 0;  // 0  is excluded
    int latest_num_keyframes_ = 0;  

    while (!infile.eof()) {
      int from_id, to_id;
      //      int this_kf_id ;
      Aff3f relative_pose = Aff3f::Identity();
      float ip; 
      auto & m = relative_pose.matrix();
      infile >> latest_num_keyframes_ >> from_id >> to_id >> ip;
      infile >> m(0,0) >> m(0,1) >>m(0,2) >> m(0,3)
             >> m(1,0) >> m(1,1) >>m(1,2) >> m(1,3)
             >> m(2,0) >> m(2,1) >>m(2,2) >> m(2,3);

      if (initialized == false) {
        file_keyframe_id_start = latest_num_keyframes_;
        file_from_id_start = from_id;
        total_num_keyframes_ = latest_num_keyframes_;
        initialized = true;
      }
      
      if (from_id == to_id) continue;
      std::cout<<"\n=============================================\n";
      printf("Read num_kf %d, from %d to %d\n", latest_num_keyframes_, from_id, to_id  );
      gtsam::Pose3 pose = affine3f_to_pose3(relative_pose);

      // deal with pose list
      if (to_id > newest_id) {
        newest_id = to_id;
        if ( keyframe_ids.find(from_id) != keyframe_ids.end()  ) {
          poses.push_back(poses[from_id - file_from_id_start] * relative_pose ); // init pose
        } else {
          poses.push_back(poses[ref_ids[from_id - file_from_id_start] - file_from_id_start] * relative_poses[from_id - file_from_id_start ] * relative_pose);
        }
        newest_id = to_id;
        ref_ids.push_back ( from_id);
        relative_poses.push_back( relative_pose);
        std::cout<<"add new frame "<<newest_id<<", total frame size is "<<poses.size()<<std::endl;
      }

      // deal with pose graph
      if (latest_num_keyframes_-1  > total_num_keyframes_)  {
        int new_kf_id = from_id;

        // just change  the new keyframe!
        // first, optimize previous
        std::cout<<"ready to do graph optimization \n";
        factor_graph_.print();
        gtsam::FixedLagSmoother::Result  result_smoother = smoother_.update(factor_graph_, graph_values_new_, timesteps_new_);
        graph_values_results_ = smoother_.calculateEstimate();
        std::cout<<"Optimization finish. Smoother result is  \n";
        result_smoother.print();
        graph_values_results_.print(" all graph values after optimization");
        timesteps_new_.clear();
        factor_graph_.resize(0);
        graph_values_new_.clear();
        for (auto key : graph_values_results_.keys()) {
          std::cout<<"update pose with key: "<<key<<". "<<std::flush;
          gtsam::Pose3 pose_gtsam = graph_values_results_.at<gtsam::Pose3>(key);
          Mat44 mat = pose_gtsam.matrix();
          Mat44f pose_mat = mat.cast<float>();
          Eigen::Affine3f pose;
          pose.linear() = pose_mat.block(0,0,3,3).cast<float>();
          pose.translation() = pose_mat.block(0,3,3,1).cast<float>();
          poses[key-file_from_id_start] = pose;
        }

        std::cout<<"Add new keyframe "<<new_kf_id<<std::endl;
        // add new to the factor graph and add ne wvalues
        int last_kf_id = keyframes[keyframes.size()-1];
        // compute initial pose 
        Aff3f init_pose_new_kf, last_kf_to_new_kf;
        if (last_kf_id == new_kf_id - 1) {
          init_pose_new_kf = poses[last_kf_id-file_from_id_start] * relative_poses[new_kf_id-file_from_id_start];
          last_kf_to_new_kf = relative_poses[new_kf_id-file_from_id_start];
        } else {
          init_pose_new_kf = poses[last_kf_id - file_from_id_start] * relative_poses[new_kf_id-1 - file_from_id_start] * relative_poses[new_kf_id - file_from_id_start];
          last_kf_to_new_kf = relative_poses[new_kf_id-1-file_from_id_start] * relative_poses[new_kf_id-file_from_id_start];
        }
        std::cout<<"init pose is\n "<<init_pose_new_kf.matrix()<<std::endl;
        factor_graph_.add(build_poes_factor(last_kf_to_new_kf, last_kf_id, new_kf_id, 0.2, 0.1 ));
        std::cout<<"add between factor from "<<last_kf_id<<" to "<<new_kf_id;
        std::cout<<last_kf_to_new_kf.matrix()<<std::endl;
        timesteps_new_[new_kf_id - file_from_id_start] = (double) (total_num_keyframes_ + 1);
        std::cout<<" with new timestep "<<total_num_keyframes_+1<<"\n";
        graph_values_new_.insert(new_kf_id, affine3f_to_pose3( init_pose_new_kf ) );
        keyframe_ids.insert(new_kf_id);
        keyframes.push_back(new_kf_id);
        
        total_num_keyframes_++;
      } else{
        if (keyframe_ids.find(from_id) != keyframe_ids.end() &&
            keyframe_ids.find(to_id) != keyframe_ids.end()) {
          if (ip > ip_thres ) {
		factor_graph_.add(build_poes_factor(relative_pose, from_id, to_id, rot_noise, trans_noise));
          std::cout<<"add between factor from "<<from_id<<" to "<<to_id<<std::endl;
          std::cout<<relative_pose.matrix()<<std::endl;
	  }
	  }
      }
    }

    
    infile.close();

    std::ofstream outfile(outfile_name);
    if (outfile.is_open()) {
      // write traj
      for (int i = 0; i < poses.size(); i++) {
        Mat44f m;
	bool w = false;
        if (keyframe_ids.find(i+file_from_id_start) != keyframe_ids.end()) {
          m = poses[i].matrix();
	  w = true;
        } else {
          if (keyframe_ids.find(ref_ids[i+file_from_id_start])!= keyframe_ids.end() ) {
            m = (poses[ref_ids[i]-file_from_id_start] * relative_poses[i]).matrix();
            w = true;
	  }
        }
	if(w)
        outfile << m(0,0) <<" "<< m(0,1) <<" "<< m(0,2)  <<" "<< m(0,3) <<" "
                << m(1,0) << " "<< m(1,1) <<" "<< m(1,2) <<" "<< m(1,3) <<" "
                << m (2,0) << " "<< m(2,1) << " "<< m(2,2) << " "<< m(2,3)<<std::endl;


        
      }
      
    }
    
    
  }
    
  
  return 0;
}
