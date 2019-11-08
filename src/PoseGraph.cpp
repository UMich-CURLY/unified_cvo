#include <iostream>
#include <cstdio>
#include <fstream>
#include <cassert>

// Graphs
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/geometry/Pose3.h>
// Factors
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>

#include "graph_optimizer/PoseGraph.hpp"
#include "utils/data_type.hpp"
#include "utils/conversions.hpp"
namespace cvo {

  using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

  
  PoseGraph::PoseGraph():
    isam2_(nullptr){
    // ISAM2 solver
    gtsam::ISAM2Params isam_params;
    isam_params.relinearizeThreshold = 0.01;
    isam_params.relinearizeSkip = 1;
    isam_params.cacheLinearizedFactors = false;
    isam_params.enableDetailedResults = true;
    isam_params.print();
    this->isam2_ .reset( new gtsam::ISAM2(isam_params));
    
  }

  PoseGraph::~PoseGraph() {
    
    
  }

  static Eigen::Affine3f read_tracking_init_guess() {
    FILE * tracking_init_guess_f = fopen("cvo_init.txt", "r");
    Eigen::Affine3f init_guess = Eigen::Affine3f::Identity();
    auto & m = init_guess.matrix();
    if (tracking_init_guess_f) {
      fscanf(tracking_init_guess_f,
             "%f %f %f %f %f %f %f %f %f %f %f %f\n",
             &m(0,0), &m(0,1), &m(0,2), &m(0,3),
             &m(1,0), &m(1,1), &m(1,2), &m(1,3),
             &m(2,0), &m(2,1), &m(2,2), &m(2,3));
      fclose(tracking_init_guess_f);
    } else {
      printf("No tracking init guess file found! use identity\n");
      m.setIdentity();
    }
    printf("First init guess of cvo tracking is \n");
    std::cout<<m<<std::endl;
    return init_guess;
  }

  Eigen::Affine3f PoseGraph::compute_frame_pose_in_graph(std::shared_ptr<Frame> frame) {
    Eigen::Affine3f output;
    if (frame->is_keyframe()) {
      output = frame->pose_in_graph();
    } else {
      int ref_id = frame->tracking_relative_transform().ref_frame_id();
      Eigen::Affine3f ref_frame_pose = id2keyframe_[ref_id]->pose_in_graph();
      output = ref_frame_pose * frame->tracking_relative_transform().ref_frame_to_curr_frame();
    }
    return output;
  }

  float PoseGraph::track_new_frame(std::shared_ptr<Frame> new_frame,
                                   bool & is_keyframe) {
    Eigen::Affine3f tracking_pose;
    if (tracking_relative_transforms_.size() == 0) {
      is_keyframe = true;
      all_frames_since_last_keyframe_ = {};
      tracking_pose.setIdentity();
      new_frame->set_relative_transform(new_frame->id, tracking_pose, 1);

    } else {

      Eigen::Affine3f cvo_init;
      auto  last_keyframe = all_frames_since_last_keyframe_[0];
      auto  last_frame = last_two_frames_.back();
      auto  slast_frame = last_two_frames_.front();
      if (tracking_relative_transforms_.size() == 1)
        cvo_init = read_tracking_init_guess();
      else {
        Eigen::Affine3f slast_frame_pose_in_graph = compute_frame_pose_in_graph(slast_frame);
        Eigen::Affine3f last_frame_pose_in_graph = compute_frame_pose_in_graph(last_frame);

        Eigen::Affine3f slast_frame_to_last_frame = slast_frame_pose_in_graph.inverse() * last_frame_pose_in_graph;
        Eigen::Affine3f last_keyframe_to_last_frame = last_keyframe->pose_in_graph().inverse() * last_frame_pose_in_graph;
        cvo_init = (last_keyframe_to_last_frame * slast_frame_to_last_frame).inverse();
      }
      auto & last_kf_points = last_keyframe->points();
      auto & curr_points = new_frame->points();
      printf("Call Set pcd from frame %d to frame %d\n", last_keyframe->id, new_frame->id);
      cvo_align_.set_pcd(last_kf_points, curr_points, cvo_init, true);
      cvo_align_.align();

      Eigen::Affine3f result = cvo_align_.get_transform();
      auto inner_prod = cvo_align_.inner_product();
      new_frame->set_relative_transform(last_keyframe->id, result, inner_prod);
      std::cout<<"Cvo Align Result between "<<last_keyframe->id<<" and "<<new_frame->id<<",inner product "<<inner_prod<<", transformation is \n" <<result.matrix()<<"\n";
      // decide keyframe
      if (inner_prod < 1.8) {
        is_keyframe = true;
        if (last_keyframe != last_frame) {
          auto last_frame_points = last_frame->points();
          Eigen::Affine3f eye = Eigen::Affine3f::Identity();
          cvo_align_.set_pcd(last_frame_points, curr_points, eye, true );
          cvo_align_.align();
          new_frame->set_relative_transform(last_frame->id, cvo_align_.get_transform(),
                                            cvo_align_.inner_product());
          std::cout<<"Due to keyframe, retrack: Cvo Align Result between "<<last_frame->id<<" and "<<new_frame->id<<",inner product "<<cvo_align_.inner_product()<<", transformation is \n" <<cvo_align_.get_transform().matrix() <<"\n";
        }
      } else {
        is_keyframe = false;

      }
    }
    tracking_relative_transforms_.push_back(new_frame->tracking_relative_transform());
    new_frame->set_keyframe(is_keyframe);
    return new_frame->tracking_relative_transform().cvo_inner_product();
  }
  
  void PoseGraph::add_new_frame(std::shared_ptr<Frame> new_frame) {
    std::cout<<"add_new_frame: id "<<new_frame->id<<std::endl;
    std::cout<<"---- number of points is "<<new_frame->points().num_points()<<std::endl;
    //new_frame->points().write_to_color_pcd(std::to_string(new_frame->id)+".pcd"  );
    bool is_keyframe = false;

    // tracking
    track_new_frame(new_frame, is_keyframe);

    // deal with keyframe and nonkeyframe
    printf("Tracking: is keyframe is %d\n", is_keyframe);
    if(is_keyframe) {
      // pose graph optimization
      id2keyframe_[new_frame->id] = new_frame;
      if (keyframes_.size() == 0) {
        init_pose_graph(new_frame);
      } else {
        pose_graph_optimize(new_frame);
      }
      keyframes_.push_back (new_frame);
      all_frames_since_last_keyframe_.clear();
      new_frame->construct_map();
      
    } else {
      int ref_frame_id = tracking_relative_transforms_[new_frame->id].ref_frame_id();
      auto ref_frame = id2keyframe_[ref_frame_id];
      ref_frame->add_points_to_map_from(*new_frame);
    }

    // maintain the data structures in PoseGraph.hpp
    all_frames_since_last_keyframe_.push_back(new_frame);
    last_two_frames_.push(new_frame);
    if (last_two_frames_.size() > 2)
      last_two_frames_.pop();
    
    new_frame->set_keyframe(is_keyframe);
  }

  void PoseGraph::init_pose_graph(std::shared_ptr<Frame> new_frame) {
    //fill config values
    gtsam::Vector4 q_WtoC;
    q_WtoC << 0,0,0,1;
    gtsam::Vector3 t_WtoC;
    t_WtoC << 0,0,0;
    gtsam::Vector6 prior_pose_noise;
    prior_pose_noise << gtsam::Vector3::Constant(0.1), gtsam::Vector3::Constant(0.1);

    // prior state and noise
    gtsam::Pose3 prior_state(gtsam::Quaternion(q_WtoC(3), q_WtoC(0), q_WtoC(1), q_WtoC(2)),
                             t_WtoC);
    auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas( prior_pose_noise);
    
    //factor_graph_.add(gtsam::PriorFactor<gtsam::Pose3>(X(new_frame->id),
    //                                                   prior_state, pose_noise));
    factor_graph_.add(gtsam::PriorFactor<gtsam::Pose3>((new_frame->id),
                                                       prior_state, pose_noise));
    //graph_values_.insert(X(new_frame->id), prior_state);
    graph_values_.insert((new_frame->id), prior_state);
    //key2id_[X(new_frame->id)] = id;

    factor_graph_.print("gtsam Initial Graph\n");
    
  }

  void PoseGraph::pose_graph_optimize(std::shared_ptr<Frame> new_frame) {

    assert(tracking_relative_transforms_.size() > 1);
    

    int new_id = new_frame->id;
    int last_kf_id = all_frames_since_last_keyframe_[0]->id;
    auto last_kf = id2keyframe_[last_kf_id];
    
    auto tf_last_keyframe_to_last_frame = tracking_relative_transforms_[new_id - 1].ref_frame_to_curr_frame();
    auto tf_last_keyframe_to_newframe = tf_last_keyframe_to_last_frame * tracking_relative_transforms_[new_id].ref_frame_to_curr_frame();

    Eigen::Affine3f tf_WtoNew_eigen = last_kf->pose_in_graph() * tf_last_keyframe_to_newframe;
    gtsam::Pose3 tf_WtoNew = affine3f_to_pose3(tf_WtoNew_eigen);
    gtsam::Pose3 odom_last_kf_to_new = affine3f_to_pose3(tf_last_keyframe_to_newframe);
    // TODO? use the noise from inner product??
    gtsam::Vector6 prior_pose_noise;
    prior_pose_noise << gtsam::Vector3::Constant(0.1), gtsam::Vector3::Constant(0.1);
    auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas( prior_pose_noise);
    std::cout<<"optimize the pose graph with gtsam...\n";
    std::cout<<" new frames's tf_WtoNew \n"<<tf_WtoNew<<"\n";
    std::cout<<" new frames' odom_last_kf_to_new \n"<<odom_last_kf_to_new<<"\n";
    // factor_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(X(last_kf_id), X(new_id),
    //                                                    odom_last_kf_to_new, pose_noise));
    factor_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>((last_kf_id), (new_id),
                                                         odom_last_kf_to_new, pose_noise));
    // TOOD: add init value for this state
    //graph_values_.insert(X(new_id), tf_WtoNew);
    graph_values_.insert((new_id), tf_WtoNew);
    //key2id_
    std::cout<<"Just add new keyframe to the graph, the size of keyframe_ (without the new frame) is "<<keyframes_.size()<<"\n";
    //TODO align two functions to get another between factor

    if (keyframes_.size()>1) {
      auto kf_second_last = keyframes_[keyframes_.size()-2];
      auto kf_second_last_id = kf_second_last->id;
      printf("doing map2map align between frame %d and %d\n", kf_second_last_id, keyframes_[keyframes_.size()-1]->id );
      std::unique_ptr<CvoPointCloud> map_points_kf_second_last = kf_second_last->export_points_from_map();
      std::unique_ptr<CvoPointCloud> map_points_kf_last = keyframes_[keyframes_.size()-1]->export_points_from_map();
      std::cout<<"Map points from the two kf exported\n"<<std::flush;
      Eigen::Affine3f init_guess = kf_second_last->pose_in_graph().inverse() * last_kf->pose_in_graph();
      cvo_align_.set_pcd(*map_points_kf_second_last, *map_points_kf_last,
                         init_guess, true);

      Eigen::Affine3f cvo_result = cvo_align_.get_transform();
      std::cout<<"map2map transform is \n"<<cvo_result.matrix()<<std::endl;
      // TODO: check cvo align quality
      gtsam::Pose3 tf_slast_kf_to_last_kf = affine3f_to_pose3(cvo_result);
      //factor_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(X(kf_second_last_id ), X(last_kf_id ),
      //                                                     tf_slast_kf_to_last_kf, pose_noise));
      factor_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>((kf_second_last_id ), (last_kf_id ),
                                                           tf_slast_kf_to_last_kf, pose_noise));
      graph_values_.print("\ngraph init values\n");
      std::cout<<"Just add the edge between two  maps\n"<<std::flush;
    }
    try {
      factor_graph_.print("factor graph\n");
      gtsam::ISAM2Result result = isam2_->update(factor_graph_, graph_values_ ); // difference from optimize()?
      graph_values_ = isam2_->calculateEstimate();

      std::cout<<"Optimization finish\n";
      update_optimized_poses_to_frames();
      factor_graph_.resize(0);
      graph_values_.clear();
    } catch(gtsam::IndeterminantLinearSystemException &e) {
      std::cerr<<("FORSTER2 gtsam indeterminate linear system exception!\n");
      std::cerr << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }

    
  }

  void PoseGraph::update_optimized_poses_to_frames() {
    std::cout<<"graph key size: "<<graph_values_.size()<<std::endl;
    for (auto key : graph_values_.keys()) {
      std::cout<<"key: "<<key<<". "<<std::flush;
      gtsam::Pose3 pose_gtsam= graph_values_.at<gtsam::Pose3>( key ) ;
      std::cout<<"pose_gtsam "<<pose_gtsam<<std::endl<<std::flush;
      Mat44 pose_mat = pose_gtsam.matrix();
      Eigen::Affine3f pose;
      pose.linear() = pose_mat.block(0,0,3,3).cast<float>();
      pose.translation() = pose_mat.block(0,3,3,1).cast<float>();
      id2keyframe_[key]->set_pose_in_graph(pose);
      std::cout<<"frame "<<key<< " new pose_in_graph is \n"<<id2keyframe_[key]->pose_in_graph().matrix()<<std::endl;
    }
    
  }

  void PoseGraph::write_trajectory(std::string filename) {
    std::ofstream outfile (filename);
    if (outfile.is_open()) {
      
      //outfile.write()
        
      outfile.close();
    }
    
  }
  
}
