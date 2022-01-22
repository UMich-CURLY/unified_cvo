#include "cvo/IRLS.hpp"
#include "cvo/IRLS_State_CPU.hpp"
#include "cvo/local_parameterization_se3.hpp"
#include <sophus/se3.hpp>
#include <ceres/local_parameterization.h>
#include <memory>
namespace cvo {
  
  CvoBatchIRLS::CvoBatchIRLS(//const std::vector<Mat34d_row, Eigen::aligned_allocator<Mat34d_row>> & init_poses,
                             const std::vector<CvoFrame::Ptr> &  frames,
                             const std::vector<bool> & pivot_flags,                             
                             const std::list<BinaryState::Ptr> & states,
                             const CvoParams * params
                             ) : pivot_flags_(&pivot_flags),
                                 states_(&states),
                                 frames_(&frames),
                                 params_(params) {
    //problem_.reset(new ceres::Problem);
    //cvo::LocalParameterizationSE3 * se3_parameterization = new LocalParameterizationSE3();

    //for (auto & frame : frames) {
    //  problem_->AddParameterBlock(frame->pose_vec, 12, se3_parameterization);
    //}

    /*
    for (auto & state_ptr : states) {
      std::shared_ptr<BinaryStateCPU> state_ptr_cpu = std::dynamic_pointer_cast<BinaryStateCPU>(state_ptr);
      cvo::CvoBinaryCost * cost_per_state = new cvo::CvoBinaryCost(state_ptr);
      problem_->AddResidualBlock(cost_per_state, nullptr, state_ptr_cpu->frame1->pose_vec, state_ptr_cpu->frame2->pose_vec);

      
   
      if (frames_.find(state_ptr->frame1) == frames_.end()) {
        frames_.insert(state_ptr->frame1);
        //problem_->SetParameterization(state_ptr->frame1->pose_vec, se3_parameterization);              
      }
      
      if (frames_.find(state_ptr->frame2) == frames_.end()) {
        frames_.insert(state_ptr->frame2);
        problem_->SetParameterization(state_ptr->frame2->pose_vec, se3_parameterization);
      }
   
    }
    */


    
  }

  static
  void pose_snapshot(const std::vector<cvo::CvoFrame::Ptr> & frames,
                     std::vector<Sophus::SE3d> & poses_curr ) {
    poses_curr.resize(frames.size());
    for (int i = 0; i < frames.size(); i++) {
      Mat34d_row pose_eigen = Eigen::Map<Mat34d_row>(frames[i]->pose_vec);
      Sophus::SE3d pose(pose_eigen.block<3,3>(0,0), pose_eigen.block<3,1>(0,3));
      poses_curr[i] = pose;
    }
  }

  static
  double change_of_all_poses(std::vector<Sophus::SE3d> & poses_old,
                             std::vector<Sophus::SE3d> & poses_new) {
    double change = 0;
    for (int i = 0; i < poses_old.size(); i++) {
      change += (poses_old[i].inverse() * poses_new[i]).log().norm();
    }
    return change;
  }

  void CvoBatchIRLS::solve() {
    
    int iter_ = 0;
    bool converged = false;
    double ell = params_->multiframe_ell_init;
    
    while (!converged) {

      std::cout<<"\n\n==============================================================\n";
      std::cout << "Iter "<<iter_<< ": Solved Ceres problem, ell is " <<ell<<"\nPoses are\n"<<std::flush;
      for (auto && frame: *frames_){
        std::cout<<Eigen::Map<Mat34d_row>(frame->pose_vec)<<std::endl<<"\n";
      }

      std::vector<Sophus::SE3d> poses_old(frames_->size());
      pose_snapshot(*frames_, poses_old);

      ceres::Problem problem;
      LocalParameterizationSE3 * se3_parameterization = new LocalParameterizationSE3();
      for (auto & frame : *frames_) {
        std::cout<<"Frame number of points "<<frame->points->num_points()<<std::endl;
        problem.AddParameterBlock(frame->pose_vec, 12, se3_parameterization);
      }
      

      std::vector<int> invalid_factors(states_->size());
      int counter = 0;
      for (auto && state : *states_) {
        int invalid_ip_mat = state->update_inner_product();
        //invalid_factors[counter] = invalid_ip_mat;
        if (!invalid_ip_mat) {
          state->add_residual_to_problem(problem);
          counter++;
        }
      }
      if (counter == 0) break;
      //   for (auto && frame : *frames_) {
      for (int k = 0; k < frames_->size(); k++) {
        problem.SetParameterization(frames_->at(k)->pose_vec, se3_parameterization);
        if (pivot_flags_->at(k))
          problem.SetParameterBlockConstant(frames_->at(k)->pose_vec);          
      }


      ceres::Solver::Options options;
      options.function_tolerance = 1e-8;
      options.gradient_tolerance = 1e-8;
      options.parameter_tolerance = 1e-8;
      //options.check_gradients = true;
      //options.line_search_direction_type = ceres::BFGS;
      options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
      options.linear_solver_type = ceres::SPARSE_SCHUR;
      options.preconditioner_type = ceres::JACOBI;
      options.visibility_clustering_type = ceres::CANONICAL_VIEWS;
      
      
      options.num_threads = 24;
      options.max_num_iterations = params_->multiframe_iterations_per_ell;
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);

      std::cout << summary.FullReport() << std::endl;

      std::vector<Sophus::SE3d> poses_new(frames_->size());
      pose_snapshot(*frames_, poses_new);
      double param_change = change_of_all_poses(poses_old, poses_new);
      std::cout<<"Update is "<<param_change<<std::endl;
      if (param_change < 1e-5 * poses_new.size()
          || iter_ % 5 == 0
          ) {
        if (ell >= params_->multiframe_ell_min) {
          ell = ell *  params_->multiframe_ell_decay_rate;
          std::cout<<"Reduce ell to "<<ell<<std::endl;          
          for (auto && state : *states_) 
            std::dynamic_pointer_cast<BinaryStateCPU>(state)->update_ell();
        } else {
          converged = true;
          std::cout<<"End: pose change is "<<param_change<<std::endl;          
        }
      }
      if ( iter_ > params_->multiframe_max_iters) {
        converged = true;
      }
      
      iter_++;
        
    }
    
  }

  void CvoBatchIRLS::remove_frame(CvoFrame::Ptr to_remove) {
    
  }
  void CvoBatchIRLS::add_frame(CvoFrame::Ptr to_remove) {
    
  }

  
}
