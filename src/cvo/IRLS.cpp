#include "cvo/IRLS.hpp"
#include "cvo/IRLS_State.hpp"
#include "cvo/local_parameterization_se3.hpp"
//#include "cvo/IRLS_cuSolver.hpp"
#include <sophus/se3.hpp>
#include <ceres/local_parameterization.h>
#include <memory>
#include <chrono>
#include "cvo/IRLS_State.hpp"
#include "cvo/CvoFrame.hpp"
#include "cvo/CvoParams.hpp"
#include <ceres/ceres.h>
#include <string>
#include "utils/CvoPointCloud.hpp"
#include "utils/PoseLoader.hpp"
namespace cvo {

  CvoBatchIRLS::CvoBatchIRLS(//const std::vector<Mat34d_row, Eigen::aligned_allocator<Mat34d_row>> & init_poses,
                             const std::vector<CvoFrame::Ptr> &  frames,
                             const std::vector<bool> & pivot_flags,                             
                             const std::list<BinaryState::Ptr> & states,
                             const CvoParams * params
                             ) : pivot_flags_(pivot_flags),
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
                     std::vector<Sophus::SE3d> & poses_curr,
                     const std::string & fname) {
    poses_curr.resize(frames.size());
    for (int i = 0; i < frames.size(); i++) {
      Mat34d_row pose_eigen = Eigen::Map<Mat34d_row>(frames[i]->pose_vec);
      Sophus::SE3d pose(pose_eigen.block<3,3>(0,0), pose_eigen.block<3,1>(0,3));
      poses_curr[i] = pose;
    }
    if (fname.size() > 0) {
      write_traj_file(fname, poses_curr);
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

  static void solve_with_ceres() {
    
  }

  static void solve_with_cuSolver() {
    
  }


  void CvoBatchIRLS::solve() {
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> gt;
    std::string err_file_name;
    solve(gt, err_file_name);
  }

  static void transform_vector_of_poses(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> & poses_in,
                                        //const Sophus::SE3d & pose_anchor,
                                        const CvoBatchIRLS::Mat34d_row & pose_anchor_eigen,
                                        std::vector<Sophus::SE3d> & poses_out) {
    poses_out.resize(poses_in.size());
    
    Sophus::SE3d pose_anchor(pose_anchor_eigen.block<3,3>(0,0), pose_anchor_eigen.block<3,1>(0,3));
    
    for (int i = 1; i < poses_in.size(); i++) {
      Eigen::Matrix4d i_from_0 = poses_in[0].inverse() * poses_in[i];
      Sophus::SE3d poses_in_0_i(i_from_0.block<3,3>(0,0), i_from_0.block<3,1>(0,3));
      std::cout<<__func__<<"\n"<<i_from_0<<"\n"<<poses_in_0_i.matrix()<<"\m"<<poses_in[i]<<"\n";
      poses_out[i] = pose_anchor * poses_in_0_i;
      //std::cout<<"Pose_out: "<<poses_out[i].matrix()<<"\n";
    }
    if (poses_out.size() > 0) {
      poses_out[0] = pose_anchor;
    }
    
  }


  void CvoBatchIRLS::solve(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> & gt,
                           const std::string & err_file_name) {

    
    int iter_ = 0;
    bool converged = false;
    double ell = params_->multiframe_ell_init;

    //std::ofstream nonzeros("nonzeros.txt");
    //std::ofstream loss_change("loss_change.txt");
    std::cout<<"gt.size()   is "<< (gt.size()) <<"\n"
             <<"frame szie is "<<frames_->size()<<"\n"
             <<" err_file_name.size() = "<< err_file_name.size()<<"\n";
    bool is_comparing_with_gt = (gt.size() == frames_->size() && err_file_name.size() > 0);
    std::vector<Sophus::SE3d> gt_aligned(gt.size());
    std::vector<double> err_history;
    std::ofstream err_f;
    if (is_comparing_with_gt) {
      Mat34d_row pose_anchor_eigen = Eigen::Map<Mat34d_row>((*frames_)[0]->pose_vec);
      transform_vector_of_poses(gt, pose_anchor_eigen, gt_aligned);
      err_f.open(err_file_name);
      err_f<<"#ell, nonzeros, err\n";
    }
    
    double ceres_time = 0;
    double kernel_eval_time = 0;
    double ceres_add_residual_time = 0;
    int num_ells = 0;
    int last_nonzeros = 0;
    bool ell_should_decay_when_nonzero_max = false;
    std::cout<<"Solve: total number of states is "<<states_->size()<<std::endl;
    std::string pose_log_fname = "";    
    while (!converged) {

      std::cout<<"\n\n==============================================================\n";
      std::cout<<"Iter "<<iter_<< ": Solved Ceres problem\n";

      std::cout<<"Poses are\n"<<std::flush;
      for (auto && frame: *frames_){
        std::cout<<Eigen::Map<Eigen::Matrix<double,3,4, Eigen::DontAlign>>(frame->pose_vec)<<std::endl<<"\n";
      }

      std::vector<Sophus::SE3d> poses_old(frames_->size());
      pose_log_fname = "";
      pose_snapshot(*frames_, poses_old, pose_log_fname);
      std::vector<float> ell_old;
      ell_old.reserve(states_->size());
      for (auto & state : *states_) 
        ell_old.emplace_back((state->get_ell()));
      


      ceres::Problem problem;
      LocalParameterizationSE3 * se3_parameterization = new LocalParameterizationSE3();
      for (auto & frame : *frames_) {

        std::cout<<"Frame number of points "<<frame->points->num_points()<<std::endl;
        //if (params_->is_using_kdtree == false)
        frame->transform_pointcloud();
        problem.AddParameterBlock(frame->pose_vec, 12, se3_parameterization);
      }

      std::vector<int> invalid_factors(states_->size());
      int counter_nonzer_factors = 0;
      int total_nonzeros = 0;
      unsigned int num_residuals = 0;
      for (auto && state : *states_) {
        
        std::cout<<"ell of factor between " <<(state)->get_frame1()<<" and "<<(state)->get_frame2()<<" is "<<(state) ->get_ell()<<", multiframe_is_optimizing_ell is "<<params_->multiframe_is_optimizing_ell<<"\n";
        
        auto start = std::chrono::system_clock::now();
        if (params_->multiframe_is_free_memory_each_iter)
          state->malloc_state_memory();        
        int nonzeros_ip_mat = state->update_inner_product();
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double, std::milli> t_all = end - start;
        kernel_eval_time += ( (static_cast<double>(t_all.count())) / 1000);
        total_nonzeros += nonzeros_ip_mat;

        start = std::chrono::system_clock::now();
        num_residuals += (state->add_residual_to_problem(problem));
        if (params_->multiframe_is_free_memory_each_iter)
          state->free_state_memory();
        end = std::chrono::system_clock::now();
        t_all = end - start;
        ceres_add_residual_time += ( (static_cast<double>(t_all.count())) / 1000);
        if (nonzeros_ip_mat  > params_->multiframe_min_nonzeros)
          counter_nonzer_factors++;
      }
      if (is_comparing_with_gt) {
        double param_err = change_of_all_poses(gt_aligned, poses_old);
        std::cout<<"Before iter" <<iter_<<"'s optimization, error w.r.t gt is "<<param_err<<std::endl;
        err_f << ell<<","<<total_nonzeros<<","<<param_err<<"\n";
        err_history.push_back(param_err);
      }
      
      //nonzeros <<ell<<", "<< total_nonzeros<<"\n"<<std::flush;

      std::cout<<"Total nonzeros "<<total_nonzeros<<", last_nonzeros "<<last_nonzeros<<", num_residuals is "<<num_residuals<<std::endl;
      std::cout<<"iter_ "<<iter_<<", multiframe_iterations_per_ell "<<params_->multiframe_iterations_per_ell<<std::endl;

      /// break condition 1: reaches max iters or all factors are zero
      if (counter_nonzer_factors == 0)
        break;
      
      //if (iter_ > 0 &&  iter_ % params_->multiframe_iterations_per_ell == 0 )
      //  ell_should_decay_when_nonzero_max = true;
      
      //if (params_->multiframe_iterations_per_ell 
      //    || total_nonzeros > last_nonzeros
      //    || !ell_should_decay_when_nonzero_max
      //    ) {
        
      last_nonzeros = total_nonzeros;
      for (int k = 0; k < frames_->size(); k++) {
        problem.SetParameterization(frames_->at(k)->pose_vec, se3_parameterization);
        if (pivot_flags_.at(k))
          problem.SetParameterBlockConstant(frames_->at(k)->pose_vec);          
      }


      ceres::Solver::Options options;
      options.function_tolerance = 1e-5;
      options.gradient_tolerance = 1e-5;
      options.parameter_tolerance = 1e-5;
      //options.check_gradients = true;
      //options.line_search_direction_type = ceres::BFGS;
      options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
      options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; //ceres::SPARSE_SCHUR;
      //options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY; //ceres::SPARSE_SCHUR;
      //options.preconditioner_type = ceres::JACOBI;
      //options.visibility_clustering_type = ceres::CANONICAL_VIEWS;
      options.num_threads = params_->multiframe_least_squares_num_threads;
      options.max_num_iterations = params_->multiframe_iterations_per_solve;


      auto start = std::chrono::system_clock::now();
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
      auto end = std::chrono::system_clock::now();
      std::chrono::duration<double, std::milli> t_all = end - start;
      ceres_time += ( (static_cast<double>(t_all.count())) / 1000);
        

      std::cout << summary.FullReport() << std::endl;
      //loss_change << ell <<", "<< summary.final_cost - summary.initial_cost <<std::endl;

        
      if (params_->multiframe_is_optimizing_ell == false )  {
        //  if (//param_change < 1e-5 * poses_new.size()

        //num_ells ++;
        //ell_should_decay_when_nonzero_max = false;
        //if (num_ells == 2)
        //  break;

        if (ell <= params_->multiframe_ell_min)
          converged = true;
      }
      for (auto && state : *states_) 
        state->update_ell();
        
      std::vector<Sophus::SE3d> poses_new(frames_->size());
      if (is_comparing_with_gt)      
        pose_log_fname = "pose_iter_"+std::to_string(iter_)+".txt";      
      pose_snapshot(*frames_, poses_new, pose_log_fname);
      double param_change = change_of_all_poses(poses_old, poses_new);
      std::cout<<"Pose Update is "<<param_change<<std::endl;

      std::vector<float> ell_new;
      ell_new.reserve(states_->size());
      for (auto & state : *states_) 
        ell_new.emplace_back((state->get_ell()));
      double ell_change = 0;
      for (int l = 0; l < ell_new.size(); l++) 
        ell_change += std::fabs(ell_new[l] - ell_old[l]);
      std::cout<<"Ell Update is "<<ell_change<<std::endl;      
      if (param_change < 1e-8 && ell_change < 1e-3)
        converged = true;
      

      //last_nonzeros = 0;              
      iter_++;
      if (iter_ >= params_->multiframe_max_iters)
        converged = true;
        
    }

    //nonzeros.close();
    //loss_change.close();
    if (is_comparing_with_gt) {
      std::cout<<"The gt poses putting into the local frame is \n";
      for (int j = 0; j < gt_aligned.size(); j++) {
        std::cout<<gt_aligned[j].matrix()<<"\n";
      }
      
      std::vector<Sophus::SE3d> poses_new(frames_->size());
      pose_log_fname = ""; //"pose_iter_"+std::to_string(iter_)+".txt";
      pose_snapshot(*frames_, poses_new, pose_log_fname);
      double param_err = change_of_all_poses( gt_aligned, poses_new);
      err_history.push_back(param_err);
      std::cout<<"Finish registration at iter " <<iter_<<", error w.r.t gt changes from "<<err_history[0]<<" to " <<param_err<<std::endl;
      
      //err_f << param_err<<"\n";
      err_f.close();
    }

    std::cout<<"kernel eval time is "<<kernel_eval_time<<"\n";
    std::cout<<"ceres add residual time is "<<ceres_add_residual_time<<"\n";    
    std::cout<<"ceres running time is "<<ceres_time<<"\n";
  }

}
