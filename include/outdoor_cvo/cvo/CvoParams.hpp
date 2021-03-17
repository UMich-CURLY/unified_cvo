#pragma once
#include <cstdio>
#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>

namespace cvo {

  
  
  // all params to tune :(
  struct CvoParams {
    float ell_init_first_frame;
    float ell_init;
    float ell_min;
    int min_ell_iter_limit;
    float ell_max;
    double dl;           // changes for ell in each iteration
    double dl_step;
    float sigma;        // kernel signal variance (set as std)      
    float sp_thres;     // kernel sparsification threshold       
    float c;            // so(3) inner product scale     
    float d;            // R^3 inner product scale
    float c_ell;        // kernel characteristic length-scale for color kernel
    float c_sigma;      // kernel signal variance for color kernel
    float s_ell;        // length-scale for semantic labels
    float s_sigma;      // signal variance for semantic labels
    int MAX_ITER;       // maximum number of iteration
    float eps;          // the program stops if norm(omega)+norm(v) < eps
    float eps_2;        // threshold for se3 distance
    float min_step;     // minimum step size for integration
    float max_step;
    float step;         // integration step size

    float ell_decay_rate;

    float ell_decay_rate_first_frame;
    int ell_decay_start;
    int ell_decay_start_first_frame;
    
    int indicator_window_size;
    float indicator_stable_threshold;
    int is_pcl_visualization_on;
    int is_using_least_square;
    
    int is_ell_adaptive;
    int is_full_ip_matrix;
    
    CvoParams() :
      ell_init_first_frame(0.5),
      ell_init(0.5),
      ell_min(0.05),
      min_ell_iter_limit(1),
      ell_max(1.2),
      dl(0),
      dl_step(0.3),
      sigma(0.1),
      sp_thres(0.0006),
      c(7.0),
      d(7.0),
      c_ell(0.15),
      c_sigma(0.6),
      s_ell(0.1),
      s_sigma(0.8),
      MAX_ITER(10000),
      min_step(2e-5),
      eps(0.00005),
      eps_2(0.000012),
      ell_decay_rate(0.9),
      ell_decay_rate_first_frame(0.99),
      ell_decay_start(30),
      ell_decay_start_first_frame(300),
      indicator_window_size(15),
      indicator_stable_threshold(0.2),
      is_pcl_visualization_on(0),
      is_using_least_square(0),
      is_ell_adaptive(0),
      is_full_ip_matrix(0){}
    
  };

  inline void read_CvoParams_yaml(const char *filename, CvoParams * params) {
    cv::FileStorage fs;
    std::cout<<"open "<<filename<<std::endl;
    fs.open(std::string(filename), cv::FileStorage::READ );
    if (!fs.isOpened()) {
      std::cerr << "Failed to open CvoParam file " << filename << std::endl;
      return;
    }

    params->ell_init_first_frame = (float) fs["ell_init_first_frame"];
    params->ell_init = (float) fs["ell_init"];
    params->ell_min = (float) fs["ell_min"];
    params->min_ell_iter_limit = (int) fs["min_ell_iter_limit"];
    params->ell_max = (float) fs["ell_max"];
    params->dl = (double) fs["dl"];
    params->dl_step = (double) fs["dl_step"];
    params->sigma = (float) fs["sigma"];
    params->sp_thres = (float) fs["sp_thres"];
    params->c = (float) fs["c"];
    params->d = (float) fs["d"];
    params->c_ell = (float) fs["c_ell"];
    params->c_sigma = (float) fs["c_sigma"];
    params->s_ell = (float) fs["s_ell"];
    params->s_sigma = (float) fs["s_sigma"];
    params->MAX_ITER = (float) fs["MAX_ITER"];
    params->eps = (float) fs["eps"];
    params->eps_2 = (float) fs["eps_2"];
    params->min_step = (float) fs["min_step"];

    params->max_step = (float) fs["max_step"];

    params->ell_decay_rate = (float) fs["ell_decay_rate"];
    params->ell_decay_rate_first_frame = (float) fs["ell_decay_rate_first_frame"];
    
    params->ell_decay_start = (int) fs["ell_decay_start"];
    params->ell_decay_start_first_frame = (int) fs["ell_decay_start_first_frame"];
    
    params->indicator_window_size = (int) fs["indicator_window_size"];
    params->indicator_stable_threshold = (float) fs["indicator_stable_threshold"];
    params->is_pcl_visualization_on = (int) fs["is_pcl_visualization_on"];
    params->is_using_least_square = (int) fs["is_using_least_square"];
    params->is_full_ip_matrix = (int) fs["is_full_ip_matrix"];
    std::cout<<"read: ell_init is "<<params->ell_init<<", MAX_ITER is "<<params->MAX_ITER<<", c is "<<params->c<<", d is "<<params->d<<", indicator window size is "<<params->indicator_window_size<<std::endl;
    fs.release();
    return;
  }

  
}
