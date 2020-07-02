#pragma once
#include <cstdio>
#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>

namespace cvo {

  
  
  // all params to tune :(
  struct CvoParams {
    int gpu_thread_num;
    float kdtree_dist_threshold;

    float ell_init_first_frame;
    float ell_init;
    float ell_min;
    float ell_max;
    double dl;           // changes for ell in each iteration
    double dl_step;
    float sigma;        // kernel signal variance (set as std)      
    float sp_thres;     // kernel sparsification threshold       
    float c;            // so(3) inner product scale     
    float d;            // R^3 inner product scale
    float color_scale;  // color space inner product scale
    float c_ell;        // kernel characteristic length-scale for color kernel
    float c_sigma;      // kernel signal variance for color kernel
    float s_ell;        // length-scale for semantic labels
    float s_sigma;      // signal variance for semantic labels
    int MAX_ITER;       // maximum number of iteration
    float eps;          // the program stops if norm(omega)+norm(v) < eps
    float eps_2;        // threshold for se3 distance
    float min_step;     // minimum step size for integration
    float step;         // integration step size

    float ell_decay_rate;
    int ell_decay_start;
    int indicator_window_size;
    float indicator_stable_threshold;
  };

  inline void read_CvoParams_yaml(const char *filename, CvoParams * params) {
    cv::FileStorage fs;
    std::cout<<"open "<<filename<<std::endl;
    fs.open(std::string(filename), cv::FileStorage::READ );
    if (!fs.isOpened()) {
      std::cerr << "Failed to open " << filename << std::endl;
      return;
    }

    params->ell_init_first_frame = (float) fs["ell_init_first_frame"];
    params->ell_init = (float) fs["ell_init"];
    params->ell_min = (float) fs["ell_min"];
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

    params->ell_decay_rate = (float) fs["ell_decay_rate"];
    params->ell_decay_start = (int) fs["ell_decay_start"];
    params->indicator_window_size = (int) fs["indicator_window_size"];
    params->indicator_stable_threshold = (float) fs["indicator_stable_threshold"];

    std::cout<<"read: ell_init is "<<params->ell_init<<", MAX_ITER is "<<params->MAX_ITER<<", c is "<<params->c<<", d is "<<params->d<<", indicator window size is "<<params->indicator_window_size<<std::endl;
    fs.release();
    return;
  }

  inline void read_CvoParams(const char * filename, CvoParams * params) {
    //FILE * ptr = fopen(filename, "r");
    std::ifstream f;
    f.open(filename);
    if (f.is_open() )  {
      memset(params, 0, sizeof(CvoParams));
      printf( "reading cvo params from file %s\n", filename);
      //fscanf(ptr, "%f\n%f\n%f\n%lf\n%lf\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%d\n%f\n%f\n%f\n",
      f >> params->ell_init
        >> params->ell_min
        >> params->ell_max
        >> params->dl
        >> params->dl_step
        >> params->sigma
        >> params->sp_thres
        >> params->c
        >> params->d
        >> params->c_ell
        >> params->c_sigma
        >> params->s_ell
        >> params->s_sigma
        >> params->MAX_ITER
        >> params->min_step
        >> params->eps
        >> params->eps_2;
      f.close();
    //fclose(ptr);

      std::cout<<"read: ell_init is "<<params->ell_init<<", MAX_ITER is "<<params->MAX_ITER<<", c is "<<params->c<<", d is "<<params->d<<std::endl;
    } else {
      printf("Error: the CvoParam file is empty\n");
    }

  }


  
}
