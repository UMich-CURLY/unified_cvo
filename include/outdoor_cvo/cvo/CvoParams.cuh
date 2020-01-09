#pragma once
#include <cstdio>

#define KDTREE_K_SIZE 100

namespace cvo {

  

  // all params to tune :(
  struct CvoParams {
    int gpu_thread_num;
    float kdtree_dist_threshold;
    
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
    
    
  };

  void read_CvoParams(const char * filename, CvoParams * params) {
    FILE * ptr = fopen(filename, "r");
        if (ptr!=NULL) 
    {
      printf( "reading cvo params from file\n");
      fscanf(ptr, "%f\n%f\n%f\n%lf\n%lf\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%d\n%f\n",
             & params->ell_init
             ,& params->ell_min
             ,& params->ell_max
             ,& params->dl
             ,& params->dl_step
             //,& params->min_dl_step
             //,& params->max_dl_step
             ,& params->sigma
             ,& params->sp_thres
             , &params->c
             ,& params->d
             ,& params->c_ell
             ,& params->c_sigma
             , &params->s_ell
             ,& params->s_sigma
             ,& params->MAX_ITER
             ,& params->min_step
             ,& params->eps
             ,& params->eps_2);
      fclose(ptr);
    }
  }


  
}
