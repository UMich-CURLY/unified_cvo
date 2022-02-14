#pragma once
#include <cstdio>
#include <iostream>
#include <fstream>
//#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>
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

    int nearest_neighbors_max;
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

    // what to compute inside the inner product
    int is_using_geometry;
    int is_using_intensity;
    int is_using_semantics;
    int is_using_range_ell;
    int is_using_kdtree;
    int is_exporting_association;
    int is_using_geometric_type;

    // for multiframe registration
    int multiframe_using_cpu;
    int multiframe_max_iters;
    float multiframe_ell_init;
    float multiframe_ell_min;
    int  multiframe_iter_per_ell;
    float multiframe_ell_decay_rate;
    int multiframe_iterations_per_ell;
    int multiframe_expected_points;
    
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
      is_full_ip_matrix(0),
      is_using_geometry(1),
      is_using_intensity(0),
      is_using_semantics(0),
      is_using_range_ell(0),
      is_using_kdtree(0),
      is_using_geometric_type(0),
      is_exporting_association(0),
      multiframe_using_cpu(1),
      multiframe_max_iters(200),      
      nearest_neighbors_max(512),
      multiframe_ell_init(0.15),
      multiframe_ell_min(0.05),
      multiframe_iter_per_ell(10),
      multiframe_ell_decay_rate(0.7),
      multiframe_iterations_per_ell(8),
      multiframe_expected_points(1000)
    {}
    
  };

  /*
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
    params->is_using_geometry = (int) fs["is_using_geometry"];
    params->is_using_intensity = (int) fs["is_using_intensity"];
    params->is_using_semantics = (int) fs["is_using_semantics"];
    params->is_using_range_ell = (int) fs["is_using_range_ell"];
    params->is_using_kdtree = (int) fs["is_using_kdtree"];
    params->is_exporting_association = (int) fs["is_exporting_association"];
    //if (fs.find("is_using_cpu") != fs.end()) {
    params->is_using_cpu = (int) fs["is_using_cpu"];
    //} else
    //  params->is_using_cpu = 1;
    
    params->nearest_neighbors_max = (int)fs["nearest_neighbors_max"];
    
    std::cout<<"read: ell_init is "<<params->ell_init<<", MAX_ITER is "<<params->MAX_ITER<<", c is "<<params->c<<", d is "<<params->d<<", indicator window size is "<<params->indicator_window_size<<std::endl;
    fs.release();
    return;
    }
  */


  inline void read_CvoParams_yaml(const char *filename, CvoParams * params) {
    YAML::Node fs = YAML::LoadFile(filename);
    std::cout<<"open "<<filename<<std::endl;

    if (fs["ell_init_first_frame"]) 
      params->ell_init_first_frame = fs["ell_init_first_frame"].as<float>();
    if (fs["ell_init"])
      params->ell_init = fs["ell_init"].as<float>();
    if (fs["ell_min"])
      params->ell_min = fs["ell_min"].as<float>();
    if (fs["min_ell_iter_limit"])
      params->min_ell_iter_limit = fs["min_ell_iter_limit"].as<int>();
    if (fs["ell_max"])
      params->ell_max = fs["ell_max"].as<float>();
    if (fs["dl"])
      params->dl = fs["dl"].as<double>();
    if(fs["dl_step"])
      params->dl_step =  fs["dl_step"].as<double>();
    if (fs["sigma"])
      params->sigma = fs["sigma"].as<float>();
    if (fs["sp_thres"])
      params->sp_thres = fs["sp_thres"].as<float>();
    if (fs["c"])
      params->c = fs["c"].as<float>();
    if (fs["d"])
      params->d = fs["d"].as<float>();
    if (fs["c_ell"])
      params->c_ell = fs["c_ell"].as<float>();
    if (fs["c_sigma"])
      params->c_sigma = fs["c_sigma"].as<float>();
    if (fs["s_ell"])
      params->s_ell = fs["s_ell"].as<float>();
    if (fs["s_sigma"])
      params->s_sigma = fs["s_sigma"].as<float>();
    if (fs["MAX_ITER"])
      params->MAX_ITER = fs["MAX_ITER"].as<int>();
    if (fs["eps"])
      params->eps = fs["eps"].as<float>();
    if (fs["eps_2"])
      params->eps_2 = fs["eps_2"].as<float>();
    if (fs["min_step"])
      params->min_step = fs["min_step"].as<float>();
    if (fs["max_step"])
      params->max_step = fs["max_step"].as<float>();
    if (fs["ell_decay_rate"])
      params->ell_decay_rate = fs["ell_decay_rate"].as<float>();
    if (fs["ell_decay_rate_first_frame"])
      params->ell_decay_rate_first_frame = fs["ell_decay_rate_first_frame"].as<float>();
    if (fs["ell_decay_start"])
      params->ell_decay_start = fs["ell_decay_start"].as<int>();
    if (fs["ell_decay_start_first_frame"])
      params->ell_decay_start_first_frame = fs["ell_decay_start_first_frame"].as<int>();
    if (fs["indicator_window_size"])
      params->indicator_window_size = fs["indicator_window_size"].as<int>();
    if (fs["indicator_stable_threshold"])
      params->indicator_stable_threshold = fs["indicator_stable_threshold"].as<float>();
    if (fs["is_pcl_visualization_on"])
      params->is_pcl_visualization_on = fs["is_pcl_visualization_on"].as<int>();
    if (fs["is_using_least_square"])
      params->is_using_least_square = fs["is_using_least_square"].as<int>();
    if (fs["is_full_ip_matrix"])
      params->is_full_ip_matrix = fs["is_full_ip_matrix"].as<int>();
    if (fs["is_using_geometry"])
      params->is_using_geometry = fs["is_using_geometry"].as<int>();
    if (fs["is_using_intensity"])
      params->is_using_intensity = fs["is_using_intensity"].as<int>();
    if (fs["is_using_semantics"])
      params->is_using_semantics = fs["is_using_semantics"].as<int>();
    if (fs["is_using_range_ell"])
      params->is_using_range_ell = fs["is_using_range_ell"].as<int>();
    if (fs["is_using_kdtree"])
      params->is_using_kdtree = fs["is_using_kdtree"].as<int>();
    if (fs["is_using_geometric_type"])
      params->is_using_geometric_type = fs["is_using_geometric_type"].as<int>();
    if (fs["is_exporting_association"])
      params->is_exporting_association = fs["is_exporting_association"].as<int>();
    if (fs["nearest_neighbors_max"])
      params->nearest_neighbors_max = fs["nearest_neighbors_max"].as<int>();
    if (fs["multiframe_using_cpu"])
      params->multiframe_using_cpu = fs["multiframe_using_cpu"].as<int>();
    if (fs["multiframe_ell_init"])
      params->multiframe_ell_init = fs["multiframe_ell_init"].as<float>();
    if (fs["multiframe_max_iters"])
      params->multiframe_max_iters = fs["multiframe_max_iters"].as<int>();
    if (fs["multiframe_ell_min"])
      params->multiframe_ell_min = fs["multiframe_ell_min"].as<float>();
    if (fs["multiframe_ell_decay_rate"])
      params->multiframe_ell_decay_rate = fs["multiframe_ell_decay_rate"].as<float>();
    if (fs["multiframe_iterations_per_ell"])
      params->multiframe_iterations_per_ell = fs["multiframe_iterations_per_ell"].as<int>();
    if (fs["multiframe_expected_points"])
      params->multiframe_expected_points = fs["multiframe_expected_points"].as<int>();

    
    
    std::cout<<"read: ell_init is "<<params->ell_init<<", MAX_ITER is "<<params->MAX_ITER<<", c is "<<params->c<<", d is "<<params->d<<", indicator window size is "<<params->indicator_window_size<<std::endl;

    return;
    }

  
}
