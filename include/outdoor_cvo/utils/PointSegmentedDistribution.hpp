#pragma once

// for the newly defined pointtype
#define PCL_NO_PRECOMPILE

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <boost/shared_ptr.hpp>
#include <pcl/impl/point_types.hpp>

#include <Eigen/Core>

//#define NUM_CLASSES 19
//#define FEATURE_DIMENSIONS 5


namespace pcl {


  template <unsigned int FEATURE_DIM, unsigned int NUM_CLASS>
  struct PointSegmentedDistribution
  {
    PCL_ADD_POINT4D;                 
    PCL_ADD_RGB;
    float features[FEATURE_DIM];
    int   label;
    float label_distribution[NUM_CLASS];   // templated on any number of classes. TODO
    float normal[3];
    //float label_distribution[14];
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned

#ifdef __CUDACC__ 
    inline __host__ __device__ PointSegmentedDistribution() {
#else
    inline PointSegmentedDistribution() {
#endif
      this->x = 0;
      this->y = 0;
      this->z = 0;
      label = -1;
      this->r = 0;
      this->g = 0;
      this->b = 0;
      memset(features, 0, sizeof(float) * FEATURE_DIM);
      memset(label_distribution, 0, sizeof(float) * NUM_CLASS);
      memset(normal, 0, sizeof(float)*3);
    }
    /*
    __host__ __device__ PointSegmentedDistribution(const PointSegmentedDistribution<FEATURE_DIM, NUM_CLASS>& p_init) {
      x = p_init.x;
      y = p_init.y;
      z = p_init.z;
      r = p_init.r;
      g = p_init.g;
      b = p_init.b;
      label = p_init.label;
      memcpy(features, p_init.features, FEATURE_DIMENSIONS * sizeof(float) );
      memcpy(label_distribution, p_init.label_distribution, NUM_CLASSES * sizeof(float) );      
    }*/
#ifdef __CUDACC__
    __host__ __device__ 
#endif
    PointSegmentedDistribution(float a, float b, float c) {
      this->x = a;
      this->y = b;
      this->z = c;
      label = -1;
      this->r = 0;
      this->g = 0;
      this->b = 0;
      memset(features, 0, sizeof(float) * FEATURE_DIM);
      memset(label_distribution, 0, sizeof(float) * NUM_CLASS);
      memset(normal, 0, sizeof(float)*3);
    }
    
  } EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment


  template <unsigned int FEATURE_DIM, unsigned int NUM_CLASS, typename PointWithXYZRGB>
  void PointSeg_to_PointXYZRGB(const pcl::PointCloud<pcl::PointSegmentedDistribution<FEATURE_DIM, NUM_CLASS>> & pc_seg,
                               typename pcl::PointCloud<PointWithXYZRGB> & pc_rgb) {
    pc_rgb.resize(pc_seg.size());
    for (int i = 0; i < pc_rgb.size(); i++) {
      auto & p_rgb = pc_rgb[i];
      auto & p_seg = pc_seg[i];

      p_rgb.x = p_seg.x;
      p_rgb.y = p_seg.y;
      p_rgb.z = p_seg.z;
      p_rgb.r = p_seg.r;
      p_rgb.g = p_seg.g;
      p_rgb.b = p_seg.b;
      
    }
    pc_rgb.header = pc_seg.header;
  }

  template <unsigned int FEATRURE_DIM, unsigned int NUM_CLASS >
  void PointSeg_to_PointXYZ(const pcl::PointCloud<pcl::PointSegmentedDistribution<FEATRURE_DIM, NUM_CLASS>> & pc_seg,
                            pcl::PointCloud<pcl::PointXYZ> & pc) {
    pc.resize(pc_seg.size());
    for (int i = 0; i < pc.size(); i++) {
      auto & p = pc[i];
      auto & p_seg = pc_seg[i];

      p.x = p_seg.x;
      p.y = p_seg.y;
      p.z = p_seg.z;
    }
    pc.header = pc_seg.header;
  }
    
    template <unsigned int FEATRURE_DIM, unsigned int NUM_CLASS >
    void print_point(const pcl::PointSegmentedDistribution<FEATRURE_DIM, NUM_CLASS> & p) {
      std::cout<<"The point is at ("<<p.x<<", "<<p.y<<", "<<p.z<<")\n";
      std::cout<<"the features are ";
      for (int i = 0; i < FEATURE_DIMENSIONS; i++)
        std::cout<<p.features[i]<<", ";
      std::cout<<std::endl;
      for (int i = 0; i < NUM_CLASSES; i++)
        std::cout<<p.label_distribution[i]<<", ";
      std::cout<<std::endl;
                           
    }




}
