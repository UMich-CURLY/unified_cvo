#pragma once
#include "utils/CvoPointCloud.hpp"
#include <memory>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>


namespace cvo {

  
  template <typename PointT>
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr calculate_fpfh_pcl(typename pcl::PointCloud< PointT>::Ptr cloud,
                                                                float normal_radius,
                                                                float fpfh_radius){

    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal> ());

    // Create the normal estimation class, and pass the input dataset to it
    typename pcl::NormalEstimation<PointT, pcl::Normal> ne;
    ne.setInputCloud (cloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    typename pcl::search::KdTree<PointT>::Ptr tree_normal (new typename pcl::search::KdTree<PointT> ());
    ne.setSearchMethod (tree_normal);

    // Output datasets
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

    // Use all neighbors in a sphere of radius 3cm
    ne.setRadiusSearch (normal_radius);

    // Compute the features
    ne.compute (*cloud_normals);


    // Create the FPFH estimation class, and pass the input dataset+normals to it

    typename  pcl::FPFHEstimation<PointT, pcl::Normal, pcl::FPFHSignature33> fpfh;

    fpfh.setInputCloud (cloud);
    fpfh.setInputNormals (cloud_normals);

    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    typename  pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    fpfh.setSearchMethod (tree);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs (new pcl::PointCloud<pcl::FPFHSignature33> ());
    // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!

    fpfh.setRadiusSearch (fpfh_radius);

    // Compute the features
    fpfh.compute (*fpfhs);

    return fpfhs;
  }


  template <typename PointT>
  std::shared_ptr<cvo::CvoPointCloud> load_pcd_and_fpfh(const std::string & source_file,
                                                        float radius_normal,
                                                        float radius_fpfh) {
    
    typename pcl::PointCloud<PointT>::Ptr source_pcd(new typename pcl::PointCloud<PointT>);
    pcl::io::loadPCDFile(source_file, *source_pcd);
    //std::cout<<"Read  source "<<source_pcd->size()<<" points\n";
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_source = calculate_fpfh_pcl<pcl::PointXYZRGB>(source_pcd, radius_normal, radius_fpfh);
    std::shared_ptr<cvo::CvoPointCloud> source(new cvo::CvoPointCloud(*source_pcd));
    for (int i = 0 ; i < source->size(); i++)  {
      memcpy((*source)[i].label_distribution, (*fpfh_source)[i].histogram, sizeof(float)*33  );
    }
    return source;
    
  }

}
