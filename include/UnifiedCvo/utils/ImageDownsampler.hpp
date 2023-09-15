#pragma once
#include <opencv2/opencv.hpp>
#include "CvoPointCloud.hpp"
#include "utils/ImageStereo.hpp"
#include "utils/CvoPointCloud.hpp"
#include <memory>
#include <iostream>
#include "utils/Calibration.hpp"
#include "utils/VoxelMap.hpp"

namespace cvo {
  /*  
  std::shared_ptr<cvo::CvoPointCloud> stereo_downsampling(const cv::Mat & left, const cv::Mat & right,
                                                          const cvo::Calibration & calib,
                                                          float multiframe_downsample_voxel_size){
    
    std::shared_ptr<cvo::ImageStereo> raw(new cvo::ImageStereo(left, right));
    std::shared_ptr<cvo::CvoPointCloud> pc_full(new cvo::CvoPointCloud(*raw,  calib, cvo::CvoPointCloud::FULL));
    std::shared_ptr<cvo::CvoPointCloud> pc_edge_raw(new cvo::CvoPointCloud(*raw, calib, cvo::CvoPointCloud::DSO_EDGES));


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr raw_pcd_edge(new pcl::PointCloud<pcl::PointXYZRGB>);
    pc_edge_raw->export_to_pcd<pcl::PointXYZRGB>(*raw_pcd_edge);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr raw_pcd_surface(new pcl::PointCloud<pcl::PointXYZRGB>);    
    pc_full->export_to_pcd<pcl::PointXYZRGB>(*raw_pcd_surface);
    std::cout<<"raw pcd edge size is "<<raw_pcd_edge->size()<<"\n"
             <<"raw pcd surface size is "<<raw_pcd_surface->size()<<"\n";
    

    float leaf_size = multiframe_downsample_voxel_size;
    std::cout<<"Current leaf size is "<<leaf_size<<std::endl;



    // edges
    cvo::VoxelMap<pcl::PointXYZRG> edge_voxel(leaf_size / 5); // /10    
    for (int k = 0; k < raw_pcd_edge->size(); k++)  {
      std::cout<<"inserting "<<raw_pcd_edge->points[k]<<" to voxel\n";
      edge_voxel.insert_point(&raw_pcd_edge->points[k]);
    }
    std::vector<pcl::PointXYZRGB*> edge_results = edge_voxel.sample_points();
    pcl::PointCloud<pcl::PointXYZRGB> edge_pcl;
    edge_pcl.resize(edge_results.size());
    #pragma omp parallel for
    for (int k = 0; k < edge_results.size(); k++)
      edge_pcl[k] = (*edge_results[k]);
    std::cout<<"edge voxel selected points "<<edge_results.size()<<std::endl;

    /// surface
    cvo::VoxelMap<pcl::PointXYZRGB> surface_voxel(leaf_size);    
    for (int k = 0; k < raw_pcd_surface->size(); k++) 
      surface_voxel.insert_point(&raw_pcd_surface->points[k]);
    std::vector<pcl::PointXYZRGB*> surface_results = surface_voxel.sample_points();
    pcl::PointCloud<pcl::PointXYZRGB> surface_pcl;
    surface_pcl.resize(surface_results.size());    
    #pragma omp parallel for
    for (int k = 0; k < surface_results.size(); k++)
      surface_pcl[k] = (*surface_results[k]);
    std::cout<<"surface voxel selected points "<<surface_results.size()<<std::endl;

    int total_selected_pts_num = edge_results.size() + surface_results.size();

    std::shared_ptr<cvo::CvoPointCloud> pc_edge(new cvo::CvoPointCloud(edge_pcl, cvo::CvoPointCloud::GeometryType::EDGE));
    std::shared_ptr<cvo::CvoPointCloud> pc_surface(new cvo::CvoPointCloud(surface_pcl, cvo::CvoPointCloud::GeometryType::SURFACE));
    std::shared_ptr<cvo::CvoPointCloud> pc(new cvo::CvoPointCloud);
    *pc = *pc_edge + *pc_surface;
    
    std::cout<<"Voxel number points is "<<pc->num_points()<<std::endl;

    //pcl::PointCloud<pcl::PointXYZRGB> pcd_to_save;
    return pc;
  }
  */

  std::shared_ptr<cvo::CvoPointCloud> voxel_downsample(std::shared_ptr<cvo::CvoPointCloud> pc_in,
                                                       float leaf_size,
                                                       std::unordered_set<const cvo::CvoPoint *> & selected_pts) {
    cvo::VoxelMap<const cvo::CvoPoint> voxel(leaf_size); // /10    
    for (int k = 0; k < pc_in->size(); k++)  {
      //if (k == 0)
      //  std::cout<<"inserting "<<pc_in->xyz_at(k).transpose()<<" to voxel\n";
      voxel.insert_point(&pc_in->point_at(k));
    }
    std::vector<const cvo::CvoPoint*> results = voxel.sample_points();
    std::shared_ptr<cvo::CvoPointCloud> pc(new cvo::CvoPointCloud);//(, cvo::CvoPointCloud::GeometryType::EDGE));
    //pc.reserve(results.size(), FEATURE_DIMENSIONS, NUM_CLASSES);
    for (int k = 0; k < results.size(); k++) {
      if (selected_pts.find(results[k]) == selected_pts.end()) {
        pc->push_back(*results[k]);
        selected_pts.insert(results[k]);
      }
    }
    std::cout<<"Voxel selected points "<<results.size()<<std::endl;
    return pc;
  }
  
  std::shared_ptr<cvo::CvoPointCloud> stereo_downsampling(const cv::Mat & left, const cv::Mat & right,
                                                          const cvo::Calibration & calib,
                                                          float multiframe_downsample_voxel_size){
    
    std::shared_ptr<cvo::ImageStereo> raw(new cvo::ImageStereo(left, right));
    std::shared_ptr<cvo::CvoPointCloud> pc_full(new cvo::CvoPointCloud(*raw,  calib, cvo::CvoPointCloud::FULL));
    std::shared_ptr<cvo::CvoPointCloud> pc_edge_raw(new cvo::CvoPointCloud(*raw, calib, cvo::CvoPointCloud::DSO_EDGES));

    std::cout<<"pc[0] "<<pc_edge_raw->xyz_at(0).transpose();

    float leaf_size = multiframe_downsample_voxel_size;
    std::cout<<"Current leaf size is "<<leaf_size<<std::endl;

    /// edges
    std::unordered_set<const cvo::CvoPoint *> selected_pts;
    std::shared_ptr<cvo::CvoPointCloud> pc_edge = voxel_downsample(pc_edge_raw, leaf_size / 5, selected_pts);
    
    /// surface
    std::shared_ptr<cvo::CvoPointCloud> pc_surface = voxel_downsample(pc_full, leaf_size, selected_pts);

    *pc_edge += *pc_surface;
    std::cout<<"Voxel number points is "<<pc_edge->num_points()<<std::endl;

    return pc_edge;
  }  
  
  
}
