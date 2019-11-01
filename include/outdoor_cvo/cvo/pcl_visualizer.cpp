/* ----------------------------------------------------------------------------
 * Copyright 2019, Tzu-yuan Lin <tzuyuan@umich.edu>, Maani Ghaffari <maanigj@umich.edu>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   pcl_visualizer.cpp
 *  @author Tzu-yuan Lin, Maani Ghaffari 
 *  @brief  Source file for rkhs pcl visualizer
 *  @date   May 31, 2019
 **/

#include "pcl_visualizer.hpp"

namespace cvo{

pcl_visualizer::pcl_visualizer():
    init(false),
    frame_id(0),
    accum_transform(Eigen::Affine3f::Identity()),
    viewer(new pcl::visualization::PCLVisualizer ("CVO Pointcloud Visualization"))
{  
}

pcl_visualizer::~pcl_visualizer(){

}

void pcl_visualizer::add_pcd_to_viewer(string fixed_vis_pth, string moving_vis_pth, Eigen::Affine3f transform){

    // accumulate transformation
    accum_transform = accum_transform * transform.matrix();

    // load point clouds for visualization
    if(pcl::io::loadPCDFile<pcl::PointXYZRGB>(fixed_vis_pth, fixed_vis)==-1)
        PCL_ERROR("couldn't find full fixed file");
    if(pcl::io::loadPCDFile<pcl::PointXYZRGB>(moving_vis_pth, moving_vis)==-1)
        PCL_ERROR("couldn't find full moving file");

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ptr_fixed_vis(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ptr_moved_vis(new pcl::PointCloud<pcl::PointXYZRGB>());

    // initialize pointer to pcd_vis
    *ptr_fixed_vis = fixed_vis;
    *ptr_moved_vis = moved_vis;

    // if it's the first pair, add target pcd into viewer
    if(init==false){
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> fixed_rgb(ptr_fixed_vis);
        viewer->addPointCloud<pcl::PointXYZRGB> (ptr_fixed_vis,fixed_rgb, "frame0");
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "frame0");
        frame_id+=1;
        init = true;
    }

    // transform moving_vis
    string cloud_name = "frame" + to_string(frame_id);

    // transform moving pcd
    pcl::transformPointCloud(moving_vis, moved_vis, accum_transform);
    *ptr_moved_vis = moved_vis;
    
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> moved_rgb(ptr_moved_vis);
    viewer->addPointCloud<pcl::PointXYZRGB> (ptr_moved_vis,moved_rgb, cloud_name);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, cloud_name);
    frame_id+=1;
}

void pcl_visualizer::visualize(){
    while (!viewer->wasStopped()){
        viewer->spinOnce ();
    }
}

void pcl_visualizer::visualize_unaligned_pcd(){
    // visualize unaligned pointclouds
    
    pcl::visualization::PCLVisualizer::Ptr unmoved_viewer (new pcl::visualization::PCLVisualizer ("Unaligned Pointcloud"));
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ptr_fixed_vis(new pcl::PointCloud<pcl::PointXYZRGB>()); 
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ptr_moving_vis(new pcl::PointCloud<pcl::PointXYZRGB>());
    
    *ptr_fixed_vis = fixed_vis;
    *ptr_moving_vis = moving_vis;

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> fixed_rgb(ptr_fixed_vis);
    unmoved_viewer->addPointCloud<pcl::PointXYZRGB> (ptr_fixed_vis,fixed_rgb, "original_cloud");
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> moving_rgb(ptr_moving_vis);
    unmoved_viewer->addPointCloud<pcl::PointXYZRGB> (ptr_moving_vis,moving_rgb, "untransformed_cloud");
    unmoved_viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "original_cloud");
    unmoved_viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "untransformed_cloud");
    
    while (!viewer->wasStopped()&&!unmoved_viewer->wasStopped()){
        viewer->spinOnce ();
        unmoved_viewer->spinOnce ();
    }
}
}