/* ----------------------------------------------------------------------------
 * Copyright 2019, Tzu-yuan Lin <tzuyuan@umich.edu>, Maani Ghaffari <maanigj@umich.edu>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   pcl_visualizer.hpp
 *  @author Tzu-yuan Lin, Maani Ghaffari 
 *  @brief  Header file for rkhs pcl visualizer
 *  @date   May 31, 2019
 **/

#ifndef PCL_VISUALIZER_H
#define PCL_VISUALIZER_H

#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <string.h>

using namespace std;
namespace cvo{
class pcl_visualizer{

    private:
        pcl::PointCloud< pcl::PointXYZRGB > fixed_vis;  // target point cloud for visualization
        pcl::PointCloud< pcl::PointXYZRGB > moving_vis; // unmoved source point cloud for visualization
        pcl::PointCloud< pcl::PointXYZRGB > moved_vis;  // moved source point cloud for visualization

        bool init;           // initialize indicator for adding first frame into viewer
        int frame_id;
        pcl::visualization::PCLVisualizer::Ptr viewer;
        Eigen::Affine3f accum_transform; 

    public:

        // constructor and destructor
        pcl_visualizer();
        ~pcl_visualizer();

        /**
         * @brief add current aligned pcd to visualizer
         */
        void add_pcd_to_viewer(string fixed_vis_pth, string moving_vis_pth, Eigen::Affine3f transform);

        /**
         * @brief visualize accumulated aligned pcd using pcl viewer
         *        should not be called along with visualize_unaligned_pcd()
         */
        void visualize();

        /**
         * @brief visualize accumulated algined pcds and the last unaligned pcd
         *        should not be called along with visualize()
         **/
        void visualize_unaligned_pcd();
};
}
#endif  // PCL_VISUALIZER_H