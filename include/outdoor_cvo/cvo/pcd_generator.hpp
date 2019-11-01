/* ----------------------------------------------------------------------------
 * Copyright 2019, Tzu-yuan Lin <tzuyuan@umich.edu>, Maani Ghaffari <maanigj@umich.edu>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   pcd_generator.hpp
 *  @author Tzu-yuan Lin, Maani Ghaffari 
 *  @brief  Header file for point cloud generator
 *  @date   July 12, 2019
 **/

#ifndef PCD_GENERATOR_H
#define PCD_GENERATOR_H

#include "data_type.h"
#include "CvoPixelSelector.h"

#include <string.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
// #include <pcl/io/pcd_io.h>
// #include <pcl/point_types.h>


using namespace std;

namespace cvo{
class pcd_generator{
    private:
        // private variables
        
        int num_want;
        int num_selected;
        int dep_thres;
        
        int w_pyr[PYR_LEVELS];  // width for each pyramid
        int h_pyr[PYR_LEVELS];

        float* map; // map for selected pixels
        int pcd_idx = 0;
    public:
        // public variables

        int dataset_seq;   // sequence of the dataset for calibration. ex. 1 for fr1

    private:
        // private functions

        /**
         * @brief make image pyramid and calculate gradient for each level
         */
        void make_pyramid(frame* ptr_fr);

        /**
         * @brief select point using PixelSelector from DSO
         */
        void select_point(frame* ptr_fr);

        /**
         * @brief visualize selected pixels in the image
         */
        void visualize_selected_pixels(frame* ptr_fr);

        /**
         * @brief retrieve points x,y,z positions from pixels
         **/
        void get_points_from_pixels(frame* ptr_fr, point_cloud* ptr_pcd);

        /**
         * @brief get features from pixels (R,G,B,dx,dy)
         **/
        void get_features(frame* ptr_fr, point_cloud* ptr_pcd);

        

    public:
        // public functions

        // constructor and destructor
        pcd_generator(int img_idx);
        ~pcd_generator();

        /**
         * @brief load image and preprocess for point selector
         */
  void load_image(const cv::Mat& rgb_img, const cv::Mat& dep_pth, MatrixXf_row semantic_labels, \
                        frame* ptr_fr);

        /**
         * @brief select points and generate point clouds
         */
        void create_pointcloud(frame* ptr_fr, point_cloud* ptr_pcd);

        /**
         * @brief write point cloud as pcd files. this function requires pcl library
         */
        // void write_pcd_to_disk(point_cloud* ptr_pcd, const string& folder);


};
}
#endif //PCD_GENERATOR_H
