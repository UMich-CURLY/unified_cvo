#ifndef LIDARPOINTSELECTOR_HPP
#define LIDARPOINTSELECTOR_HPP

#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "utils/data_type.hpp"


namespace cvo
{

class LidarPointSelector{
  public:
    LidarPointSelector(int num_want,
                        double intensity_bound, 
                        double depth_bound,
                        double distance_bound,
                        int num_beams);
    ~LidarPointSelector();
    void edge_detection(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in,
                        pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out,
                        std::vector <double> & output_depth_grad,
                        std::vector <double> & output_intenstity_grad);
    void edge_detection(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in,
                        const std::vector<int> & semantic_in,
                        pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out,
                        std::vector <double> & output_depth_grad,
                        std::vector <double> & output_intenstity_grad,
                        std::vector<int> & semantic_out);
    void loam_point_selector(pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudIn,
                            pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out,
                            std::vector <float> & edge_or_surface);
    void legoloam_point_selector(pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudIn,
                                pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out,
                                std::vector <float> & edge_or_surface);

  private:
    int get_quadrant(pcl::PointXYZI point);
    int _num_want;
    double _intensity_bound;
    double _depth_bound;
    double _distance_bound;
    int _num_beams;
};


}

#endif