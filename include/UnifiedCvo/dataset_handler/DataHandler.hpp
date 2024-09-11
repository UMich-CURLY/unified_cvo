#pragma once
#include <cstdint>
#include <iostream>
#include <string>
#include <memory>

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>
namespace cvo {

  class DatasetHandler {
  public:

    virtual int read_next_rgbd(cv::Mat & rgb_img, 
                               cv::Mat & depth_img) { return 0; }
    virtual int read_next_rgbd(cv::Mat & rgb_img, 
                               std::vector<float> & depth) { return 0; }
    virtual int read_next_rgbd(cv::Mat & rgb_img, 
                               std::vector<uint16_t> & depth) { return 0; }


    virtual int read_next_stereo(cv::Mat & left,
                                 cv::Mat & right) { return 0; }

    virtual int read_next_lidar(pcl::PointCloud<pcl::PointXYZI>::Ptr pc) { return 0; }
    virtual int read_next_lidar(pcl::PointCloud<pcl::PointXYZI>::Ptr pc,
                                std::vector<int> & semantics) { return 0; }

    
    virtual int read_next_lidar_mono(cv::Mat & image,
                                     pcl::PointCloud<pcl::PointXYZ>::Ptr pc  ) { return 0; }
    virtual ~DatasetHandler() = 0;
    
    virtual void set_start_index(int start) = 0;
    virtual int get_current_index() = 0;
    virtual int get_total_number() = 0;
    virtual void next_frame_index() = 0;
  };

  inline DatasetHandler::~DatasetHandler() {}
  
}


