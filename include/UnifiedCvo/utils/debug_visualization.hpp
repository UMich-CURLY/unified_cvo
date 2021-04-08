//#include "Pnt.h"
#pragma once

#include <vector>
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "utils/data_type.hpp"
namespace cvo {


  template <class Pnt>
  inline void save_points_as_hsv_pcd(std::string  filename, const std::vector<Pnt> & pts) {
    pcl::PointCloud<pcl::PointXYZHSV> cloud;
    //pcl::PointCloud<pcl::PointXYZ> cloud;

    cloud.width = pts.size(); 
    cloud.height = 1;
    cloud.is_dense = false;
    cloud.points.resize(pts.size());

    for (size_t i = 0; i < cloud.points.size(); i++) {
      cloud.points[i].x = pts[i].local_coarse_xyz(0);
      cloud.points[i].y = pts[i].local_coarse_xyz(1);
      cloud.points[i].z = pts[i].local_coarse_xyz(2);
      cloud.points[i].h = pts[i].rgb(0);
      cloud.points[i].s = pts[i].rgb(1);
      cloud.points[i].v = pts[i].rgb(2);
      //cloud.points[i].rgb = *reinterpret_cast<float*>(&rgb);
    }
    pcl::io::savePCDFileASCII(filename.c_str(), cloud );
  }
  
  template <class Pnt>
  inline void save_points_as_color_pcd(std::string  filename, const std::vector<Pnt> & pts) {
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    //pcl::PointCloud<pcl::PointXYZ> cloud;

    cloud.width = pts.size(); 
    cloud.height = 1;
    cloud.is_dense = false;
    cloud.points.resize(pts.size());

    for (size_t i = 0; i < cloud.points.size(); i++) {
      cloud.points[i].x = pts[i].local_coarse_xyz(0);
      cloud.points[i].y = pts[i].local_coarse_xyz(1);
      cloud.points[i].z = pts[i].local_coarse_xyz(2);

      uint8_t r = pts[i].rgb(2);
      uint8_t g = pts[i].rgb(1);
      uint8_t b = pts[i].rgb(0);
      uint32_t rgb = ((uint32_t) r << 16 |(uint32_t) g << 8  | (uint32_t) b ) ;
      cloud.points[i].rgb = *reinterpret_cast<float*>(&rgb);
    }
    pcl::io::savePCDFileASCII(filename.c_str(), cloud );
  }

  template <class Pnt>
  inline void save_points_as_gray_pcd(std::string  filename, const std::vector<Pnt> & pts) {
    pcl::PointCloud<pcl::PointXYZ> cloud;

    cloud.width = pts.size();
    cloud.height = 1;
    cloud.is_dense = false;
    cloud.points.resize(pts.size());

    for (size_t i = 0; i < cloud.points.size(); i++) {
      cloud.points[i].x = pts[i].local_coarse_xyz(0);
      cloud.points[i].y = pts[i].local_coarse_xyz(1);
      cloud.points[i].z = pts[i].local_coarse_xyz(2);
    }
    pcl::io::savePCDFileASCII(filename.c_str(), cloud );
  }

  
  inline void visualize_semantic_image(std::string name, float * image_semantics, int num_class, int w, int h) {
    cv::Mat img(h, w, CV_8UC1);


    std::cout<<"visualize_semantic_image: first point labels are ";
    for (int i = 0; i < num_class; i++) {
      std::cout<<image_semantics[i]<<", ";
    }
    std::cout<<"\n";
    
    for (int r = 0; r < h; r++ ) {
      for (int c = 0; c < w; c++){
        int label;
        float * start = image_semantics + num_class * (r * w + c);
        int max_prob = 0;
        for (int i = 0; i < num_class; i++) {
          if (*(i+start) > max_prob) {
            max_prob = *(i+start);
            label = i;
            
          }
        }
        img.at<uint8_t>(r, c) = (uint8_t) label;
      }
      
    }
    //cv::imshow("labels", img);
    //cv::waitKey(500);
    cv::imwrite(name, img);
  }

  
  inline void save_img(std::string filename,
                       float * img,
                       int num_channel,
                       int w, int h) {
    cv::Mat paint (h, w, CV_32FC(num_channel), img);
    paint.convertTo(paint, CV_8UC(num_channel));
    cv::imwrite(filename, paint);
    
  }
  
  inline void save_img(std::string filename,
                       uint8_t * img,
                       int num_channel,
                       int w, int h) {
    cv::Mat paint (h, w, CV_8UC(num_channel), img);
    cv::imwrite(filename, paint);
    
  }
  
  template <class Pnt>
  inline void save_img_with_projected_points(std::string filename,
                                             float * img_gray,
                                             int num_channel,
                                             int w, int h,
                                             const Mat33f & intrinsic, 
                                             const std::vector<Pnt> & pts,
                                             // true: write.
                                             // false: imshow
                                             bool write_or_imshow) {
    
    cv::Mat paint (h, w, CV_32FC(num_channel), img_gray);
    paint.convertTo(paint, CV_8UC(num_channel));
    if (num_channel == 1)
      cv::cvtColor(paint, paint, cv::COLOR_GRAY2BGR);
    //std::cout<<intrinsic<<std::endl;
    for (auto && p: pts) {
      auto xyz = p.local_coarse_xyz;
      Vec3f uv = intrinsic * xyz;
      uv(0) = uv(0) / uv(2);
      uv(1) = uv(1) / uv(2);

      //float rgb[3];
      //HSVtoRGB(p.rgb(0), p.rgb(1), p.rgb(2), rgb );
      //std::cout<<p.rgb(0)<<","<<p.rgb(1)<<","<<p.rgb(2)<<"\n";      
      cv::circle(paint,cv::Point2f(uv(0), uv(1)), 5.0,
                 cv::Scalar((int)p.rgb[0], (int)p.rgb[1], (int)p.rgb[2]));
      
    }
    if (write_or_imshow) {
      cv::imwrite(filename, paint);      
    } else {
      cv::imshow("projected image", paint);
      cv::waitKey(100);
    }
      
  }

}
