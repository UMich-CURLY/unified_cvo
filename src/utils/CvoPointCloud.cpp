#include <string>
#include <fstream>
#include "utils/def_assert.hpp"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <utility>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <tbb/tbb.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "utils/CvoPointCloud.hpp"
#include "utils/StaticStereo.hpp"
#include "utils/CvoPixelSelector.hpp"
#include "utils/LidarPointSelector.hpp"
#include "utils/LidarPointType.hpp"
//#include "opencv2/xfeatures2d.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include "utils/RawImage.hpp"
#include "utils/ImageStereo.hpp"
#include "utils/ImageRGBD.hpp"
#include "utils/CvoPoint.hpp"
//#include "mapping/bkioctomap.h"
namespace cvo{

  enum PointCloudType {
    STEREO,
    RGBD,
    LIDAR
  };



  // filter out sky or too-far-away pixels
  static bool is_good_point(const Vec3f & xyz, const std::pair<int, int> & uv, int h, int w ) {
    int u = uv.first;
    int v = uv.second;

    if (std::isnan(xyz(0)) || std::isnan(xyz(1)) || std::isnan(xyz(2)) ||
        !std::isfinite(xyz(0)) || !std::isfinite(xyz(1)) || !std::isfinite(xyz(2)) 
        )
      return false;
    
    if ( u < 2 || u > w -2 || v < 100 || v > h-30 )
      return false;

    if (xyz.norm() >= 55) // 55
      return false;

    return true;
  }

  // filter away pixels that are too far away
  static bool is_good_point(const Vec3f & xyz ) {
    if (std::isnan(xyz(0)) || std::isnan(xyz(1)) || std::isnan(xyz(2))
        ||         !std::isfinite(xyz(0)) || !std::isfinite(xyz(1)) || !std::isfinite(xyz(2)) )
      return false;

    
    if (xyz.norm() > 55) // 55
      return false;

    return true;
  }

  static void copy_eigen_dynamic_matrix(const Eigen::MatrixXf * source,
                                        Eigen::MatrixXf * target
                                        ) {
    if (source->cols() == 0 || source->rows() == 0)
      return;
    target->resize(source->rows(), source->cols());
    #pragma omp parallel for
    for (int i = 0; i < source->cols(); i++)
      target->col(i) = source->col(i);
    
  }

  void write_all_to_label_pcd(const std::string name,
                              const pcl::PointCloud<pcl::PointXYZI> & pc,
                              int num_class,
                              const std::vector<int> & semantic)  {
    pcl::PointCloud<pcl::PointXYZL> pc_label;
    for (int i = 0; i < pc.size(); i++) {
      pcl::PointXYZL p;
      p.x = pc[i].x;
      p.y = pc[i].y;
      p.z = pc[i].z;
      p.label = semantic[i];
      pc_label.push_back(p);
    }
    pcl::io::savePCDFileASCII(name ,pc_label); 
    std::cout << "Finished write to label pcd" << std::endl; 
  }


  static void stereo_surface_sampling(const cv::Mat & left_gray,
                                      const std::vector<std::pair<int, int>> & dso_selected_uv,
                                      bool is_using_canny,
                                      bool is_using_uniform_rand,
                                      bool is_using_orb,
                                      int expected_points,
                                      // output
                                      std::vector<float> & edge_or_surface,
                                      std::vector<std::pair<int, int>> & final_selected_uv
                    
                                      ) {
    /*    selected_inds_map.resize(left_gray.total(), -1);
    for (auto && uv : dso_selected_uv) {
      int u = uv(0);
      int v = uv(1);
      selected_inds_map[v * left_gray.cols + u]  = 0;
      }*/
    // canny
    cv::Mat detected_edges;
    if (is_using_canny)
      cv::Canny( left_gray, detected_edges, 50, 50*3, 3 );
    int counter = 0;
    std::vector<std::pair<int, int>> tmp_uvs_canny, tmp_uvs_surface;
    if (is_using_canny) {
      for (int r = 0 ; r < left_gray.rows; r++) {
        for (int c = 0; c < left_gray.cols; c++) {
          // using Canny
          if (is_using_canny &&  detected_edges.at<uint8_t>(r, c) > 0)  {
            // selected_inds_map[r * left_gray.cols + c] = EDGE;
            tmp_uvs_canny.push_back(std::pair{c, r});
          }
        }
      }
    }
    if (is_using_uniform_rand) {
      for (int r = 0 ; r < left_gray.rows; r++) {
        for (int c = 0; c < left_gray.cols; c++) {
          
          // using uniform sampling
          if ( (!is_using_canny || detected_edges.at<uint8_t>(r, c) == 0 ) &&
               is_using_uniform_rand &&
               //r > left_gray.rows   &&
               rand() % 10 == 0)  {
            //selected_inds_map[r * left_gray.cols + c] = SURFACE;
            tmp_uvs_surface.push_back(std::pair{c, r});
        }
          
        }
        
      }
    }

    if (is_using_orb) {
      cv::Ptr< cv::ORB > orb_detector = cv::ORB::create (expected_points / 3,
                                                         /* float scaleFactor=*/1.2f,
                                                         /* int nlevels=*/8,
                                                         /* int edgeThreshold=*/31,
                                                         /*int firstLevel=*/0,
                                                         /*int WTA_K=*/2,
                                                         /*int scoreType=*/cv::ORB::HARRIS_SCORE,
                                                         /*int patchSize=*/31,
                                                         /*int fastThreshold=*/20);
      std::vector<cv::KeyPoint> keypoints; 
      orb_detector->detect ( left_gray, keypoints );
      int found_orb = keypoints.size();
      for (int i = 0; i < keypoints.size(); i++) {
        int c = (int)keypoints[i].pt.x;        
        int r = (int)keypoints[i].pt.y;
        //selected_inds_map[r * left_gray.cols + c] = 0;
        final_selected_uv.push_back(std::pair{c, r});
        edge_or_surface.push_back(1.0);
        edge_or_surface.push_back(0.0);
      
        found_orb++;
      }
    }
    
    std::cout<<"Canny size "<<tmp_uvs_canny.size()<<", surface size "<<tmp_uvs_surface.size()<<"\n";
    int total_selected_canny = tmp_uvs_canny.size();
    int total_selected_surface = tmp_uvs_surface.size();
    //expected_points -= found_orb;
    for (int i = 0; i < tmp_uvs_canny.size(); i++) {
      if (rand() % total_selected_canny < expected_points  * 1 / 4  ) {
        final_selected_uv.push_back(tmp_uvs_canny[i]);
        edge_or_surface.push_back(1.0);
        edge_or_surface.push_back(0.0);
      }
    }
    for (int i = 0; i < tmp_uvs_surface.size(); i++) {
      if (rand() % total_selected_surface < expected_points * 3 / 4 ) {
        final_selected_uv.push_back(tmp_uvs_surface[i]);
        edge_or_surface.push_back(0.0);
        edge_or_surface.push_back(1.0);
        
      }
      
    }
    

    std::cout<<" final selected uv size is "<<final_selected_uv.size()<<std::endl;
    //cv::imwrite("canny.png", detected_edges);    
    
  }

  

  
  static void select_points_from_image(const RawImage & left_image,
                                       PointCloudType pt_type,
                                       CvoPointCloud::PointSelectionMethod pt_selection_method,
                                       // result
                                       std::vector<float> & edge_or_surface,
                                       std::vector<std::pair<int, int>> & output_uv

                                       )  {
    cv::Mat left_gray;
    if (left_image.channels() == 1)
      left_gray = left_image.image();
    else {
      cv::cvtColor(left_image.image(), left_gray, cv::COLOR_BGR2GRAY);
    }
    /*****************************************/
    if (pt_selection_method == CvoPointCloud::CV_FAST) {
      // using opencv FAST
      //-- Step 1: Detect the keypoints using SURF Detector
      int minHessian = 400;
      std::vector<cv::KeyPoint> keypoints;    
      cv::FAST(left_gray,keypoints, 5,false);

      //// for geometric
      int thresh, num_want, num_min, break_thresh;
      if (pt_type == RGBD) {
        thresh = 9; num_want = 15000; num_min = 12000; break_thresh = 13;
      } else if (pt_type == STEREO) {
        thresh = 4; num_want = 24000; num_min = 15000; break_thresh = 50;
        if (left_image.num_classes() > 0)
          num_want = 28000;
      }
        
      while (keypoints.size() > num_want)  {
        keypoints.clear();
        thresh++;
        cv::FAST(left_gray,keypoints, thresh,false);
        if (thresh == break_thresh) break;
      }
      while (keypoints.size() < num_min ) {
        keypoints.clear();
        thresh--;
        cv::FAST(left_gray,keypoints, thresh,false);
        if (thresh== 0) break;

      }
    
      //detector->detect( left_gray, keypoints, cv::Mat() );
      for (auto && kp: keypoints) {
        std::pair<int, int> xy{ (int)kp.pt.x, (int)kp.pt.y};
        output_uv.push_back(xy);
        edge_or_surface.push_back(1);
        edge_or_surface.push_back(0);
      }
      bool debug_plot = false;
      if (debug_plot) {
        std::cout<<"Number of selected points is "<<output_uv.size()<<"\n";
        cv::Mat heatmap(left_image.rows(), left_image.cols(), CV_32FC1, cv::Scalar(0) );
        int w = heatmap.cols;
        int h = heatmap.rows;
        for (int i = 0; i < output_uv.size(); i++) 
          cv::circle(heatmap, cv::Point( output_uv[i].first, output_uv[i].second ), 1, cv::Scalar(255, 0 ,0), 1);
        cv::imwrite("FAST_selected_pixels.png", heatmap);
      }
    } 
    /*****************************************/
    // using DSO semi dense point selector
    else if (pt_selection_method == CvoPointCloud::DSO_EDGES) {
      int expected_points = 10000;
      dso_select_pixels(left_image,
                        expected_points,
                        output_uv);
      edge_or_surface.resize(output_uv.size() * 2);
      //std::cout<<"Just resized to "<<edge_or_surface.size()<<"\n";
      for (int i = 0; i < output_uv.size(); i++) {
        edge_or_surface[i*2] = 0.9;
        edge_or_surface[i*2 +1]=0.1;
        //std::cout<<i<<", ";
      }
      //std::cout<<"\n"<<"finish dso sampling\n";
      
    }
    //******************************************/
    // using canny or random point selection
    else if (pt_selection_method == CvoPointCloud::CANNY_EDGES) {
      //std::vector<bool> selected_inds_map;
      std::vector<std::pair<int, int>> final_selected_uv;
      int expected_points = 10000;
      stereo_surface_sampling(left_gray, output_uv, true, true, true, expected_points,
                              edge_or_surface, output_uv);


      
      
    }
    /* edge only */
    else if (pt_selection_method == CvoPointCloud::EDGES_ONLY) {
      //std::vector<bool> selected_inds_map;
      std::vector<std::pair<int, int>> final_selected_uv;
      int expected_points = 10000;
      stereo_surface_sampling(left_gray, output_uv, true, false, false, expected_points,
                              edge_or_surface, output_uv);
      
    }
    
    /********************************************/
    // using full point cloud
    else if (pt_selection_method == CvoPointCloud::FULL) {

      output_uv.clear();
      cv::Mat detected_edges;
          
      cv::Canny( left_gray, detected_edges, 50, 50*3, 3 );
      
      for (int h = 0; h < left_image.rows(); h++){

        for (int w = 0; w < left_image.cols(); w++){
          std::pair<int,int> uv {w , h};
          output_uv.push_back(uv);
          if (detected_edges.at<uint8_t>(h, w)  > 0 ) {
            edge_or_surface.push_back(0.95);
            edge_or_surface.push_back(0.05);
          }      else {
            edge_or_surface.push_back(0.05);
            edge_or_surface.push_back(0.95);
            
          }
          
        }
      }
    } else {
      std::cerr<<"This point selection method is not implemented.\n";
      return;
    }
    return;
  }

  /*  
  CvoPointCloud::CvoPointCloud(const RawImage & rgb_raw_image,
                               const std::vector<uint16_t> & depth_image,
                               const Calibration &calib,
                               PointSelectionMethod pt_selection_method){
 
    std::vector<std::pair<int, int>> output_uv;
    std::vector<float> geometry;
    select_points_from_image(rgb_raw_image, RGBD, pt_selection_method,
                             geometry,
                             output_uv);
      
    std::vector<int> good_point_ind;
    int h = rgb_raw_image.rows();
    int w = rgb_raw_image.cols();
    //positions_.resize(output_uv.size());
    Mat33f intrinsic = calib.intrinsic();

    for (int i = 0; i < output_uv.size(); i++) {
      auto uv = output_uv[i];
      int u = uv(0);
      int v = uv(1);
      Vec3f xyz;

      //uint16_t dep = depth_image.at<uint16_t>(cv::Point(u, v));
      uint16_t dep = depth_image[v * w + u];
        
      if(dep!=0 && !isnan(dep)){

        // construct depth
        xyz(2) = dep/calib.scaling_factor();
            
        // construct x and y
        xyz(0) = (u-intrinsic(0,2)) * xyz(2) / intrinsic(0,0);
        xyz(1) = (v-intrinsic(1,2)) * xyz(2) / intrinsic(1,1);
            
        // add point to pcd
        good_point_ind.push_back(i);
        //positions_[i] = xyz;
        positions_.push_back(xyz);
        geometric_types_.push_back(geometry[i*2]);
        geometric_types_.push_back(geometry[i*2+1]);
      }
    }

    num_points_ = good_point_ind.size();
    num_classes_ = rgb_raw_image.num_classes();
    feature_dimensions_ = rgb_raw_image.channels() + 2;
    features_.resize(num_points_, feature_dimensions_);
    for (int i = 0; i < num_points_ ; i++) {
      int u = output_uv[good_point_ind[i]](0);
      int v = output_uv[good_point_ind[i]](1);
      if (rgb_raw_image.channels() == 3) {
        cv::Vec3b avg_pixel = rgb_raw_image.image().at<cv::Vec3b>(v,u);
        float gradient_0 = rgb_raw_image.gradient()[v * w + u];
        float gradient_1 = rgb_raw_image.gradient()[v * w + u + 1];
        features_(i,0) = ((float)(avg_pixel [0]) )/255.0;
        features_(i,1) = ((float)(avg_pixel[1]) )/255.0;
        features_(i,2) = ((float)(avg_pixel[2]) )/255.0;
        features_(i,3) = gradient_0/ 500.0 + 0.5;
        features_(i,4) = gradient_1/ 500.0 + 0.5;
      } else if (rgb_raw_image.channels() == 1) {
        uint8_t avg_pixel = rgb_raw_image.image().at<uint8_t>(v,u);
        float gradient_0 = rgb_raw_image.gradient()[v * w + u];
        float gradient_1 = rgb_raw_image.gradient()[v * w + u + 1];
        features_(i,0) = ((float)(avg_pixel) )/255.0;
        features_(i,1) = gradient_0/ 500.0 + 0.5;
        features_(i,2) = gradient_1/ 500.0 + 0.5;          
      } else {
        std::cerr<<"CvoPointCloud: channel unknown\n";
      }
    }
 
  }
  */

  template <typename DepthType>
  CvoPointCloud::CvoPointCloud(const ImageRGBD<DepthType> & raw_image,
                               const Calibration &calib,
                               PointSelectionMethod pt_selection_method){

    const cv::Mat & rgb_raw_image = raw_image.image();
    const std::vector<DepthType> & depth_image = raw_image.depth_image();
 
    std::vector<std::pair<int, int>> output_uv;
    std::vector<float> geometry;
    select_points_from_image(rgb_raw_image, RGBD, pt_selection_method,
                             geometry,
                             output_uv);
      
    std::vector<int> good_point_ind;
    int h = raw_image.rows();
    int w = raw_image.cols();
    //positions_.resize(output_uv.size());
    Mat33f intrinsic = calib.intrinsic();

    for (int i = 0; i < output_uv.size(); i++) {
      CvoPoint point;
      auto uv = output_uv[i];
      int u = uv.first;
      int v = uv.second;
//      Vec3f xyz;
 
      //uint16_t dep = depth_image.at<uint16_t>(cv::Point(u, v));
      DepthType dep = depth_image[v * w + u];
      //std::cout<<__func__<<": dep["<<v<<", "<<u<<"] = "<<dep<<std::endl;
        
      if(dep!=0 && !isnan(dep)){

        // construct depth
//        xyz(2) = dep/calib.scaling_factor();
        point.z = dep/calib.scaling_factor();

        //if (xyz(2) > 15.0)
        //  continue;
        
        // construct x and y
//        xyz(0) = (u-intrinsic(0,2)) * xyz(2) / intrinsic(0,0);
        point.x = (u-intrinsic(0,2)) * point.z / intrinsic(0,0);
//        xyz(1) = (v-intrinsic(1,2)) * xyz(2) / intrinsic(1,1);
        point.y = (v-intrinsic(1,2)) * point.z / intrinsic(1,1);
        
        // check for labels
//        if (raw_image.num_classes()) {
//          auto labels = Eigen::Map<const VecXf_row>((raw_image.semantic_image().data()+ (v * w + u)*raw_image.num_classes()), raw_image.num_classes() );
//          int max_class = 0;
//          labels.maxCoeff(&max_class);
//          //if(max_class == 10)// exclude unlabeled points
//          //  continue;
//        }

        // add point to pcd
        good_point_ind.push_back(i);
        //positions_[i] = xyz;
//        positions_.push_back(xyz);
//        geometric_types_.push_back(geometry[i*2]);
        point.geometric_type[0] = geometry[i*2];
//        geometric_types_.push_back(geometry[i*2+1]);
        point.geometric_type[1] = geometry[i*2+1];

        if (raw_image.channels() == 3) {
          cv::Vec3b avg_pixel = raw_image.image(). template at<cv::Vec3b>(v,u);
          float gradient_0 = raw_image.gradient()[v * w + u];
          float gradient_1 = raw_image.gradient()[v * w + u + 1];
          point.features[0] = ((float)(avg_pixel [0]) )/255.0;
          point.features[1] = ((float)(avg_pixel[1]) )/255.0;
          point.features[2] = ((float)(avg_pixel[2]) )/255.0;
          point.features[3] = gradient_0/ 500.0 + 0.5;
          point.features[4] = gradient_1/ 500.0 + 0.5;
        } else if (raw_image.channels() == 1) {
          uint8_t avg_pixel = raw_image.image(). template at<uint8_t>(v,u);
          float gradient_0 = raw_image.gradient()[v * w + u];
          float gradient_1 = raw_image.gradient()[v * w + u + 1];
          point.features[0] = ((float)(avg_pixel) )/255.0;
          point.features[1]  = gradient_0/ 500.0 + 0.5;
          point.features[2] = gradient_1/ 500.0 + 0.5;
        } else {
          std::cerr<<"CvoPointCloud: channel unknown\n";
        }
        points_.push_back(point);
      }
    }

    num_points_ = good_point_ind.size();
    num_classes_ = raw_image.num_classes();
    num_geometric_types_ = 2;    
    if (num_classes_ )
      labels_.resize(num_points_, num_classes_);
    feature_dimensions_ = raw_image.channels() + 2;
//    for (int i = 0; i < num_points_ ; i++) {
//      int u = output_uv[good_point_ind[i]].first;
//      int v = output_uv[good_point_ind[i]].second;
//      if (raw_image.channels() == 3) {
//        cv::Vec3b avg_pixel = raw_image.image(). template at<cv::Vec3b>(v,u);
//        float gradient_0 = raw_image.gradient()[v * w + u];
//        float gradient_1 = raw_image.gradient()[v * w + u + 1];
//        features_(i,0) = ((float)(avg_pixel [0]) )/255.0;
//        features_(i,1) = ((float)(avg_pixel[1]) )/255.0;
//        features_(i,2) = ((float)(avg_pixel[2]) )/255.0;
//        features_(i,3) = gradient_0/ 500.0 + 0.5;
//        features_(i,4) = gradient_1/ 500.0 + 0.5;
//      } else if (raw_image.channels() == 1) {
//        uint8_t avg_pixel = raw_image.image(). template at<uint8_t>(v,u);
//        float gradient_0 = raw_image.gradient()[v * w + u];
//        float gradient_1 = raw_image.gradient()[v * w + u + 1];
//        features_(i,0) = ((float)(avg_pixel) )/255.0;
//        features_(i,1) = gradient_0/ 500.0 + 0.5;
//        features_(i,2) = gradient_1/ 500.0 + 0.5;
//      } else {
//        std::cerr<<"CvoPointCloud: channel unknown\n";
//      }
//      if (num_classes_) {
//        labels_.row(i) = Eigen::Map<const VecXf_row>((raw_image.semantic_image().data()+ (v * w + u)*num_classes_), num_classes_);
//        int max_class = 0;
//        labels_.row(i).maxCoeff(&max_class);
//      }
//    }
 
  }


  template 
  CvoPointCloud::CvoPointCloud(const ImageRGBD<float> & raw_image,
                                      const Calibration &calib,
                                      PointSelectionMethod pt_selection_method);
  
  template 
  CvoPointCloud::CvoPointCloud(const ImageRGBD<uint16_t> & raw_image,
                                         const Calibration &calib,
                                         PointSelectionMethod pt_selection_method);
  

  

  template <>
  CvoPointCloud::CvoPointCloud(const pcl::PointCloud<pcl::PointXYZRGB> & pc) {
    num_points_ = pc.size();
    num_classes_ = 0;
    feature_dimensions_ = 3;
    num_geometric_types_ = 2;

    positions_.resize(pc.size());
    features_.resize(num_points_, feature_dimensions_);
    geometric_types_.resize(num_points_ * num_geometric_types_);
    for (int i = 0; i < num_points_; i++) {
      auto & p = (pc)[i];
      cvo::CvoPoint point(p.x, p.y, p.z);
      point.r = (int)p.r;
      point.g = (int)p.g;
      point.b = (int)p.b;
      point.features[0] = ((float)(int)p.r) / 255.0;
      point.features[1] = ((float)(int)p.g) / 255.0;
      point.features[2] = ((float)(int)p.b) / 255.0;
      point.geometric_type[0] = 0;
      point.geometric_type[1] = 1;
      //features_(i, 3) = 0;
      //features_(i, 4) = 0;
      points_.push_back(point);
    }
    
  }

  

  template <>
  CvoPointCloud::CvoPointCloud(const pcl::PointCloud<pcl::PointXYZRGB> & pc,
                               GeometryType g_type) {
    num_points_ = pc.size();
    num_classes_ = 0;
    num_geometric_types_ = 2;
    feature_dimensions_ = 5;

    positions_.resize(pc.size());
    features_.resize(num_points_, feature_dimensions_);
    geometric_types_.resize(num_points_ * 2);
    for (int i = 0; i < num_points_; i++) {
      auto & p = (pc)[i];
      cvo::CvoPoint point(p.x, p.y, p.z);
      point.r = (int)p.r;
      point.g = (int)p.g;
      point.b = (int)p.b;
      point.features[2] = ((float)(int)p.r) / 255.0;
      point.features[1] = ((float)(int)p.g) / 255.0;
      point.features[0] = ((float)(int)p.b) / 255.0;
      point.features[3] = 0;
      point.features[4] = 0;

      if (g_type == GeometryType::SURFACE) {
        point.geometric_type[0] = 0;
        point.geometric_type[1] = 1;
      } else {
        point.geometric_type[0] = 1;
        point.geometric_type[1] = 0;
        
      }
      points_.push_back(point);
    }
    
  }


  template <>
  CvoPointCloud::CvoPointCloud(const pcl::PointCloud<pcl::PointXYZ> & pc) {
    num_points_ = pc.size();
    num_classes_ = 0;
    feature_dimensions_ = 0;
    num_geometric_types_ = 2;

    positions_.resize(pc.size());
    //features_.resize(num_points_, feature_dimensions_);
    geometric_types_.resize(num_points_ * 2);
    for (int i = 0; i < pc.size(); i++) {
      auto & p = (pc)[i];
      cvo::CvoPoint point(p.x, p.y, p.z);
      point.geometric_type[0] = 1;
      point.geometric_type[1] = 0;
      points_.push_back(point);
    }
    
  }

  void CvoPointCloud::erase(size_t index) {
    if (index > this->size())
      return;

    positions_[index] = positions_.back();
    positions_.pop_back();

    features_.row(index) = features_.row(num_points_-1).eval();
    features_.conservativeResize(num_points_-1, Eigen::NoChange);

    labels_.row(index) = labels_.row(num_points_-1).eval();
    labels_.conservativeResize(num_points_-1, Eigen::NoChange);

    geometric_types_[index * num_geometric_types_] = geometric_types_[num_geometric_types_ * num_points_ -2];
    geometric_types_[index * num_geometric_types_ + 1] = geometric_types_[num_geometric_types_ * num_points_ - 1];
    geometric_types_.pop_back();
    geometric_types_.pop_back();
      
    num_points_--;
  }

  template <>
  CvoPointCloud::CvoPointCloud(const pcl::PointCloud<pcl::PointNormal> & pc) {
    num_points_ = pc.size();
    num_classes_ = 0;
    feature_dimensions_ = 0;
    num_geometric_types_ = 2;

    positions_.resize(pc.size());
    //features_.resize(num_points_, feature_dimensions_);
    geometric_types_.resize(num_points_ * 2);
    for (int i = 0; i < pc.size(); i++) {
      auto & p = (pc)[i];
      cvo::CvoPoint point(p.x, p.y, p.z);
      point.geometric_type[0] = 1;
      point.geometric_type[1] = 0;
      points_.push_back(point);
    }
    
  }
  

  
  /*
  template <>
  CvoPointCloud::CvoPointCloud<CvoPoint>(const pcl::PointCloud<CvoPoint> & pc) {
    num_points_ = pc.size();
    num_classes_ = 0;
    feature_dimensions_ = ;

    positions_.resize(pc.size());
    features_.resize(num_points_, feature_dimensions_);
    for (int i = 0; i < num_points_; i++) {
      Eigen::Vector3f xyz;
      auto & p = (pc)[i];
      xyz << p.x, p.y, p.z;
      positions_[i] = xyz;

      features_(i,0) = ((float)(int)p.r) / 255.0;
      features_(i,1) = ((float)(int)p.g) / 255.0;
      features_(i,2) = ((float)(int)p.b) / 255.0;
      
    }
    
    }*/

  
  //template CvoPointCloud::CvoPointCloud<pcl::PointXYZRGB>(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc);
  CvoPointCloud::CvoPointCloud(const ImageStereo & raw_image,
                               const Calibration &calib,
                               PointSelectionMethod pt_selection_method) {

    const cv::Mat & left_image = raw_image.image();
    const std::vector<float> & left_disparity = raw_image.disparity();
    
    std::vector<std::pair<int, int>> output_uv;
    std::vector<float> geometry;
    select_points_from_image(left_image, STEREO, pt_selection_method,
                             geometry,
                             output_uv);    
    auto & pre_depth_selected_ind = output_uv;
    std::vector<int> good_point_ind;
    int h = raw_image.rows();
    int w = raw_image.cols();
    //cv::Mat depth_map(h, w, CV_32F, cv::Scalar(0));
    //bool is_recording_depth_map = true;
    //static unsigned int depth_map_counter = 0;
    //positions_.resize(pre_depth_selected_ind.size());
    for (int i = 0; i < pre_depth_selected_ind.size(); i++) {
      auto uv = pre_depth_selected_ind[i];
      Vec3f xyz;

      StaticStereo::TraceStatus trace_status = StaticStereo::pt_depth_from_disparity(raw_image,
                                                                                     left_disparity,
                                                                                     calib,
                                                                                     uv,
                                                                                     xyz );
      if (trace_status == StaticStereo::TraceStatus::GOOD && 
          is_good_point (xyz, uv, h, w) 
          //is_good_point(xyz)
          ) {
        int u = uv.first;
        int v = uv.second;
        if(raw_image.num_classes() ){
          auto labels = Eigen::Map<const VecXf_row>((raw_image.semantic_image().data()+ (v * w + u)*raw_image.num_classes()), raw_image.num_classes() );
          int max_class = 0;
          labels.maxCoeff(&max_class);
          if( max_class == 10)
            // exclude unlabeled points
            continue;
        }

        good_point_ind.push_back(i);
        //good_point_xyz.push_back(xyz);
        positions_.push_back(xyz);
        //positions_[i] = xyz;
        //std::cout<<"xyz: "<<xyz.transpose()<<std::endl;
        geometric_types_.push_back(geometry[i*2]);
        geometric_types_.push_back(geometry[i*2+1]);
      }
    }
     
    // start to fill in class members
    num_points_ = good_point_ind.size();
    num_classes_ = raw_image.num_classes();
    num_geometric_types_ = 2;
    if (num_classes_ )
      labels_.resize(num_points_, num_classes_);
    feature_dimensions_ = raw_image.channels() + 2;
    features_.resize(num_points_, feature_dimensions_);
    for (int i = 0; i < num_points_ ; i++) {
      int u = pre_depth_selected_ind[good_point_ind[i]].first;
      int v = pre_depth_selected_ind[good_point_ind[i]].second;
      if (raw_image.channels() == 3) {
        cv::Vec3b avg_pixel = raw_image.image().at<cv::Vec3b>(v,u);
        float gradient_0 = raw_image.gradient()[v * w + u];
        float gradient_1 = raw_image.gradient()[v * w + u + 1];
        features_(i,0) = ((float)(avg_pixel [0]) )/255.0;
        features_(i,1) = ((float)(avg_pixel[1]) )/255.0;
        features_(i,2) = ((float)(avg_pixel[2]) )/255.0;
        features_(i,3) = gradient_0/ 500.0 + 0.5;
        features_(i,4) = gradient_1/ 500.0 + 0.5;
      } else if (raw_image.channels() == 1) {
        uint8_t avg_pixel = raw_image.image().at<uint8_t>(v,u);
        float gradient_0 = raw_image.gradient()[v * w + u];
        float gradient_1 = raw_image.gradient()[v * w + u + 1];
        features_(i,0) = ((float)(avg_pixel) )/255.0;
        features_(i,1) = gradient_0/ 500.0 + 0.5;
        features_(i,2) = gradient_1/ 500.0 + 0.5;          
      } else {
        std::cerr<<"CvoPointCloud: channel unknown\n";
      }

      if (num_classes_) {
        labels_.row(i) = Eigen::Map<const VecXf_row>((raw_image.semantic_image().data()+ (v * w + u)*num_classes_), num_classes_);
        int max_class = 0;
        labels_.row(i).maxCoeff(&max_class);
      }

    }

    
  }
  


  /*
  CvoPointCloud::CvoPointCloud(const RawImage & left_image,
                               const cv::Mat & right_image,
                               const Calibration & calib,
                               PointSelectionMethod pt_selection_method)   {

    cv::Mat  left_gray, right_gray;
    if (left_image.channels() == 3)
      cv::cvtColor(left_image.image(), left_gray, cv::COLOR_BGR2GRAY);
    else
      left_gray = left_image.image();

    if (right_image.channels() == 3)
      cv::cvtColor(right_image, right_gray, cv::COLOR_BGR2GRAY);
    else
      right_gray = right_image;
    std::vector<float> left_disparity;
    StaticStereo::disparity(left_gray, right_gray, left_disparity);
    
    std::vector<std::pair<int, int>> output_uv;
    std::vector<float> geometry;
    select_points_from_image(left_image, STEREO, pt_selection_method,
                             geometry,
                             output_uv);    
    auto & pre_depth_selected_ind = output_uv;
    std::vector<int> good_point_ind;
    int h = left_image.rows();
    int w = left_image.cols();
    //cv::Mat depth_map(h, w, CV_32F, cv::Scalar(0));
    //bool is_recording_depth_map = true;
    //static unsigned int depth_map_counter = 0;
    //positions_.resize(pre_depth_selected_ind.size());
    for (int i = 0; i < pre_depth_selected_ind.size(); i++) {
      auto uv = pre_depth_selected_ind[i];
      Vec3f xyz;

      StaticStereo::TraceStatus trace_status = StaticStereo::pt_depth_from_disparity(left_image,
                                                                                     left_disparity,
                                                                                     calib,
                                                                                     uv,
                                                                                     xyz );
      if (trace_status == StaticStereo::TraceStatus::GOOD && 
          is_good_point (xyz, uv, h, w) 
          //is_good_point(xyz)
          ) {
        int u = uv(0);
        int v = uv(1);
        if(left_image.num_classes() ){
          auto labels = Eigen::Map<const VecXf_row>((left_image.semantic_image().data()+ (v * w + u)*left_image.num_classes()), left_image.num_classes() );
          int max_class = 0;
          labels.maxCoeff(&max_class);
          if( max_class == 10)
            // exclude unlabeled points
            continue;
        }

        good_point_ind.push_back(i);
        //good_point_xyz.push_back(xyz);
        positions_.push_back(xyz);
        //positions_[i] = xyz;
        //std::cout<<"xyz: "<<xyz.transpose()<<std::endl;
        geometric_types_.push_back(geometry[i*2]);
        geometric_types_.push_back(geometry[i*2+1]);
      }
    }
     
    // start to fill in class members
    num_points_ = good_point_ind.size();
    num_classes_ = left_image.num_classes();
    if (num_classes_ )
      labels_.resize(num_points_, num_classes_);
    feature_dimensions_ = left_image.channels() + 2;
    features_.resize(num_points_, feature_dimensions_);
    for (int i = 0; i < num_points_ ; i++) {
      int u = pre_depth_selected_ind[good_point_ind[i]](0);
      int v = pre_depth_selected_ind[good_point_ind[i]](1);
      if (left_image.channels() == 3) {
        cv::Vec3b avg_pixel = left_image.image().at<cv::Vec3b>(v,u);
        float gradient_0 = left_image.gradient()[v * w + u];
        float gradient_1 = left_image.gradient()[v * w + u + 1];
        features_(i,0) = ((float)(avg_pixel [0]) )/255.0;
        features_(i,1) = ((float)(avg_pixel[1]) )/255.0;
        features_(i,2) = ((float)(avg_pixel[2]) )/255.0;
        features_(i,3) = gradient_0/ 500.0 + 0.5;
        features_(i,4) = gradient_1/ 500.0 + 0.5;
      } else if (left_image.channels() == 1) {
        uint8_t avg_pixel = left_image.image().at<uint8_t>(v,u);
        float gradient_0 = left_image.gradient()[v * w + u];
        float gradient_1 = left_image.gradient()[v * w + u + 1];
        features_(i,0) = ((float)(avg_pixel) )/255.0;
        features_(i,1) = gradient_0/ 500.0 + 0.5;
        features_(i,2) = gradient_1/ 500.0 + 0.5;          
      } else {
        std::cerr<<"CvoPointCloud: channel unknown\n";
      }

      if (num_classes_) {
        labels_.row(i) = Eigen::Map<const VecXf_row>((left_image.semantic_image().data()+ (v * w + u)*num_classes_), num_classes_);
        int max_class = 0;
        labels_.row(i).maxCoeff(&max_class);
      }

    }
  }
  */
  CvoPointCloud::CvoPointCloud(const CvoPointCloud & input) {
    num_points_ = input.num_points_;
    num_classes_ = input.num_classes_;
    feature_dimensions_ = input.feature_dimensions_;
//    num_geometric_types_ = input.num_geometric_types_;
//    features_.resize(num_points_, feature_dimensions_);
//    #pragma omp parallel for
//    for (int i = 0; i < feature_dimensions_; i++)
//      features_.col(i) = input.features_.col(i);
//
//    positions_ = input.positions_;
//    //features_ = input.features_;
//    labels_ = input.labels_;
//    geometric_types_ = input.geometric_types_;
    points_ = input.points_;
  }

  CvoPointCloud & CvoPointCloud::operator=(const CvoPointCloud& input) {
    if (&input == this) return *this;
    num_points_ = input.num_points_;
    num_classes_ = input.num_classes_;
    feature_dimensions_ = input.feature_dimensions_;
//    num_geometric_types_ = input.num_geometric_types_;
//    positions_ = input.positions_;
//    features_.resize(num_points_, feature_dimensions_);
//    #pragma omp parallel for
//    for (int i = 0; i < feature_dimensions_; i++)
//      features_.col(i) = input.features_.col(i);
//
//    //features_ = input.features_;
//    labels_ = input.labels_;
//    geometric_types_ = input.geometric_types_;
    points_ = input.points_;
    return *this;
  }

  CvoPointCloud & CvoPointCloud::operator+=(const CvoPointCloud & to_add) {
    assert (this->feature_dimensions_ == to_add.feature_dimensions_
            && this->num_classes_ == to_add.num_classes_ ) ;
      //std::cout<<"Warning: adding cvo pointcloud of different classes or features\n";
      //return *this;
    

    //CvoPointCloud new_points;
    //new_points.positions_.resize(0);
    //new_points.positions_.insert(new_points.positions_.end(),  this->positions_.begin(), this->positions_.end());
    //new_points.positions_.insert(new_points.positions_.end(),  to_add.positions_.begin(), to_add.positions_.end());
//    this->positions_.insert(this->positions_.end(),  to_add.positions_.begin(), to_add.positions_.end());
    this->points_.insert(this->points_.end(),  to_add.points_.begin(), to_add.points_.end());
//    if (this->feature_dimensions_) {
//      this->features_.conservativeResize(this->num_points_ + to_add.num_points_, this->feature_dimensions_);
//      //new_points.features_.resize(this->num_points_ + to_add.num_points_,
//      //                            this->feature_dimensions_);
//      //new_points.features_.block(0, 0, this->num_points_, this->feature_dimensions_) = this->features_;
//      this->features_.block(this->num_points_, 0, to_add.num_points_, this->feature_dimensions_) = to_add.features_;
//    }
//
//    if (this->num_classes_) {
//      //new_points.labels_.resize(this->num_points_ + to_add.num_points_,
//      //                          this->num_classes_);
//      this->labels_.conservativeResize( this->num_points_ + to_add.num_points_, this->num_classes_);
//      //new_points.labels_.block(0, 0, this->num_points_, this->num_classes_) = this->labels_;
//      this->labels_.block(this->num_points_, 0, to_add.num_points_, this->num_classes_) = to_add.labels_;
//    }
    
    //new_points.geometric_types_.resize(0);
    //new_points.geometric_types_.insert(new_points.geometric_types_.end(),
    //                                   this->geometric_types_.begin(),this->geometric_types_.end() );
//    this->geometric_types_.insert(this->geometric_types_.end(),
//                                  to_add.geometric_types_.begin(),to_add.geometric_types_.end() );

    this->num_points_ = this->num_points_ + to_add.num_points_;
    //this>feature_dimensions_ = this->feature_dimensions_;
    // new_points.num_classes_ = this->num_classes_;

    return *this;
  }

  CvoPointCloud operator+(CvoPointCloud lhs,        // passing lhs by value helps optimize chained a+b+c
                                 const CvoPointCloud& rhs) // otherwise, both parameters may be const references
  {
    lhs += rhs; // reuse compound assignment
    return lhs; // return the result by value (uses move constructor)
  }

  CvoPointCloud::CvoPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pc, int target_num_points, int beam_num, PointSelectionMethod pt_selection_method) {
    int expected_points = target_num_points;
    double intensity_bound = 0.4;
    double depth_bound = 4.0;
    double distance_bound = 40.0;
    std::vector <int> selected_indexes;
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out (new pcl::PointCloud<pcl::PointXYZI>);    

    std::vector <double> output_depth_grad;
    std::vector <double> output_intenstity_grad;

    if (pt_selection_method == LOAM) {
      std::vector <float> edge_or_surface;
      LidarPointSelector lps(expected_points, intensity_bound, depth_bound, distance_bound, beam_num);

      // running edge detection + lego loam point selection
      pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_edge (new pcl::PointCloud<pcl::PointXYZI>);
      pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_surface (new pcl::PointCloud<pcl::PointXYZI>);
      lps.edge_detection(pc, pc_out_edge, output_depth_grad, output_intenstity_grad, selected_indexes);
      lps.legoloam_point_selector(pc, pc_out_surface, edge_or_surface, selected_indexes);    
      //*pc_out += *pc_out_edge;
      //*pc_out += *pc_out_surface;
      //
      num_points_ = selected_indexes.size();
      std::cout<<"pc_out size is "<<pc_out->size()<<std::endl;
      std::cout << "\nList of selected lego indexes " << pc_out_surface->size()<< std::endl;
      for(int i=0; i<10; i++){
        std::cout << selected_indexes[i] << " ";
      }
      std::cout<<std::flush;
      assert(num_points_ == selected_indexes.size());

    } else if (pt_selection_method == RANDOM) {

      random_surface_with_edges(pc, expected_points, intensity_bound, depth_bound, distance_bound, beam_num,
                                output_depth_grad, output_intenstity_grad, selected_indexes);
      num_points_ = selected_indexes.size();
    } else {
      std::cerr<<" This point selection method is not implemented\n";
      return;
    }


    // fill in class members
    num_classes_ = 0;
    num_geometric_types_ = 2;
    
    // features_ = Eigen::MatrixXf::Zero(num_points_, 1);
    feature_dimensions_ = 1;
    features_.resize(num_points_, feature_dimensions_);
    //normals_.resize(num_points_,3);
    //types_.resize(num_points_, 2);
    positions_.resize(num_points_);


    
    for (int i = 0; i < num_points_ ; i++) {
      Vec3f xyz;
      int idx = selected_indexes[i];
      xyz << pc->points[idx].x, pc->points[idx].y, pc->points[idx].z;
      positions_[i] = xyz;
      features_(i, 0) = pc->points[idx].intensity;

      // TODO: change geometric_types
      geometric_types_.push_back(1.0);
      geometric_types_.push_back(0.0);
    }

    //#if  defined(IS_USING_COVARIANCE)  && defined(__CUDACC__)
    //std::cout<<"compute covariance\n";
    //compute_covariance(*pc, selected_indexes);

    
    std::cout<<"Construct Cvo PointCloud, num of points is "<<num_points_<<" from "<<pc->size()<<" input points "<<std::endl;    
    //write_to_intensity_pcd("kitti_lidar.pcd");
  }

  CvoPointCloud::CvoPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pc, const std::vector<int> & semantic,
                               int num_classes, int target_num_points, int beam_num,
                               PointSelectionMethod pt_selection_method) {

    //write_all_to_label_pcd("kitti_semantic_lidar_pre.pcd", *pc, num_classes, semantic);

    int expected_points = target_num_points;
    double intensity_bound = 0.4;
    double depth_bound = 4.0;
    std::vector<int> selected_indexes;
    double distance_bound = 75.0;

    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out (new pcl::PointCloud<pcl::PointXYZI>);
    std::vector <double> output_depth_grad;
    std::vector <double> output_intenstity_grad;
    std::vector <int> semantic_out;

    std::cout<<"construct semantic lidar CvoPointCloud...\n";

    if (pt_selection_method == LOAM) {
      std::cout<<"using loam, not using normals"<<std::endl;
      std::vector <float> edge_or_surface;
      LidarPointSelector lps(expected_points, intensity_bound, depth_bound, distance_bound, beam_num);

      // running edge detection + lego loam point selection
      // edge detection
      pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_edge (new pcl::PointCloud<pcl::PointXYZI>);
      lps.edge_detection(pc, semantic, pc_out_edge, output_depth_grad, output_intenstity_grad, selected_indexes, semantic_out);  
      *pc_out += *pc_out_edge;
      // lego loam surface
      pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_surface (new pcl::PointCloud<pcl::PointXYZI>); 
      lps.legoloam_point_selector(pc, semantic, pc_out_surface, edge_or_surface, selected_indexes, semantic_out);
      std::cout<<"semantic out size = "<<semantic_out.size()<<std::endl;
      *pc_out += *pc_out_surface;
      num_points_ = pc_out->size();
      assert(num_points_ == selected_indexes.size());
      assert(num_points_ == semantic_out.size());


      //pcl::io::savePCDFileASCII("pc_out_edge.pcd", *pc_out_edge);
      // pcl::io::savePCDFileASCII("pc_out_legoloam.pcd", *pc_out_surface);
    
    }

    else if (pt_selection_method == LIDAR_EDGES) {
      edge_detection(pc, semantic, expected_points, intensity_bound, depth_bound, distance_bound, beam_num,
                     pc_out, output_depth_grad, output_intenstity_grad, selected_indexes, semantic_out);
    } else {
      std::cerr<<"The point selection method is not implemented\n";
      return;
    }

    // fill in class members
    num_points_ = selected_indexes.size();
    num_classes_ = num_classes;
    num_geometric_types_ = 2;      
    
    // features_ = Eigen::MatrixXf::Zero(num_points_, 1);
    feature_dimensions_ = 1;
    features_.resize(num_points_, feature_dimensions_);
    labels_.resize(num_points_, num_classes_);
    positions_.resize(num_points_);
    int actual_ids = 0;
    for (int i = 0; i < num_points_ ; i++) {
      Vec3f xyz;
      int idx = selected_indexes[i];

      if (semantic_out[i] == -1)
        continue;

      // std::cout<<"pc_out (x,y,z,i)=("<<pc_out->points[i].x<<","<<pc_out->points[i].y<<","<<pc_out->points[i].z<<","<<pc_out->points[i].intensity<<"; output pointcloud (index,x,y,z,i)=("<<idx<<", "<<pc->points[idx].x<<","<<pc->points[idx].y<<","<<pc->points[idx].z<<","<<pc->points[idx].intensity<<")"<<std::endl;
      
      xyz << pc->points[idx].x, pc->points[idx].y, pc->points[idx].z;
      positions_[actual_ids] = (xyz);
      features_(actual_ids, 0) = pc->points[idx].intensity;

      // add one-hot semantic labels
      VecXf_row one_hot_label;
      one_hot_label = VecXf_row::Zero(1,num_classes_);
      one_hot_label[semantic_out[i]] = 1;

      labels_.row(actual_ids) = one_hot_label;
      int max_class = 0;
      labels_.row(actual_ids).maxCoeff(&max_class);

      // TODO: change geometric_types
      geometric_types_.push_back(1.0);
      geometric_types_.push_back(0.0);
      

      actual_ids ++;


    }
    std::cout<<"Construct Cvo PointCloud, num of points is "<<num_points_<<" from "<<pc->size()<<" input points "<<std::endl;
    //write_to_label_pcd("kitti_semantic_lidar.pcd");
    //write_to_intensity_pcd("kitti_intensity_lidar.pcd");
  }


  CvoPointCloud::CvoPointCloud(){
    num_points_=0;
    num_classes_ = 0;
    feature_dimensions_ = 0;
    num_geometric_types_ = 2;      
    
  }

  CvoPointCloud::CvoPointCloud(int feature_dimensions, int num_classes) {
    num_points_=0;
    num_classes_ = num_classes;
    feature_dimensions_ = feature_dimensions;
    num_geometric_types_ = 2;          
  }  
  
  CvoPointCloud::~CvoPointCloud() {
    // std::cout<<"Destruct CvoPointCloud..\n"<<std::flush;
    
  }

  int CvoPointCloud::read_cvo_pointcloud_from_file(const std::string & filename) {
    std::ifstream infile(filename);
    if (infile.is_open()) {
      infile>> num_points_;
      infile >> feature_dimensions_;
      infile>> num_classes_;
      positions_.clear();
      positions_.resize(num_points_);
      features_.resize(num_points_, feature_dimensions_);
      if (num_classes_)
        labels_.resize(num_points_, num_classes_ );
      for (int i = 0; i < num_points_; i++) {
        float u, v;
        infile >> u >>v;
        float idepth;
        infile >> idepth;
        for (int j = 0; j < feature_dimensions_ ; j++)
          infile >> features_(i, j);
        //features_(i,0) = features_(i,0) / 255.0;
        //features_(i,1) = features_(i,1) / 255.0;
        //features_(i,2) = features_(i,2) / 255.0;
        //features_(i,3) = features_(i,3) / 500.0 + 0.5;
        //features_(i,4) = features_(i,4) / 500.0 + 0.5;
        
        for (int j = 0; j < 3; j++)
          infile >> positions_[i]( j);
        for (int j = 0; j < num_classes_; j++)
          infile >> labels_(i, j);
      }
      infile.close();
      /*
        std::cout<<"Read pointcloud with "<<num_points_<<" points in "<<num_classes_<<" classes. \n The first point is ";
        std::cout<<" xyz: "<<positions_[0].transpose()<<", rgb_dxdy is "<<features_.row(0) <<"\n";
        if (num_classes_)
        std::cout<<" semantics is "<<labels_.row(0)<<std::endl;
        std::cout<<"The last point is ";
        std::cout<<" xyz: "<<positions_[num_points_-1].transpose()<<", rgb_dxdy is "<<features_.row(num_points_-1)<<"\n";
        if (num_classes_)
        std::cout<<" semantics is "<<labels_.row(num_points_-1)<<std::endl;
      */
      return 0;
    } else
      return -1;
    
  }


  template <>
  void CvoPointCloud::export_to_pcd<pcl::PointXYZRGB>(pcl::PointCloud<pcl::PointXYZRGB> & pc)  const {
    /*

    std::unordered_map<int, std::tuple<uint8_t, uint8_t, uint8_t>> label2color;
    label2color[0]  =std::make_tuple(128, 64,128 ); // road
    label2color[1]  =std::make_tuple(244, 35,232 ); // sidewalk
    label2color[2]  =std::make_tuple(70, 70, 70 ); // sidewalk
    label2color[3]  =std::make_tuple(102,102,156   ); // building
    label2color[4] =std::make_tuple(190,153,153 ); // pole
    label2color[5] =std::make_tuple(153,153,153  ); // sign
    label2color[6]  =std::make_tuple(250,170, 30   ); // vegetation
    label2color[7]  =std::make_tuple(220,220,  0   ); // terrain
    label2color[8] =std::make_tuple(107,142, 35 ); // sky
    label2color[9]  =std::make_tuple(152,251,152 ); // water
    label2color[10]  =std::make_tuple(70,130,180  ); // person
    label2color[11]  =std::make_tuple( 220, 20, 60   ); // car
    label2color[12]  =std::make_tuple(255,  0,  0  ); // bike
    label2color[13] =std::make_tuple( 0,  0,142 ); // stair
    label2color[14]  =std::make_tuple(0,  0, 70 ); // background
    label2coor[15]  =std::make_tuple(0, 60,100 ); // background
    label2color[16]  =std::make_tuple(0, 80,100 ); // background
    label2color[17]  =std::make_tuple( 0,  0,230 ); // background
    label2color[18]  =std::make_tuple(119, 11, 32 ); // background

    */
    pc.resize(num_points_);
    for (int i = 0; i < num_points_; i++) {
      pcl::PointXYZRGB p;
//      p.x = positions_[i]( 0);
//      p.y = positions_[i]( 1);
//      p.z = positions_[i]( 2);
      p.x = points_[i].x;
      p.y = points_[i].y;
      p.z = points_[i].z;

      if (feature_dimensions_) {
        uint8_t r = static_cast<uint8_t>(std::min(255, (int)(points_[i].features[2] * 255) ) );
        uint8_t g = static_cast<uint8_t>(std::min(255, (int)(points_[i].features[1] * 255) ) );
        uint8_t b = static_cast<uint8_t>(std::min(255, (int)(points_[i].features[0] * 255)));
        //if (num_classes_ ) {
        //  int max_class;
        //  labels_.row(i).maxCoeff(&max_class);
        //  auto c = label2color[max_class];
        //  auto r = std::get<0>(c);
        //  auto g = std::get<1>(c);
        //  auto b = std::get<2>(c);
        //if (i == 0)
        //  std::cout<<"export to pcd: color r is "<< (int )r <<std::endl;
        // }
        //uint32_t rgb = ((uint32_t) r << 16 |(uint32_t) g << 8  | (uint32_t) b ) ;
        //p.rgb = *reinterpret_cast<float*>(&rgb);
        p.r = r;
        p.g = g;
        p.b = b;
      }
      pc[i] = p;
    }
    
  }


  template <>
  void CvoPointCloud::export_to_pcd<pcl::PointXYZ>(pcl::PointCloud<pcl::PointXYZ> & pc)  const {
    pc.resize(num_points_);
    for (int i = 0; i < num_points_; i++) {
      pcl::PointXYZ p;
//      p.x = positions_[i](0);
//      p.y = positions_[i](1);
//      p.z = positions_[i](2);
      p.x = points_[i].x;
      p.y = points_[i].y;
      p.z = points_[i].z;
      pc[i] = p;
    }
    
  }
  

  Eigen::Vector3f CvoPointCloud::at(unsigned int index) const {
    assert (index < num_points_ && index >= 0);
    Eigen::Vector3f p  = positions_[index];
    return p;
  }

  Eigen::Vector2f CvoPointCloud::geometry_type_at(unsigned int index) const {
    Eigen::Map<const Eigen::Vector2f> gtype(geometric_types_.data()+ index*2);
    Eigen::Vector2f ret = gtype;
    return gtype;
  }
  

  void CvoPointCloud::write_to_color_pcd(const std::string & name) const {
    pcl::PointCloud<pcl::PointXYZRGB> pc;
    export_to_pcd<pcl::PointXYZRGB>(pc);
    pcl::io::savePCDFileASCII(name ,pc);  
  }

  void CvoPointCloud::write_to_pcd(const std::string & name) const {
    pcl::PointCloud<pcl::PointXYZ> pc;
    for (int i = 0; i < num_points_; i++) {
      pcl::PointXYZ p;
      p.x = positions_[i]( 0);
      p.y = positions_[i]( 1);
      p.z = positions_[i]( 2);
      pc.push_back(p);
    }
    pcl::io::savePCDFileASCII(name, pc); 
    std::cout << "Finished write to pcd" << std::endl; 
  }

  void CvoPointCloud::write_to_label_pcd(const std::string & name) const {
    if (num_classes_ < 1)
      return;
    
    pcl::PointCloud<pcl::PointXYZL> pc;
    for (int i = 0; i < num_points_; i++) {
      pcl::PointXYZL p;
      p.x = positions_[i](0);
      p.y = positions_[i](1);
      p.z = positions_[i](2);
      int l;
      labels_.row(i).maxCoeff(&l);
      p.label = (uint32_t) l;
      pc.push_back(p);
    }
    pcl::io::savePCDFileASCII(name ,pc); 
    std::cout << "Finished write to label pcd" << std::endl; 
  }

  void CvoPointCloud::write_to_txt(const std::string & name) const {
    std::ofstream outfile(name);
    if (outfile.is_open()) {
      outfile << num_points_<<" "<<num_classes_<<"\n";
      for (int i = 0; i < num_points_; i++) {
        outfile << positions_[i](0)<<" "<<positions_[i](1) <<" "<<positions_[i](2)<<std::endl;
        for (int j = 0; j < feature_dimensions_; j++) {
          outfile << features_(i, j)<<" ";
        }
        if (num_classes_)
          for (int j = 0; j < num_classes_; j++) {
            outfile << labels_(i, j)<<" ";
          }
        outfile << "\n";
        
      }
      outfile.close();

    }
    std::cout << "Finished write to txt" << std::endl; 
    
  }

  void CvoPointCloud::write_to_intensity_pcd(const std::string & name) const {
    pcl::PointCloud<pcl::PointXYZI> pc;
    for (int i = 0; i < num_points_; i++) {
      pcl::PointXYZI p;
      p.x = positions_[i](0);
      p.y = positions_[i](1);
      p.z = positions_[i](2);
      p.intensity = features_(i);
      pc.push_back(p);
    }
    pcl::io::savePCDFileASCII(name ,pc);  
    std::cout << "Finished write to intensity pcd" << std::endl << std::flush; 
  }
  

  
  void CvoPointCloud::transform(const Eigen::Matrix4f& pose,
                                const CvoPointCloud & input,
                                CvoPointCloud & output) {
    output.num_points_ = input.num_points();
    output.num_classes_ = input.num_classes();
    output.feature_dimensions_ = input.feature_dimensions();
    output.points_ = input.get_points();
//    output.features_ = input.features();
//    copy_eigen_dynamic_matrix(&input.features_,
//                              &output.features_);
    
//    output.labels_ = input.labels();
//    output.positions_.resize(output.num_points_);
    tbb::parallel_for(int(0), input.num_points(), [&](int j) {
        Eigen::Vector3f jth_position = Eigen::Vector3f(input.point_at(j).x, input.point_at(j).y, input.point_at(j).z);
        Eigen::Vector3f temp = (pose.block(0, 0, 3, 3) * jth_position +
                pose.block(0, 3, 3, 1)).eval();
        output.points_[j].x = temp.x();
        output.points_[j].y = temp.y();
        output.points_[j].z = temp.z();
    });
  }

  void CvoPointCloud::reserve(int num_points, int feature_dims, int num_classes) {
    num_points_ = num_points;
    num_classes_ = num_classes;
    feature_dimensions_ = feature_dims;
    num_geometric_types_ = 2;      
    positions_.resize(num_points_);
    if (feature_dims)
      features_.resize(num_points_, feature_dimensions_);
    if (num_classes_)
      labels_.resize(num_points_, num_classes_);
    geometric_types_ .resize(num_points*2);
  }
  
  int CvoPointCloud::add_point(int index, const Eigen::Vector3f & xyz, const Eigen::VectorXf & feature, const Eigen::VectorXf & label, const Eigen::VectorXf & geometry_type) {
    
    if (index >= num_points_) return -1;
    if (positions_.size() < num_points_ ||
        features_.rows() < num_points_ ||
        //labels_.rows() < num_points_ ||
        features_.cols() != feature_dimensions_ ||
        geometry_type.size() != 2
        //labels_.cols() != num_classes_
        ) {
      std::cerr<<"CvoPointCloud must be reserved before add_point\n";
      return -1;
    }

    positions_[index] = xyz;
    if (feature_dimensions_)
      features_.row(index) = feature.transpose();
    if (num_classes_)
      labels_.row(index) = label;
    if (geometry_type.size()) {
      geometric_types_[index*2] = (geometry_type(0));
      geometric_types_[index*2+1] = (geometry_type(1));
    }
    return 0;
  }



}
