#include <string>
#include <fstream>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <algorithm>
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
//#include "opencv2/xfeatures2d.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
//#include "mapping/bkioctomap.h"
namespace cvo{


  // filter out sky or too-far-away pixels
  static bool is_good_point(const Vec3f & xyz, const Vec2i uv, int h, int w ) {
    int u = uv(0);
    int v = uv(1);
    if ( u < 2 || u > w -2 || v < 100 || v > h-30 )
      return false;

    if (xyz.norm() >=55)
      return false;

    return true;
  }

  // filter away pixels that are too far away
  static bool is_good_point(const Vec3f & xyz ) {
    if (xyz.norm() > 70)
      return false;

    return true;
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

  
  
  cv::Vec3f CvoPointCloud::avg_pixel_color_pattern(const cv::Mat & raw_buffer, int u, int v, int w){
    cv::Vec3f result_cv;
    result_cv[0] = result_cv[1] = result_cv[2] = 0;
    for (int i = 0; i < 8; i++){
      cv::Vec3f pattern;
      int u_pattern = pixel_pattern[i][0]+u;
      int v_pattern = pixel_pattern[i][1]+v;
      std::cout<<"at pattern "<<" ("<<pixel_pattern[i][0]+u<<","<<pixel_pattern[i][1]+v<<"): ";
      pattern = raw_buffer.at<cv::Vec3b>(v_pattern, u_pattern);
      std::cout<<" is "<<pattern;
      result_cv = result_cv + pattern;
      std::cout<<std::endl;
    }
    std::cout<<"Result: "<<result_cv <<std::endl;
    result_cv  = (result_cv / 8);

    return result_cv;
  }


  CvoPointCloud::CvoPointCloud(const std::string & filename) {
 
    std::ifstream infile;
    int total_num_points = 0;
    num_points_ = 0;
    infile.open(filename);
    if (infile.is_open()) {
      infile >> total_num_points >> num_classes_;
      for (int i =0; i < total_num_points ; i++) {
        Vec3f pos;
        Vec5f feature;
        VecXf label(num_classes_);
        infile >> pos(0) >> pos(1) >> pos(2);
        for (int j = 0 ; j < 5; j++)
          infile >> feature(j);
        for (int j = 0; j < num_classes_; j++)
          infile >> label(j);
        if (is_good_point(pos))
          num_points_++;
      }
      
      infile.close();
    } 
   
    infile.open(filename);
    int good_point_ind = 0;
    if (infile.is_open()) {
      infile >> total_num_points >> num_classes_;
      positions_.resize(num_points_);
      feature_dimensions_ = 5;
      features_.resize(num_points_, feature_dimensions_);
      labels_.resize(num_points_, num_classes_);

      for (int i =0; i < total_num_points ; i++) {
        Vec3f pos;        
        VecXf label(num_classes_);
        infile >> pos(0) >> pos(1) >> pos(2);
        Vec5f feature;
        float feature_1;

        if(feature_dimensions_==5){          
          for (int j = 0 ; j < feature_dimensions_; j++)
            infile >> feature(j);
        }
        else if(feature_dimensions_==1){          
          infile >> feature_1;
        } 
        
        for (int j = 0; j < num_classes_; j++)
          infile >> label(j);
        if (is_good_point(pos)) {
          positions_[ good_point_ind] = pos.transpose();
          if(feature_dimensions_==5){
            features_.row(good_point_ind ) = feature.transpose();
          }
          else if(feature_dimensions_==1){
            features_(good_point_ind ) = feature_1;
          } 
          
          labels_.row(good_point_ind ) = label.transpose();
          good_point_ind ++;
        }
      }
      
      infile.close();
    } else {
      printf("empty CvoPointCloud file!\n");
      assert(0);
      
    }
    
  }
  CvoPointCloud::CvoPointCloud(const RawImage & rgb_raw_image,
                               const cv::Mat & depth_image,
                               const Calibration &calib,
                               const bool is_using_rgbd,
                               PointSelectionMethod pt_selection_method){
    if(is_using_rgbd){
      int expected_points = 5000;
      std::vector<Vec2i, Eigen::aligned_allocator<Vec2i>> output_uv;
      cv::Mat  rgb_gray;
      cv::cvtColor(rgb_raw_image.color(), rgb_gray, cv::COLOR_BGR2GRAY);  
      if (pt_selection_method == CV_FAST) {

        int minHessian = 400;
        std::vector<cv::KeyPoint> keypoints;    
        //auto detector = cv::xfeatures2d:: SURF::create( minHessian );
        //auto detector =  cv::FastFeatureDetector::create();
        cv::FAST(rgb_gray,keypoints, 5,false);

        int thresh = 9, num_want = 15000, num_min = 12000;
        while (keypoints.size() > num_want)  {
          std::cout<<"selected "<<keypoints.size()<<" points more than "<<num_want<<std::endl;
          keypoints.clear();
          thresh++;
          cv::FAST(rgb_gray,keypoints, thresh,false);
          if (thresh == 13) break;
        }
        while (keypoints.size() < num_min ) {
          std::cout<<"selected "<<keypoints.size()<<" points less than "<<num_min<<std::endl;
          keypoints.clear();
          thresh--;
          cv::FAST(rgb_gray,keypoints, thresh,false);
          if (thresh== 0) break;

        }
        std::cout<<"FAST selected "<<keypoints.size()<<std::endl;
      
        //detector->detect( left_gray, keypoints, cv::Mat() );
        for (auto && kp: keypoints) {
          Vec2i p;
          p(0) = (int)kp.pt.x;
          p(1) = (int)kp.pt.y;
          output_uv.push_back(p);
        }

      } else {
      
        select_pixels(rgb_raw_image,
                      expected_points,
                      output_uv);
      }
      std::vector<int> good_point_ind;
      int h = rgb_raw_image.color().rows;
      int w = rgb_raw_image.color().cols;
      Mat33f intrinsic = calib.intrinsic();

      for (int i = 0; i < output_uv.size(); i++) {
        auto uv = output_uv[i];
        int u = uv(0);
        int v = uv(1);
        Vec3f xyz;

        uint16_t dep = depth_image.at<uint16_t>(cv::Point(u, v));
        
        if(dep!=0 && !isnan(dep)){

          // construct depth
          xyz(2) = dep/calib.scaling_factor();
            
          // construct x and y
          xyz(0) = (u-intrinsic(0,2)) * xyz(2) / intrinsic(0,0);
          xyz(1) = (v-intrinsic(1,2)) * xyz(2) / intrinsic(1,1);
            
          // add point to pcd
          good_point_ind.push_back(i);
          positions_.push_back(xyz);

        }
      }

      num_points_ = good_point_ind.size();
      num_classes_ = rgb_raw_image.num_class();
      feature_dimensions_ = 5;
      features_.resize(num_points_, feature_dimensions_);
      for (int i = 0; i < num_points_ ; i++) {
        int u = output_uv[good_point_ind[i]](0);
        int v = output_uv[good_point_ind[i]](1);
        cv::Vec3b avg_pixel = rgb_raw_image.color().at<cv::Vec3b>(v,u);
        auto & gradient = rgb_raw_image.gradient()[v * w + u];
        features_(i,0) = ((float)(avg_pixel [0]) )/255.0;
        features_(i,1) = ((float)(avg_pixel[1]) )/255.0;
        features_(i,2) = ((float)(avg_pixel[2]) )/255.0;
        features_(i,3) = gradient(0)/ 500.0 + 0.5;
        features_(i,4) = gradient(1)/ 500.0 + 0.5;
      }
    }
  }

  CvoPointCloud::CvoPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc) {
    num_points_ = pc->size();
    num_classes_ = 0;
    feature_dimensions_ = 3;

    positions_.resize(pc->size());
    features_.resize(num_points_, feature_dimensions_);
    for (int i = 0; i < num_points_; i++) {
      Eigen::Vector3f xyz;
      auto & p = (*pc)[i];
      xyz << p.x, p.y, p.z;
      positions_[i] = xyz;

      features_(i,0) = ((float)(int)p.r) / 255.0;
      features_(i,1) = ((float)(int)p.g) / 255.0;
      features_(i,2) = ((float)(int)p.b) / 255.0;
      
    }
    
  }
  


  static void stereo_surface_sampling(const cv::Mat & left_gray,
                                      const std::vector<Vec2i, Eigen::aligned_allocator<Vec2i>> & dso_selected_uv,
                                      bool is_using_canny,
                                      bool is_using_uniform_rand,
                                      // output
                                      std::vector<bool> & selected_inds_map,
                                      std::vector<Vec2i, Eigen::aligned_allocator<Vec2i>> & final_selected_uv                   
                                      ) {
    selected_inds_map.resize(left_gray.total(), false);
    std::cout<<"dso selected uv size is "<<dso_selected_uv.size()<<std::endl;
    for (auto && uv : dso_selected_uv) {
      int u = uv(0);
      int v = uv(1);
      selected_inds_map[v * left_gray.cols + u]  = true;
    }
    // canny
    cv::Mat detected_edges;
    if (is_using_canny)
      cv::Canny( left_gray, detected_edges, 50, 50*3, 3 );

    for (int r = 0 ; r < left_gray.rows; r++) {
      for (int c = 0; c < left_gray.cols; c++) {
        // using Canny
        if (is_using_canny &&  detected_edges.at<uint8_t>(r, c) > 0) 
          selected_inds_map[r * left_gray.cols + c] = true;

        // using uniform sampling
        if (is_using_uniform_rand && r > left_gray.rows * 1 /3  &&  rand() % 50 == 0) 
          selected_inds_map[r * left_gray.cols + c] = true;

        if (selected_inds_map[r * left_gray.cols + c])
          final_selected_uv.push_back(Vec2i(c, r));                    
      }
      
    }

    //std::cout<<" final selected uv size is "<<final_selected_uv.size()<<std::endl;
    //cv::imwrite("canny.png", detected_edges);    
    
  }
  
  CvoPointCloud::CvoPointCloud(const RawImage & left_image,
                               const cv::Mat & right_image,
                               const Calibration & calib,
                               PointSelectionMethod pt_selection_method) {
    
    cv::Mat  left_gray, right_gray;
    cv::cvtColor(right_image, right_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(left_image.color(), left_gray, cv::COLOR_BGR2GRAY);

    std::vector<float> left_disparity;
    StaticStereo::disparity(left_gray, right_gray, left_disparity);
    std::vector<Vec2i, Eigen::aligned_allocator<Vec2i>> output_uv;

    /*****************************************/
    if (pt_selection_method == CV_FAST) {
      // using opencv FAST
      //-- Step 1: Detect the keypoints using SURF Detector
      int minHessian = 400;
      std::vector<cv::KeyPoint> keypoints;    
      cv::FAST(left_gray,keypoints, 5,false);

      // for semantic
      // int thresh = 4, num_want = 28000, num_min = 15000;
      //// for geometric
      int thresh = 4, num_want = 24000, num_min = 15000;
      if (left_image.num_class() > 0)
	      num_want = 28000;
      while (keypoints.size() > num_want)  {
        std::cout<<"selected "<<keypoints.size()<<" points more than "<<num_want<<std::endl;
        keypoints.clear();
        thresh++;
        cv::FAST(left_gray,keypoints, thresh,false);
        if (thresh == 50) break;
      }
      while (keypoints.size() < num_min ) {
        std::cout<<"selected "<<keypoints.size()<<" points less than "<<num_min<<std::endl;
        keypoints.clear();
        thresh--;
        cv::FAST(left_gray,keypoints, thresh,false);
        if (thresh== 0) break;

      }
      std::cout<<"FAST selected "<<keypoints.size()<<std::endl;
    
      //detector->detect( left_gray, keypoints, cv::Mat() );
      for (auto && kp: keypoints) {
        Vec2i p;
        p(0) = (int)kp.pt.x;
        p(1) = (int)kp.pt.y;
        output_uv.push_back(p);
      }
      bool debug_plot = false;
      if (debug_plot) {
        std::cout<<"Number of selected points is "<<output_uv.size()<<"\n";
        cv::Mat heatmap(left_image.color().rows, left_image.color().cols, CV_32FC1, cv::Scalar(0) );
        int w = heatmap.cols;
        int h = heatmap.rows;
        for (int i = 0; i < output_uv.size(); i++) 
          cv::circle(heatmap, cv::Point( output_uv[i](0), output_uv[i](1) ), 1, cv::Scalar(255, 0 ,0), 1);
        cv::imwrite("FAST_selected_pixels.png", heatmap);
      }
    } 
    /*****************************************/
    // using DSO semi dense point selector
    else if (pt_selection_method == DSO_EDGES) {

      int expected_points = 20000;
      select_pixels(left_image,
                    expected_points,
                    output_uv);
    }
    //******************************************/
    // using canny or random point selection
    else if (pt_selection_method == CANNY_EDGES) {
      std::vector<bool> selected_inds_map;
      std::vector<Vec2i, Eigen::aligned_allocator<Vec2i>> final_selected_uv;
      stereo_surface_sampling(left_gray, output_uv, true, false,
                              selected_inds_map, final_selected_uv);
    }
    /********************************************/
    // using full point cloud
    else if (pt_selection_method == FULL) {
      output_uv.clear();
      for (int h = 0; h < left_image.color().cols; h++){
        for (int w = 0; w < left_image.color().rows; w++){
          Vec2i uv;
          uv << h , w;
          output_uv.push_back(uv);
        }
      }
    } else {
      std::cerr<<"This point selection method is not implemented.\n";
      return;
    }
    /**********************************************/
    //auto & pre_depth_selected_ind = final_selected_uv;
    auto & pre_depth_selected_ind = output_uv;
    std::vector<int> good_point_ind;
    int h = left_image.color().rows;
    int w = left_image.color().cols;
    cv::Mat depth_map(h, w, CV_32F, cv::Scalar(0));
    bool is_recording_depth_map = true;
    static unsigned int depth_map_counter = 0;    
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
        if(left_image.num_class() ){
          auto labels = Eigen::Map<const VecXf_row>((left_image.semantic_image().data()+ (v * w + u)*left_image.num_class()), left_image.num_class() );
          int max_class = 0;
          labels.maxCoeff(&max_class);
          if( max_class == 10)
            // exclude unlabeled points
            continue;
        }

        good_point_ind.push_back(i);
        //good_point_xyz.push_back(xyz);
        positions_.push_back(xyz);
        
      }
    }
     
    // start to fill in class members
    num_points_ = good_point_ind.size();
    num_classes_ = left_image.num_class();
    if (num_classes_ )
      labels_.resize(num_points_, num_classes_);
    feature_dimensions_ = 5;
    features_.resize(num_points_, feature_dimensions_);
    for (int i = 0; i < num_points_ ; i++) {
      int u = pre_depth_selected_ind[good_point_ind[i]](0);
      int v = pre_depth_selected_ind[good_point_ind[i]](1);
      cv::Vec3b avg_pixel = left_image.color().at<cv::Vec3b>(v,u);
      auto & gradient = left_image.gradient()[v * w + u];
      features_(i,0) = ((float)(avg_pixel [0]) )/255.0;
      features_(i,1) = ((float)(avg_pixel[1]) )/255.0;
      features_(i,2) = ((float)(avg_pixel[2]) )/255.0;
      features_(i,3) = gradient(0)/ 500.0 + 0.5;
      features_(i,4) = gradient(1)/ 500.0 + 0.5;

      if (num_classes_) {
        labels_.row(i) = Eigen::Map<const VecXf_row>((left_image.semantic_image().data()+ (v * w + u)*num_classes_), num_classes_);
        int max_class = 0;
        labels_.row(i).maxCoeff(&max_class);
      }

    }
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
    
    // features_ = Eigen::MatrixXf::Zero(num_points_, 1);
    feature_dimensions_ = 1;
    features_.resize(num_points_, feature_dimensions_);
    normals_.resize(num_points_,3);
    //types_.resize(num_points_, 2);

    for (int i = 0; i < num_points_ ; i++) {
      Vec3f xyz;
      int idx = selected_indexes[i];
      xyz << pc->points[idx].x, pc->points[idx].y, pc->points[idx].z;
      positions_.push_back(xyz);
      features_(i, 0) = pc->points[idx].intensity;

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
    
    // features_ = Eigen::MatrixXf::Zero(num_points_, 1);
    feature_dimensions_ = 1;
    features_.resize(num_points_, feature_dimensions_);
    labels_.resize(num_points_, num_classes_);
    for (int i = 0; i < num_points_ ; i++) {
      Vec3f xyz;
      int idx = selected_indexes[i];

      if (semantic_out[i] == -1)
        continue;

      // std::cout<<"pc_out (x,y,z,i)=("<<pc_out->points[i].x<<","<<pc_out->points[i].y<<","<<pc_out->points[i].z<<","<<pc_out->points[i].intensity<<"; output pointcloud (index,x,y,z,i)=("<<idx<<", "<<pc->points[idx].x<<","<<pc->points[idx].y<<","<<pc->points[idx].z<<","<<pc->points[idx].intensity<<")"<<std::endl;
      
      xyz << pc->points[idx].x, pc->points[idx].y, pc->points[idx].z;
      positions_.push_back(xyz);
      features_(i, 0) = pc->points[idx].intensity;

      // add one-hot semantic labels
      VecXf_row one_hot_label;
      one_hot_label = VecXf_row::Zero(1,num_classes_);
      one_hot_label[semantic_out[i]] = 1;

      labels_.row(i) = one_hot_label;
      int max_class = 0;
      labels_.row(i).maxCoeff(&max_class);

    }
    std::cout<<"Construct Cvo PointCloud, num of points is "<<num_points_<<" from "<<pc->size()<<" input points "<<std::endl;
    //write_to_label_pcd("kitti_semantic_lidar.pcd");
    //write_to_intensity_pcd("kitti_intensity_lidar.pcd");
  }


  CvoPointCloud::CvoPointCloud(){}
  CvoPointCloud::~CvoPointCloud() {
    std::cout<<"Destruct CvoPointCloud..\n"<<std::flush;
    
  }

  int CvoPointCloud::read_cvo_pointcloud_from_file(const std::string & filename, int feature_dim) {
    std::ifstream infile(filename);
    feature_dimensions_ = feature_dim;
    if (infile.is_open()) {
      infile>> num_points_;
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
        for (int j = 0; j <5 ; j++)
          infile >> features_(i, j);
        features_(i,0) = features_(i,0) / 255.0;
        features_(i,1) = features_(i,1) / 255.0;
        features_(i,2) = features_(i,2) / 255.0;
        features_(i,3) = features_(i,3) / 500.0 + 0.5;
        features_(i,4) = features_(i,4) / 500.0 + 0.5;
        
        for (int j = 0; j < 3; j++)
          infile >> positions_[i](j);
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


  void CvoPointCloud::write_to_color_pcd(const std::string & name) const {
    pcl::PointCloud<pcl::PointXYZRGB> pc;
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
    label2color[15]  =std::make_tuple(0, 60,100 ); // background
    label2color[16]  =std::make_tuple(0, 80,100 ); // background
    label2color[17]  =std::make_tuple( 0,  0,230 ); // background
    label2color[18]  =std::make_tuple(119, 11, 32 ); // background


    for (int i = 0; i < num_points_; i++) {
      pcl::PointXYZRGB p;
      p.x = positions_[i](0);
      p.y = positions_[i](1);
      p.z = positions_[i](2);
      
      uint8_t b = static_cast<uint8_t>(std::min(255, (int)(features_(i,0) * 255) ) );
      uint8_t g = static_cast<uint8_t>(std::min(255, (int)(features_(i,1) * 255) ) );
      uint8_t r = static_cast<uint8_t>(std::min(255, (int)(features_(i,2) * 255)));
      //if (num_classes_ ) {
      //  int max_class;
      //  labels_.row(i).maxCoeff(&max_class);
      //  auto c = label2color[max_class];
      //  auto r = std::get<0>(c);
      //  auto g = std::get<1>(c);
      //  auto b = std::get<2>(c);
          
      // }
      uint32_t rgb = ((uint32_t) r << 16 |(uint32_t) g << 8  | (uint32_t) b ) ;
      p.rgb = *reinterpret_cast<float*>(&rgb);
      pc.push_back(p);
    }
    pcl::io::savePCDFileASCII(name ,pc);  
  }

  void CvoPointCloud::write_to_pcd(const std::string & name) const {
    pcl::PointCloud<pcl::PointXYZ> pc;
    for (int i = 0; i < num_points_; i++) {
      pcl::PointXYZ p;
      p.x = positions_[i](0);
      p.y = positions_[i](1);
      p.z = positions_[i](2);
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
    output.features_ = input.features();
    output.labels_ = input.labels();
    output.positions_.resize(output.num_points_);
    tbb::parallel_for(int(0), input.num_points(), [&](int j) {
      output.positions_[j] = (pose.block(0, 0, 3, 3) * input.positions()[j] + pose.block(0, 3, 3, 1)).eval();
    });
  }


}
