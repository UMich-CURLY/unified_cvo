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
#include <tbb/tbb.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "utils/CvoPointCloud.hpp"
#include "utils/StaticStereo.hpp"
#include "utils/CvoPixelSelector.hpp"
#include "utils/LidarPointSelector.hpp"


//#include "mapping/bkioctomap.h"
namespace cvo{



  static bool is_good_point(const Vec3f & xyz, const Vec2i uv, int h, int w ) {
    int u = uv(0);
    int v = uv(1);
    if ( u < 2 || u > w -3 || v < 2 || v > h-50 )
      return false;

    if (xyz.norm() > 40)
      return false;

    return true;
  }

  static bool is_good_point(const Vec3f & xyz ) {
    if (xyz.norm() > 40)
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
                  const bool& is_using_rgbd){
    if(is_using_rgbd){
      int expected_points = 5000;
      std::vector<Vec2i, Eigen::aligned_allocator<Vec2i>> output_uv;
      select_pixels(rgb_raw_image,
                    expected_points,
                    output_uv);

      std::vector<int> good_point_ind;
      int h = rgb_raw_image.color().rows;
      int w = rgb_raw_image.color().cols;
      Mat33f intrinsic = calib.intrinsic();

      // cv::Mat img_selected;
      // rgb_raw_image.color().copyTo(img_selected);

      // for(int i=0; i<h; ++i){
      //   for(int j=0; j<w; ++j){
      //     img_selected.at<cv::Vec3b>(cv::Point(j, i)).val[0] = 0;
      //     img_selected.at<cv::Vec3b>(cv::Point(j, i)).val[1] = 0;
      //     img_selected.at<cv::Vec3b>(cv::Point(j, i)).val[2] = 0;
      //   }
      // }

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
            // x = (u-cx)*z/fx
            // y = (v-cy)*z/fy
            xyz(0) = (u-intrinsic(0,2)) * xyz(2) / intrinsic(0,0);
            xyz(1) = (v-intrinsic(1,2)) * xyz(2) / intrinsic(1,1);
            
            // add point to pcd
            good_point_ind.push_back(i);
            positions_.push_back(xyz);

            // for visualization
            // cv::Vec3b avg_pixel = rgb_raw_image.color().at<cv::Vec3b>(v,u);
            // img_selected.at<cv::Vec3b>(cv::Point(u, v)).val[0] = avg_pixel [0];
            // img_selected.at<cv::Vec3b>(cv::Point(u, v)).val[1] = avg_pixel [1];
            // img_selected.at<cv::Vec3b>(cv::Point(u, v)).val[2] = avg_pixel [2];
        }
      }

      // cv::imshow("selected img: ",img_selected);
      // cv::waitKey(0);
      // cv::imwrite("selected_img.jpg", img_selected);

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



  static void stereo_surface_sampling(const cv::Mat & left_gray,
                                      const std::vector<Vec2i, Eigen::aligned_allocator<Vec2i>> & dso_selected_uv,
                                      bool is_using_canny,
                                      bool is_using_uniform_rand,
                                      // output
                                      std::vector<bool> & selected_inds_map,
                                      std::vector<Vec2i, Eigen::aligned_allocator<Vec2i>> & final_selected_uv                   
                                      ) {
    selected_inds_map.resize(left_gray.total(), false);    
    for (auto && uv : dso_selected_uv) {
      int u = uv(0);
      int v = uv(1);
      selected_inds_map[v * left_gray.cols + u]  = true;
    }
    // canny
    cv::Mat detected_edges;
    if (is_using_canny)
      cv::Canny( left_gray, detected_edges, 50, 50*3, 3 );

    if (is_using_canny || is_using_uniform_rand) {
      for (int r = 0 ; r < left_gray.rows; r++) {
        for (int c = 0; c < left_gray.cols; c++) {
          // using Canny
          if (is_using_canny && detected_edges.at<uint8_t>(r, c) > 0) 
            selected_inds_map[r * left_gray.cols + c] = true;

          // using uniform sampling
          if (is_using_uniform_rand && rand() % 100 == 0) 
            selected_inds_map[r * left_gray.cols + c] = true;

          if (selected_inds_map[r * left_gray.cols + c])
            final_selected_uv.push_back(Vec2i(c, r));                    
        }
      
      }
    }

    
    
  }
  
  CvoPointCloud::CvoPointCloud(const RawImage & left_image,
                               const cv::Mat & right_image,
                               const Calibration & calib ) {
    
    cv::Mat  left_gray, right_gray;
    cv::cvtColor(right_image, right_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(left_image.color(), left_gray, cv::COLOR_BGR2GRAY);

    std::vector<float> left_disparity;
    StaticStereo::disparity(left_gray, right_gray, left_disparity);

    int expected_points = 5000;
    std::vector<Vec2i, Eigen::aligned_allocator<Vec2i>> output_uv;
    select_pixels(left_image,
                  expected_points,
                  output_uv);

    //******************************************/
    std::vector<bool> selected_inds_map;
    std::vector<Vec2i, Eigen::aligned_allocator<Vec2i>> final_selected_uv;
    stereo_surface_sampling(left_gray, output_uv, true, true,
                            selected_inds_map, final_selected_uv);
    /********************************************/
    

    // for (int h = 0; h < left_image.color().cols; h++){
    //   for (int w = 0; w < left_image.color().rows; w++){
    //     Vec2i uv;
    //     uv << h , w;
    //     output_uv.push_back(uv);
    //   }
    // }
    
    auto & pre_depth_selected_ind = final_selected_uv;
    //auto & pre_depth_selected_ind = output_uv;

    std::vector<int> good_point_ind;
    int h = left_image.color().rows;
    int w = left_image.color().cols;
    for (int i = 0; i < pre_depth_selected_ind.size(); i++) {
      auto uv = pre_depth_selected_ind[i];
      Vec3f xyz;

      StaticStereo::TraceStatus trace_status = StaticStereo::pt_depth_from_disparity(left_image,
                                                                                     left_disparity,
                                                                                     calib,
                                                                                     uv,
                                                                                     xyz );
      if (trace_status == StaticStereo::TraceStatus::GOOD && is_good_point (xyz, uv, h, w) ) {
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
    //write_to_label_pcd("labeled_input.pcd");
    write_to_color_pcd("color_stereo.pcd");
  }
  

  CvoPointCloud::CvoPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pc, int target_num_points, int beam_num) {
    int expected_points = target_num_points;
    double intensity_bound = 0.4;
    double depth_bound = 4.0;
    double distance_bound = 75.0;
    std::vector <int> selected_indexes;
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out (new pcl::PointCloud<pcl::PointXYZI>);    
    std::vector <double> output_depth_grad;
    std::vector <double> output_intenstity_grad;

#if  !defined(IS_USING_LOAM)  && defined(IS_USING_NORMALS)
    pcl::PointCloud<pcl::Normal>::Ptr normals_out (new pcl::PointCloud<pcl::Normal>);
    edge_detection(pc, expected_points, intensity_bound, depth_bound, distance_bound, beam_num,
                   // output
                   pc_out, output_depth_grad, output_intenstity_grad, selected_indexes, normals_out);
    
    num_points_ = pc_out->size();    
#endif    

#if defined(IS_USING_LOAM) && !defined(IS_USING_NORMALS)
    std::vector <float> edge_or_surface;
    LidarPointSelector lps(expected_points, intensity_bound, depth_bound, distance_bound, beam_num);

    // running edge detection + lego loam point selection
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_edge (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_surface (new pcl::PointCloud<pcl::PointXYZI>);
    lps.edge_detection(pc, pc_out_edge, output_depth_grad, output_intenstity_grad, selected_indexes);   
    std::cout<<"Number edge selection result is "<<selected_indexes.size()<<std::endl; 
    std::cout << "\nList of selected edge indexes: " << std::endl;
    for(int i=0; i<10; i++)
      std::cout << selected_indexes[i] << " ";
    lps.legoloam_point_selector(pc, pc_out_surface, edge_or_surface, selected_indexes);    
    *pc_out += *pc_out_edge;
    *pc_out += *pc_out_surface;
    num_points_ = pc_out->size();
    assert(num_points_ == selected_indexes.size());
    std::cout << "\nList of selected lego indexes" << std::endl;
    for(int i=0; i<10; i++){
      std::cout << selected_indexes[i] << " ";
    }
    // std::cout << "\n=================" << std::endl;
    // direct downsample using pcl
    // pcl::PointCloud<pcl::PointXYZI>::Ptr pc_randomground (new pcl::PointCloud<pcl::PointXYZI>);
    // lps.legoloam_point_selector(pc, pc_randomground, edge_or_surface, selected_indexes);
    // std::cout << "downsample using pcl voxel grid" << std::endl;
    // pcl::VoxelGrid<PointType> downSizeFilter;
    // downSizeFilter.setLeafSize(0.7, 0.7, 0.7);
    // downSizeFilter.setInputCloud(pc_randomground);
    // downSizeFilter.filter(*pc_out);

#endif

#if defined(IS_USING_LOAM) && defined(IS_USING_NORMALS)
     std::cout<<"2\n";
     std::vector <float> edge_or_surface;
    std::vector <int> selected_indexes;
     LidarPointSelector lps(expected_points, intensity_bound, depth_bound, distance_bound, beam_num);
     pcl::PointCloud<pcl::Normal>::Ptr normals_out (new pcl::PointCloud<pcl::Normal>);
     pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_edge (new pcl::PointCloud<pcl::PointXYZI>);
     pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_surface (new pcl::PointCloud<pcl::PointXYZI>);
     lps.edge_detection(pc, pc_out_edge, output_depth_grad, output_intenstity_grad, selected_indexes);    
     //lps.edge_detection(pc, pc_out_edge, output_depth_grad, output_intenstity_grad);    
     lps.legoloam_point_selector(pc, pc_out_surface, edge_or_surface, selected_indexes);    
     *pc_out += *pc_out_edge;
     *pc_out += *pc_out_surface;

     normals_out = compute_pcd_normals(pc_out, 1.0);

    num_points_ = pc_out->size();
#endif


     
#if !defined(IS_USING_LOAM) && !defined(IS_USING_NORMALS)
     random_surface_with_edges(pc, expected_points, intensity_bound, depth_bound, distance_bound, beam_num,
                               output_depth_grad, output_intenstity_grad, selected_indexes);
     num_points_ = selected_indexes.size();

#endif     

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

      // if (i < 100) std::cout<<"intensity is "<< pc_out->points[i].intensity<<std::endl;

#ifdef IS_USING_NORMALS      
      normals_(i,0) = normals_out->points[i].normal_x;
      normals_(i,1) = normals_out->points[i].normal_y;
      normals_(i,2) = normals_out->points[i].normal_z;
#endif      

    }

    //#if  defined(IS_USING_COVARIANCE)  && defined(__CUDACC__)
    //std::cout<<"compute covariance\n";
    //compute_covariance(*pc, selected_indexes);

    
    std::cout<<"Construct Cvo PointCloud, num of points is "<<num_points_<<" from "<<pc->size()<<" input points "<<std::endl;    
    write_to_intensity_pcd("kitti_lidar.pcd");

  }


  CvoPointCloud::CvoPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pc, const std::vector<int> & semantic,
                               int num_classes, int target_num_points, int beam_num) {

    write_all_to_label_pcd("kitti_semantic_lidar_pre.pcd", *pc, num_classes, semantic);

    int expected_points = target_num_points;
    double intensity_bound = 0.4;
    double depth_bound = 4.0;
    std::vector<int> selected_indexes;
    double distance_bound = 40.0;
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out (new pcl::PointCloud<pcl::PointXYZI>);
    std::vector <double> output_depth_grad;
    std::vector <double> output_intenstity_grad;
    std::vector <int> semantic_out;

    std::cout<<"construct semantic lidar CvoPointCloud...\n";

#if  !defined(IS_USING_LOAM)  && defined(IS_USING_NORMALS)
    std::cout<<"not using loam, using normals"<<std::endl;
    pcl::PointCloud<pcl::Normal>::Ptr normals_out (new pcl::PointCloud<pcl::Normal>);
    edge_detection(pc, semantic, expected_points, intensity_bound, depth_bound, distance_bound, beam_num,
                   pc_out, output_depth_grad, output_intenstity_grad, selected_indexes, normals_out, semantic_out);
    
#endif    

#if defined(IS_USING_LOAM) && !defined(IS_USING_NORMALS)
    std::cout<<"using loam, not using normals"<<std::endl;
    std::vector <float> edge_or_surface;
    LidarPointSelector lps(expected_points, intensity_bound, depth_bound, distance_bound, beam_num);

    // running edge detection + lego loam point selection
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_edge (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_surface (new pcl::PointCloud<pcl::PointXYZI>);
    lps.edge_detection(pc, semantic, pc_out_edge, output_depth_grad, output_intenstity_grad, selected_indexes, semantic_out);   
    lps.legoloam_point_selector(pc, semantic, pc_out_surface, edge_or_surface, selected_indexes, semantic_out);    
    *pc_out += *pc_out_edge;
    *pc_out += *pc_out_surface;
    num_points_ = pc_out->size();
    assert(num_points_ == selected_indexes.size());

    //for (int i = 0; i < pc_out_surface->size() ; i++) {
    //    std::cout<<"pc_out_surface pointcloud (index,x,y,z,i)=("<<i<<", "<<pc_out_surface->points[i].x<<", "<<pc_out_surface->points[i].y<<", "<<pc_out_surface->points[i].z<<", "<<pc_out_surface->points[i].intensity<<")"<<std::endl;
    //}


    pcl::io::savePCDFileASCII("pc_out_edge.pcd", *pc_out_edge);
    pcl::io::savePCDFileASCII("pc_out_legoloam.pcd", *pc_out_surface);
    
#endif

#if defined(IS_USING_LOAM) && defined(IS_USING_NORMALS)
    std::cout<<"using loam and normals\n";
    std::vector <float> edge_or_surface;
    LidarPointSelector lps(expected_points, intensity_bound, depth_bound, distance_bound, beam_num);
    pcl::PointCloud<pcl::Normal>::Ptr normals_out (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_edge (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_surface (new pcl::PointCloud<pcl::PointXYZI>);
    lps.edge_detection(pc, semantic, pc_out_edge, output_depth_grad, output_intenstity_grad, semantic_out);    
    lps.legoloam_point_selector(pc, semantic, pc_out_surface, edge_or_surface, selected_indexes, semantic_out);    
    *pc_out += *pc_out_edge;
    *pc_out += *pc_out_surface;

    normals_out = compute_pcd_normals(pc_out, 1.0);
    
       

#endif

#if !defined(IS_USING_LOAM) && !defined(IS_USING_NORMALS)
    std::cout<<"not using loam and not using normals\n";
    // LidarPointSelector lps(expected_points, intensity_bound, depth_bound, distance_bound, beam_num);	
    // lps.edge_detection(pc, semantic, pc_out, output_depth_grad, output_intenstity_grad, semantic_out);
    edge_detection(pc, semantic, expected_points, intensity_bound, depth_bound, distance_bound, beam_num,
                   pc_out, output_depth_grad, output_intenstity_grad, selected_indexes, semantic_out);

#endif     


    // fill in class members
    num_points_ = selected_indexes.size();
    num_classes_ = num_classes;
    
    // features_ = Eigen::MatrixXf::Zero(num_points_, 1);
    feature_dimensions_ = 1;
    features_.resize(num_points_, feature_dimensions_);
    labels_.resize(num_points_, num_classes_);
#ifdef IS_USING_NORMALS    
    normals_.resize(num_points_,3);
#endif
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

#ifdef IS_USING_NORMALS      
      normals_(i,0) = normals_out->points[i].normal_x;
      normals_(i,1) = normals_out->points[i].normal_y;
      normals_(i,2) = normals_out->points[i].normal_z;
#endif      

    }
    std::cout<<"Construct Cvo PointCloud, num of points is "<<num_points_<<" from "<<pc->size()<<" input points "<<std::endl;
    write_to_label_pcd("kitti_semantic_lidar.pcd");
    write_to_intensity_pcd("kitti_intensity_lidar.pcd");
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
    for (int i = 0; i < num_points_; i++) {
      pcl::PointXYZRGB p;
      p.x = positions_[i](0);
      p.y = positions_[i](1);
      p.z = positions_[i](2);
      
      uint8_t b = static_cast<uint8_t>(std::min(255, (int)(features_(i,0) * 255) ) );
      uint8_t g = static_cast<uint8_t>(std::min(255, (int)(features_(i,1) * 255) ) );
      uint8_t r = static_cast<uint8_t>(std::min(255, (int)(features_(i,2) * 255)));
      uint32_t rgb = ((uint32_t) r << 16 |(uint32_t) g << 8  | (uint32_t) b ) ;
      p.rgb = *reinterpret_cast<float*>(&rgb);
      pc.push_back(p);
    }
    pcl::io::savePCDFileASCII(name ,pc);  
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
