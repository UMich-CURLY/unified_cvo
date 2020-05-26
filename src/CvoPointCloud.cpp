#include <string>
#include <fstream>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <algorithm>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include "utils/CvoPointCloud.hpp"
#include "utils/StaticStereo.hpp"
#include "utils/CvoPixelSelector.hpp"
#include "mapping/bkioctomap.h"
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
    // for (int h = 0; h < left_image.color().cols; h++){
    //   for (int w = 0; w < left_image.color().rows; w++){
    //     Vec2i uv;
    //     uv << h , w;
    //     output_uv.push_back(uv);
    //   }
    // }
    

    std::vector<int> good_point_ind;
    int h = left_image.color().rows;
    int w = left_image.color().cols;
    for (int i = 0; i < output_uv.size(); i++) {
      auto uv = output_uv[i];
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
      int u = output_uv[good_point_ind[i]](0);
      int v = output_uv[good_point_ind[i]](1);
      cv::Vec3b avg_pixel = left_image.color().at<cv::Vec3b>(v,u);
      auto & gradient = left_image.gradient()[v * w + u];
      //std::cout<<"\nopencv before pattern: "<<left_image.color().at<cv::Vec3b>(v,u)<<std::endl;
      //cv::Vec3f avg_pixel = avg_pixel_color_pattern(left_image.color(), u, v, w);
      //std::cout<<".  after pattern "<<avg_pixel.transpose()<<std::endl;
      //features_.block(i, 0, 1, 3) = avg_pixel.transpose() / 255.0;
      features_(i,0) = ((float)(avg_pixel [0]) )/255.0;
      features_(i,1) = ((float)(avg_pixel[1]) )/255.0;
      features_(i,2) = ((float)(avg_pixel[2]) )/255.0;
      features_(i,3) = gradient(0)/ 500.0 + 0.5;
      features_(i,4) = gradient(1)/ 500.0 + 0.5;

      if (num_classes_) {
        labels_.row(i) = Eigen::Map<const VecXf_row>((left_image.semantic_image().data()+ (v * w + u)*num_classes_), num_classes_);
        int max_class = 0;
        labels_.row(i).maxCoeff(&max_class);
        //float sum_row = labels_.row(i).sum();
        //labels_.row(i) = (labels_.row(i) / sum_row).eval();
        /*
        if (i == 0 || i == 1) {
          std::cout<<"Raw: ";
          for (int k = 0; k < num_classes_; k++)
            std::cout<<left_image.semantic_image()[k+ num_classes_ * (v * w + u)]<<", ";
          std::cout<<"\n";
          std::cout<<"labels_() after copy: ";
          std::cout<<labels_.row(i)<<"\n";
        }
        */
      }

    }
    //  write_to_label_pcd("labeled_input.pcd");
    // write_to_color_pcd("test.pcd");
  }

  CvoPointCloud::CvoPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pc) {
    int expected_points = 5000;
    double intensity_bound = 0.4;
    double depth_bound = 4.0;
    double distance_bound = 55.0;
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::Normal>::Ptr normals_out (new pcl::PointCloud<pcl::Normal>);
    std::vector <double> output_depth_grad;
    std::vector <double> output_intenstity_grad;
    edge_detection(pc, expected_points, intensity_bound, depth_bound, distance_bound, pc_out,\
                   output_depth_grad, output_intenstity_grad, normals_out);
     
    
    /*
      ----------visualize selected points and normals-----------
    */
    // pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    // viewer.addPointCloudNormals<pcl::PointXYZI,pcl::Normal>(pc_out, normals_out,1,0.1, "normals1");
    // viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "normals1");
    // viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "normals1");
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> rgb2 (pc_out, 0, 255, 0); //This will display the point cloud in green (R,G,B)
    // viewer.addPointCloud<pcl::PointXYZI> (pc_out, rgb2, "cloud_RGB2");
    // while (!viewer.wasStopped ())
    // {
    //   viewer.spinOnce ();
    // }

    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals_temp(new pcl::PointCloud<pcl::PointNormal>);
    cloud_with_normals_ = cloud_with_normals_temp;
    pcl::copyPointCloud(*pc_out, *cloud_with_normals_);
    pcl::copyPointCloud(*normals_out, *cloud_with_normals_);
    // pcl::io::savePCDFileASCII("test.pcd", *cloud_with_normals_);

    
    // fill in class members
    num_points_ = pc_out->size();
    num_classes_ = 0;
    
    // features_ = Eigen::MatrixXf::Zero(num_points_, 1);
    feature_dimensions_ = 1;
    features_.resize(num_points_, feature_dimensions_);
    normals_.resize(num_points_,3);

    for (int i = 0; i < num_points_ ; i++) {
      Vec3f xyz;
      xyz << pc_out->points[i].x, pc_out->points[i].y, pc_out->points[i].z;
      positions_.push_back(xyz);
      features_(i, 0) = pc_out->points[i].intensity;      
      normals_(i,0) = normals_out->points[i].normal_x;
      normals_(i,1) = normals_out->points[i].normal_y;
      normals_(i,2) = normals_out->points[i].normal_z;
    }

    // std::cout<<normals_<<std::endl;

    // write_to_intensity_pcd("kitti_lidar.pcd");
  }

  CvoPointCloud::CvoPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pc, const std::vector<int> & semantic) {
    int expected_points = 5000;
    double intensity_bound = 0.4;
    double depth_bound = 4.0;
    double distance_bound = 55.0;
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out (new pcl::PointCloud<pcl::PointXYZI>);
    std::vector <double> output_depth_grad;
    std::vector <double> output_intenstity_grad;
    std::vector <int> semantic_out;
    edge_detection(pc, semantic, expected_points, intensity_bound, depth_bound, distance_bound, pc_out, output_depth_grad, output_intenstity_grad, semantic_out);
    // fill in class members
    num_points_ = pc_out->size();
    num_classes_ = 19; //TODO: get it from input

    feature_dimensions_ = 1;
    features_.resize(num_points_, feature_dimensions_);
    labels_.resize(num_points_, num_classes_);

    for (int i = 0; i < num_points_ ; i++) {
      Vec3f xyz;
      xyz << pc_out->points[i].x, pc_out->points[i].y, pc_out->points[i].z;
      positions_.push_back(xyz);
      features_(i, 0) = pc_out->points[i].intensity; 

      // add one-hot semantic labels
      VecXf_row one_hot_label;
      one_hot_label = VecXf_row::Zero(1,num_classes_);
      one_hot_label[semantic_out[i]] = 1;

      labels_.row(i) = one_hot_label;
      int max_class = 0;
      labels_.row(i).maxCoeff(&max_class);
    }
  }
  
  CvoPointCloud::CvoPointCloud(const semantic_bki::SemanticBKIOctoMap * map,
                               const int num_classes) {
    num_classes_ = num_classes;
    int num_point_counter = 0;
    std::vector<std::vector<float> > features;
    std::vector<std::vector<float> > labels;
    positions_.reserve(65536);
    features.reserve(65536);
    labels.reserve(65536);
    feature_dimensions_ = 5;
    
    for (auto it = map->begin_leaf(); it != map->end_leaf(); ++it) {
      if (it.get_node().get_state() == semantic_bki::State::OCCUPIED) {
        // position
        semantic_bki::point3f  p = it.get_loc();
        Vec3f xyz;
        xyz << p.x(), p.y(), p.z();
        positions_.push_back(xyz);
               
        // features
        if(feature_dimensions_==5){
          std::vector<float> feature(5, 0);
          it.get_node().get_features(feature);
          features.push_back(feature);
        }
        else if(feature_dimensions_==1){
          std::vector<float> feature_1(1, 0);
          it.get_node().get_features(feature_1);
          features.push_back(feature_1);
        }
        
        // labels
        std::vector<float> label(num_classes_, 0);
        it.get_node().get_occupied_probs(label);
        labels.push_back(label);
        num_point_counter++;
      }
    }
      
    num_points_ = num_point_counter ;
    features_.resize(num_points_, feature_dimensions_);
    labels_.resize(num_points_, num_classes);

    for (int i = 0; i < num_points_; i++) {
      //memcpy(labels_.data()+ num_classes * sizeof(float) * i, labels[i].data(), num_classes * sizeof(float));
      labels_.row(i) = Eigen::Map<VecXf_row>(labels[i].data(), num_classes);
      if(feature_dimensions_==5){
        features_.row(i) = Eigen::Map<Vec5f_row>(features[i].data());
      }
      else if(feature_dimensions_==1){
        features_(i,0) = *features[i].data();
      }

    }
    //std::cout<<"Read labels from map:\nlabel" << labels_.row(0)<<"\n"<<labels_.row(num_points_-1)<<", color: ";
    //std::cout<< features_.row(0)<<"\n"<<features_.row(num_points_-1)<<"\n";
  }

  CvoPointCloud::CvoPointCloud(){}
  CvoPointCloud::~CvoPointCloud() {
    
    
  }

  int CvoPointCloud::read_cvo_pointcloud_from_file(const std::string & filename) {
    std::ifstream infile(filename);
    feature_dimensions_ = 5;
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
    std::cout << "Finished write to intensity pcd" << std::endl; 
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
