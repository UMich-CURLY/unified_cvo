#include "utils/PointSelection.hpp"
#include "utils/RawImage.hpp"
#include "utils/ImageStereo.hpp"
#include "utils/ImageRGBD.hpp"
#include "utils/CvoPixelSelector.hpp"
#include "utils/StaticStereo.hpp"
#include "utils/LidarPointSelector.hpp"
#include "utils/LidarPointType.hpp"


namespace cvo {

  // filter out sky or too-far-away pixels
  static bool is_good_point(const Vec3f & xyz, const Vec2i uv, int h, int w ) {
    int u = uv(0);
    int v = uv(1);
    if ( u < 2 || u > w -2 || v < 100 || v > h-30 )
      return false;

    if (xyz.norm() >= 55) // 55
      return false;

    return true;
  }

  void stereo_surface_sampling(const cv::Mat & left_gray,
                                const std::vector<Vec2i, Eigen::aligned_allocator<Vec2i>> & dso_selected_uv,
                                bool is_using_canny,
                                bool is_using_uniform_rand,
                                bool is_using_orb,
                                int expected_points,
                                // output
                                std::vector<float> & edge_or_surface,
                                std::vector<Vec2i, Eigen::aligned_allocator<Vec2i>> & final_selected_uv
              
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
    std::vector<Vec2i, Eigen::aligned_allocator<Vec2i>> tmp_uvs_canny, tmp_uvs_surface;
    if (is_using_canny) {
      for (int r = 0 ; r < left_gray.rows; r++) {
        for (int c = 0; c < left_gray.cols; c++) {
          // using Canny
          if (is_using_canny &&  detected_edges.at<uint8_t>(r, c) > 0)  {
            // selected_inds_map[r * left_gray.cols + c] = EDGE;
            tmp_uvs_canny.push_back(Vec2i(c, r));
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
            tmp_uvs_surface.push_back(Vec2i(c, r));
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
        final_selected_uv.push_back(Eigen::Vector2i(c, r));
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

  void select_points_from_image(const RawImage & left_image,
                                       PointCloudType pt_type,
                                       PointSelectionMethod pt_selection_method,
                                       // result
                                       std::vector<float> & edge_or_surface,
                                       std::vector<Vec2i, Eigen::aligned_allocator<Vec2i>> & output_uv

                                       )  {
    cv::Mat left_gray;
    if (left_image.channels() == 1)
      left_gray = left_image.image();
    else {
      cv::cvtColor(left_image.image(), left_gray, cv::COLOR_BGR2GRAY);
    }
    /*****************************************/
    if (pt_selection_method == CV_FAST) {
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
        Vec2i xy;
        xy(0) = (int)kp.pt.x;
        xy(1) = (int)kp.pt.y;
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
          cv::circle(heatmap, cv::Point( output_uv[i](0), output_uv[i](1) ), 1, cv::Scalar(255, 0 ,0), 1);
        cv::imwrite("FAST_selected_pixels.png", heatmap);
      }
    } 
    /*****************************************/
    // using DSO semi dense point selector
    else if (pt_selection_method == DSO_EDGES) {
      int expected_points = 10000;
      dso_select_pixels(left_image,
                        expected_points,
                        output_uv);
      edge_or_surface.resize(output_uv.size() * 2);
      for (int i = 0; i < output_uv.size(); i++) {
        edge_or_surface[i*2] = 0.9;
        edge_or_surface[i*2 +1]=0.1;
      }
      
    }
    //******************************************/
    // using canny or random point selection
    else if (pt_selection_method == CANNY_EDGES) {
      //std::vector<bool> selected_inds_map;
      std::vector<Vec2i, Eigen::aligned_allocator<Vec2i>> final_selected_uv;
      int expected_points = 10000;
      stereo_surface_sampling(left_gray, output_uv, true, true, true, expected_points,
                              edge_or_surface, output_uv);


      
      
    }
    /* edge only */
    else if (pt_selection_method == EDGES_ONLY) {
      //std::vector<bool> selected_inds_map;
      std::vector<Vec2i, Eigen::aligned_allocator<Vec2i>> final_selected_uv;
      int expected_points = 10000;
      stereo_surface_sampling(left_gray, output_uv, true, false, false, expected_points,
                              edge_or_surface, output_uv);
      
    }
    
    /********************************************/
    // using full point cloud
    else if (pt_selection_method == FULL) {

      output_uv.clear();
      for (int h = 0; h < left_image.cols(); h++){
        for (int w = 0; w < left_image.rows(); w++){
          Vec2i uv;
          uv << h , w;
          output_uv.push_back(uv);
          edge_or_surface.push_back(0.5);
          edge_or_surface.push_back(0.5);

        }
      }
    } else {
      std::cerr<<"This point selection method is not implemented.\n";
      return;
    }

  }

  // functions to support point selection and output pcl pointcloud
  void pointSelection(const ImageStereo& raw_image,
                      const Calibration& calib,
                      pcl::PointCloud<CvoPoint>& out_pc,
                      PointSelectionMethod pt_selection_method
                      ) {
      out_pc.points.clear();
      const cv::Mat& left_image = raw_image.image();
      const std::vector<float>& left_disparity = raw_image.disparity();
      std::vector<Vec2i, Eigen::aligned_allocator<Vec2i>> output_uv;
      std::vector<float> geometry;
      select_points_from_image(left_image, STEREO, pt_selection_method,
                              geometry,
                              output_uv); 
      auto & pre_depth_selected_ind = output_uv;
      std::vector<int> good_point_ind;
      int h = raw_image.rows();
      int w = raw_image.cols();
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
          int u = uv(0);
          int v = uv(1);
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
          out_pc.points.emplace_back(xyz(0), xyz(1), xyz(2));
          //positions_[i] = xyz;
          //std::cout<<"xyz: "<<xyz.transpose()<<std::endl;
          // TL: TODO check geometric_type usage
          // geometric_types_.push_back(geometry[i*2]);
          // geometric_types_.push_back(geometry[i*2+1]);
        }
      }
      for (int i = 0; i < out_pc.points.size(); i++) {
        int u = pre_depth_selected_ind[good_point_ind[i]](0);
        int v = pre_depth_selected_ind[good_point_ind[i]](1);
        std::vector<float> features(raw_image.channels() + 2, 0.0f);
        if (raw_image.channels() == 3) {
          cv::Vec3b avg_pixel = raw_image.image().at<cv::Vec3b>(v,u);
          float gradient_0 = raw_image.gradient()[v * w + u];
          float gradient_1 = raw_image.gradient()[v * w + u + 1];
          features[0] = ((float)(avg_pixel[0])) / 255.0;
          features[1] = ((float)(avg_pixel[1])) / 255.0;
          features[2] = ((float)(avg_pixel[2])) / 255.0;
          features[3] = gradient_0 / 500.0 + 0.5;
          features[4] = gradient_1 / 500.0 + 0.5;
        } else if (raw_image.channels() == 1) {
          uint8_t avg_pixel = raw_image.image().at<uint8_t>(v,u);
          float gradient_0 = raw_image.gradient()[v * w + u];
          float gradient_1 = raw_image.gradient()[v * w + u + 1];
          features[0] = ((float)(avg_pixel)) / 255.0;
          features[1] = gradient_0 / 500.0 + 0.5;
          features[2] = gradient_1 / 500.0 + 0.5;
        } else {
          std::cerr << "CvoPointCloud: channel unknown\n";
        }
        // fill into pcl pc
        if (FEATURE_DIMENSIONS >= 3) {
          out_pc[i].r = (uint8_t)std::min(255.0, (features[0] * 255.0));
          out_pc[i].g = (uint8_t)std::min(255.0, (features[1] * 255.0));
          out_pc[i].b = (uint8_t)std::min(255.0, (features[2] * 255.0));
        }
        for (int j = 0; j < FEATURE_DIMENSIONS; j++) {
          out_pc[i].features[j] = features[j];
        }
        if (raw_image.num_classes() > 0) {
          VecXf_row seman_distr = Eigen::Map<const VecXf_row>((raw_image.semantic_image().data() + (v * w + u) * raw_image.num_classes()), raw_image.num_classes());
          seman_distr.maxCoeff(&out_pc[i].label);
          for (int j = 0; j < FEATURE_DIMENSIONS; j++)
            out_pc[i].label_distribution[j] = seman_distr(j);
        }
      }
  }

  template <typename DepthType>
  void pointSelection(const ImageRGBD<DepthType>& raw_image,
                      const Calibration& calib,
                      pcl::PointCloud<CvoPoint>& out_pc,
                      PointSelectionMethod pt_selection_method) {
    const cv::Mat & rgb_raw_image = raw_image.image();
    const std::vector<DepthType> & depth_image = raw_image.depth_image();
 
    std::vector<Vec2i, Eigen::aligned_allocator<Vec2i>> output_uv;
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
      auto uv = output_uv[i];
      int u = uv(0);
      int v = uv(1);
      Vec3f xyz;
 
      //uint16_t dep = depth_image.at<uint16_t>(cv::Point(u, v));
      DepthType dep = depth_image[v * w + u];
        
      if(dep!=0 && !isnan(dep)){

        // construct depth
        xyz(2) = dep/calib.scaling_factor();

        //if (xyz(2) > 15.0)
        //  continue;
        
        // construct x and y
        xyz(0) = (u-intrinsic(0,2)) * xyz(2) / intrinsic(0,0);
        xyz(1) = (v-intrinsic(1,2)) * xyz(2) / intrinsic(1,1);
        
        // check for labels
        if (raw_image.num_classes()) {
          auto labels = Eigen::Map<const VecXf_row>((raw_image.semantic_image().data()+ (v * w + u)*raw_image.num_classes()), raw_image.num_classes() );
          int max_class = 0;
          labels.maxCoeff(&max_class);
          if(max_class == 10)// exclude unlabeled points
            continue;
        }

        // add point to pcd
        good_point_ind.push_back(i);
        out_pc.points.emplace_back(xyz(0), xyz(1), xyz(2));
        // TL: TODO check geometric_type usage
      }
    }
    for (int i = 0; i < out_pc.points.size(); i++) {
      int u = output_uv[good_point_ind[i]](0);
      int v = output_uv[good_point_ind[i]](1);
      std::vector<float> features(raw_image.channels() + 2, 0.0f);
      if (raw_image.channels() == 3) {
        // cv::Vec3b avg_pixel = raw_image.image().at<cv::Vec3b>(v,u);
        cv::Vec3b avg_pixel = rgb_raw_image.at<cv::Vec3b>(v, u);
        float gradient_0 = raw_image.gradient()[v * w + u];
        float gradient_1 = raw_image.gradient()[v * w + u + 1];
        features[0] = ((float)(avg_pixel[0])) / 255.0;
        features[1] = ((float)(avg_pixel[1])) / 255.0;
        features[2] = ((float)(avg_pixel[2])) / 255.0;
        features[3] = gradient_0 / 500.0 + 0.5;
        features[4] = gradient_1 / 500.0 + 0.5;
      } else if (raw_image.channels() == 1) {
        // uint8_t avg_pixel = raw_image.image().at<uint8_t>(v,u);
        uint8_t avg_pixel = rgb_raw_image.at<uint8_t>(v, u);
        float gradient_0 = raw_image.gradient()[v * w + u];
        float gradient_1 = raw_image.gradient()[v * w + u + 1];
        features[0] = ((float)(avg_pixel)) / 255.0;
        features[1] = gradient_0 / 500.0 + 0.5;
        features[2] = gradient_1 / 500.0 + 0.5;
      } else {
        std::cerr << "CvoPointCloud: channel unknown\n";
      }
      // fill into pcl pc
      if (FEATURE_DIMENSIONS >= 3) {
        out_pc[i].r = (uint8_t)std::min(255.0, (features[0] * 255.0));
        out_pc[i].g = (uint8_t)std::min(255.0, (features[1] * 255.0));
        out_pc[i].b = (uint8_t)std::min(255.0, (features[2] * 255.0));
      }
      for (int j = 0; j < FEATURE_DIMENSIONS; j++) {
        out_pc[i].features[j] = features[j];
      }
      if (raw_image.num_classes() > 0) {
        VecXf_row seman_distr = Eigen::Map<const VecXf_row>((raw_image.semantic_image().data() + (v * w + u) * raw_image.num_classes()), raw_image.num_classes());
        seman_distr.maxCoeff(&out_pc[i].label);
        for (int j = 0; j < FEATURE_DIMENSIONS; j++)
          out_pc[i].label_distribution[j] = seman_distr(j);
      }
    }
  }

  void pointSelection(pcl::PointCloud<pcl::PointXYZI>::Ptr pc,
                      int target_num_points,
                      int beam_num,
                      pcl::PointCloud<CvoPoint>& out_pc,
                      PointSelectionMethod pt_selection_method) {
    int expected_points = target_num_points;
    double intensity_bound = 0.4;
    double depth_bound = 4.0;
    double distance_bound = 40.0;
    std::vector <int> selected_indices;
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out (new pcl::PointCloud<pcl::PointXYZI>); 

    std::vector <double> output_depth_grad;
    std::vector <double> output_intenstity_grad;

    if (pt_selection_method == LOAM) {
      std::vector <float> edge_or_surface;
      LidarPointSelector lps(expected_points, intensity_bound, depth_bound, distance_bound, beam_num);

      // running edge detection + lego loam point selection
      pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_edge (new pcl::PointCloud<pcl::PointXYZI>);
      pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_surface (new pcl::PointCloud<pcl::PointXYZI>);
      lps.edge_detection(pc, pc_out_edge, output_depth_grad, output_intenstity_grad, selected_indices);
      lps.legoloam_point_selector(pc, pc_out_surface, edge_or_surface, selected_indices);    

      std::cout<<"pc_out size is "<<pc_out->size()<<std::endl;
      std::cout << "\nList of selected lego indexes " << pc_out_surface->size()<< std::endl;
      for(int i=0; i<10; i++){
        std::cout << selected_indices[i] << " ";
      }
      std::cout<<std::flush;

    } else if (pt_selection_method == RANDOM) {

      random_surface_with_edges(pc, expected_points, intensity_bound, depth_bound, distance_bound, beam_num,
                                output_depth_grad, output_intenstity_grad, selected_indices);
    } else {
      std::cerr<<" This point selection method is not implemented\n";
      return;
    }
    

    for (int i = 0; i < selected_indices.size(); i++) {
      int idx = selected_indices[i];
      out_pc.points.emplace_back(pc->points[idx].x, pc->points[idx].y, pc->points[idx].z);
      out_pc[i].features[0] = pc->points[idx].intensity;
      // TL: TODO check geometric_type usage
    }

    std::cout<<"Construct Cvo PointCloud, num of points is " << out_pc.size() << " from "<< pc->size() <<" input points " << std::endl;
  }

  void pointSelection(pcl::PointCloud<pcl::PointXYZI>::Ptr pc,
                      const std::vector<int> & semantic,
                      int num_classes,
                      int target_num_points,
                      int beam_num,
                      pcl::PointCloud<CvoPoint>& out_pc,
                      PointSelectionMethod pt_selection_method) {
    int expected_points = target_num_points;
    double intensity_bound = 0.4;
    double depth_bound = 4.0;
    std::vector<int> selected_indices;
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
      lps.edge_detection(pc, semantic, pc_out_edge, output_depth_grad, output_intenstity_grad, selected_indices, semantic_out);  
      *pc_out += *pc_out_edge;
      // lego loam surface
      pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_surface (new pcl::PointCloud<pcl::PointXYZI>); 
      lps.legoloam_point_selector(pc, semantic, pc_out_surface, edge_or_surface, selected_indices, semantic_out);
      std::cout<<"semantic out size = "<<semantic_out.size()<<std::endl;
      *pc_out += *pc_out_surface;
    
    } else if (pt_selection_method == LIDAR_EDGES) {
      edge_detection(pc, semantic, expected_points, intensity_bound, depth_bound, distance_bound, beam_num,
                     pc_out, output_depth_grad, output_intenstity_grad, selected_indices, semantic_out);
    } else {
      std::cerr<<"The point selection method is not implemented\n";
      return;
    }
    int actual_ids = 0;
    for (int i = 0; i < selected_indices.size(); i++) {
      int idx = selected_indices[i];
      if (semantic_out[i] == -1)
        continue;
      out_pc.points.emplace_back(pc->points[idx].x, pc->points[idx].y, pc->points[idx].z);
      out_pc[actual_ids].features[0] = pc->points[idx].intensity;

      // semantic labels
      VecXf_row one_hot_label;
      one_hot_label = VecXf_row::Zero(1, num_classes);
      one_hot_label[semantic_out[i]] = 1;
      out_pc[actual_ids].label = semantic_out[i];
      for (int j = 0; j < FEATURE_DIMENSIONS; j++)
        out_pc[actual_ids].label_distribution[j] = one_hot_label(j);
      
      // TL: TODO check geometric_type usage
      actual_ids++;
    }
  }
}