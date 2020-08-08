
#include <iostream>
#include <cstring>
#include <cmath>

#include <vector>
#include <string>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "utils/LidarPointSelector.hpp"
#include "utils/LoamScanRegistration.hpp"
#include "utils/LeGoLoamPointSelection.hpp"

namespace cvo
{

  LidarPointSelector::LidarPointSelector(int num_want,
                                        double intensity_bound, 
                                        double depth_bound,
                                        double distance_bound,
                                        int num_beams) :
    _num_want(num_want), 
    _intensity_bound(intensity_bound),
    _depth_bound(depth_bound),
    _distance_bound(distance_bound),
    _num_beams(num_beams)
  {

  }

  LidarPointSelector::~LidarPointSelector() {
  }

  void LidarPointSelector::edge_detection(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in,
                                          pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out,
                                          std::vector <double> & output_depth_grad,
                                          std::vector <double> & output_intenstity_grad) {

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*pc_in, *pc_in, indices);
    int num_points = pc_in->points.size();
    int previous_quadrant = get_quadrant(pc_in->points[0]);
    int ring_num = 0;

    for(int i = 1; i<num_points; i++) {      
      int quadrant = get_quadrant(pc_in->points[i]);
      if(quadrant == 1 && previous_quadrant == 4 && ring_num < _num_beams-1){
        ring_num += 1;
        continue;
      }

      // select points
      const auto& point_l = pc_in->points[i-1];
      const auto& point = pc_in->points[i];
      const auto& point_r = pc_in->points[i+1];
      
      double depth_grad = std::max((point_l.getVector3fMap()-point.getVector3fMap()).norm(),
                      (point.getVector3fMap()-point_r.getVector3fMap()).norm());
      
      double intenstity_grad = std::max(
                              std::abs( point_l.intensity - point.intensity ),
                              std::abs( point.intensity - point_r.intensity ));

      if( (intenstity_grad > _intensity_bound || depth_grad > _depth_bound) 
           && (point.intensity > 0.0) 
           && ((point.x!=0.0) && (point.y!=0.0) && (point.z!=0.0)) //){
           && (sqrt(point.x*point.x + point.y*point.y + point.z*point.z) < _distance_bound)){
          // std::cout << "points: " << point.x << ", " << point.y << ", " << point.z << ", " << point.intensity << std::endl;
          pc_out->push_back(pc_in->points[i]);
          output_depth_grad.push_back(depth_grad);
          output_intenstity_grad.push_back(intenstity_grad);
      }

      previous_quadrant = quadrant;      
    }

    // visualize
    // pcl::visualization::PCLVisualizer input_viewer ("Input Point Cloud Viewer");
    // input_viewer.addPointCloud<pcl::PointXYZI> (pc_in, "frame0");
    // while (!input_viewer.wasStopped ())
    // {
    //     input_viewer.spinOnce ();
    // }
    // pcl::visualization::PCLVisualizer output_viewer ("Output Point Cloud Viewer");
    // output_viewer.addPointCloud<pcl::PointXYZI> (pc_out, "frame0");
    // while (!output_viewer.wasStopped ())
    // {
    //     output_viewer.spinOnce ();
    // }
    
    // pcl::io::savePCDFile("raw_input.pcd", *pc_in);
    // pcl::io::savePCDFile("edge_detection.pcd", *pc_out);
  }

  void LidarPointSelector::edge_detection(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in,
                                          const std::vector<int> & semantic_in,
                                          pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out,
                                          std::vector <double> & output_depth_grad,
                                          std::vector <double> & output_intenstity_grad,
                                          std::vector<int> & semantic_out) {

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*pc_in, *pc_in, indices);
    int num_points = pc_in->points.size();
    int previous_quadrant = get_quadrant(pc_in->points[0]);
    int ring_num = 0;

    for(int i = 1; i<num_points; i++) {   
      if(semantic_in[i]==-1){
        // exclude unlabeled points

        continue;
      }   
      int quadrant = get_quadrant(pc_in->points[i]);
      if(quadrant == 1 && previous_quadrant == 4 && ring_num < _num_beams-1){
        ring_num += 1;
        continue;
      }

      // select points
      const auto& point_l = pc_in->points[i-1];
      const auto& point = pc_in->points[i];
      const auto& point_r = pc_in->points[i+1];
      
      double depth_grad = std::max((point_l.getVector3fMap()-point.getVector3fMap()).norm(),
                      (point.getVector3fMap()-point_r.getVector3fMap()).norm());
      
      double intenstity_grad = std::max(
                              std::abs( point_l.intensity - point.intensity ),
                              std::abs( point.intensity - point_r.intensity ));

      if( (intenstity_grad > _intensity_bound || depth_grad > _depth_bound) 
           && (point.intensity > 0.0) 
           && ((point.x!=0.0) && (point.y!=0.0) && (point.z!=0.0)) //){
           && (sqrt(point.x*point.x + point.y*point.y + point.z*point.z) < _distance_bound)){
          pc_out->push_back(pc_in->points[i]);
          output_depth_grad.push_back(depth_grad);
          output_intenstity_grad.push_back(intenstity_grad);
          semantic_out.push_back(semantic_in[i]);

          //std::cout<<" in edge detection , point "<<point.x<<", "<<point.y<<", "<<point.z<<", label "<<point.intensity<<std::endl;
      }

      previous_quadrant = quadrant;      
    }
  }

  void LidarPointSelector::loam_point_selector(pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudIn,
                                                pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out,
                                                std::vector <float> & edge_or_surface) {
    //using loam_velodyne's functions
    LoamScanRegistration lsr(-24.9f, 2, 64);
    lsr.process(*laserCloudIn, pc_out, edge_or_surface);

    // pcl::io::savePCDFile("raw_input.pcd", *laserCloudIn);
    // pcl::io::savePCDFile("loam_pointselection.pcd", *pc_out);
  }

  void LidarPointSelector::legoloam_point_selector(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in,
                                                  pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out,
                                                  std::vector <float> & edge_or_surface) {
    //using LeGO-LOAM's functions

    LeGoLoamPointSelection lego_loam;
    lego_loam.cloudHandler(pc_in, pc_out, edge_or_surface);

    // pcl::io::savePCDFile("raw_input.pcd", *laserCloudIn);
    // pcl::io::savePCDFile("loam_pointselection.pcd", *pc_out);
  }

  int LidarPointSelector::get_quadrant(pcl::PointXYZI point){
    int res = 0;
    /* because for kitti dataset lidar, we changed the coordinate...
    now.x = -raw.y;
    now.y = -raw.z;
    now.z = raw.x;
    */
    float x = point.z;
    float y = -point.x;

    if(x > 0 && y >= 0){res = 1;}
    else if(x <= 0 && y > 0){res = 2;}
    else if(x < 0 && y <= 0){res = 3;}
    else if(x >= 0 && y < 0){res = 4;}   

    return res;
  }


} // namespcae cvo