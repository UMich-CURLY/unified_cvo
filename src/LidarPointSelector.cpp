
#include <iostream>
#include <cstring>
#include <cmath>

#include <vector>
#include <string>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter_indices.h>
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
                                          std::vector <double> & output_intenstity_grad,
                                          std::vector <int> & selected_indexes) {

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*pc_in, indices);
    int num_points = indices.size();
    int previous_quadrant = get_quadrant(pc_in->points[indices[0]]);
    int ring_num = 0;
    for(int i = 1; i<num_points-1; i++) {      
      int quadrant = get_quadrant(pc_in->points[indices[i]]);
      if(quadrant == 1 && previous_quadrant == 4 && ring_num < _num_beams-1){
        ring_num += 1;
        continue;
      }

      // select points
      const auto& point_l = pc_in->points[indices[i-1]];
      const auto& point = pc_in->points[indices[i]];
      const auto& point_r = pc_in->points[indices[i+1]];
      
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
          pc_out->push_back(pc_in->points[indices[i]]);
          output_depth_grad.push_back(depth_grad);
          output_intenstity_grad.push_back(intenstity_grad);
          selected_indexes.push_back(indices[i]);
      }

      previous_quadrant = quadrant;      
    }
  }

  void LidarPointSelector::edge_detection(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in,
                                          const std::vector<int> & semantic_in,
                                          pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out,
                                          std::vector <double> & output_depth_grad,
                                          std::vector <double> & output_intenstity_grad,
                                          std::vector <int> & selected_indexes,
                                          std::vector<int> & semantic_out) {

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*pc_in, *pc_in, indices);
    int num_points = pc_in->points.size();
    int previous_quadrant = get_quadrant(pc_in->points[0]);
    int ring_num = 0;

    //for(int i = 1; i<num_points; i++) {
    for (auto i : indices) {
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
          selected_indexes.push_back(i);
          //std::cout<<" in edge detection , point "<<point.x<<", "<<point.y<<", "<<point.z<<", label "<<point.intensity<<std::endl;
      }

      previous_quadrant = quadrant;      
    }
  }

  void LidarPointSelector::loam_point_selector(pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudIn,
                                                pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out,
                                                std::vector <float> & edge_or_surface,
                                                std::vector <int> & selected_indexes) {
    //using loam_velodyne's functions
    LoamScanRegistration lsr(-24.9f, 2, 64);
    lsr.process(*laserCloudIn, pc_out, edge_or_surface, selected_indexes);

    // pcl::io::savePCDFile("raw_input.pcd", *laserCloudIn);
    // pcl::io::savePCDFile("loam_pointselection.pcd", *pc_out);
  }

  void LidarPointSelector::loam_point_selector(pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudIn,
                                                const std::vector<int> & semantic_in,
                                                pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out,
                                                std::vector <float> & edge_or_surface,
                                                std::vector<int> & semantic_out,
                                                std::vector <int> & selected_indexes) {
    //using loam_velodyne's functions
    LoamScanRegistration lsr(-24.9f, 2, 64);
    lsr.process(*laserCloudIn, pc_out, edge_or_surface, selected_indexes);

    for(int i = 0; i<pc_out->points.size(); i++) {
      semantic_out.push_back(semantic_in[selected_indexes[i]]);
    }

    // pcl::io::savePCDFile("raw_input.pcd", *laserCloudIn);
    // pcl::io::savePCDFile("loam_pointselection.pcd", *pc_out);
  }

  void LidarPointSelector::legoloam_point_selector(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in,
                                                  pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out,
                                                  std::vector <float> & edge_or_surface,
                                                  std::vector <int> & selected_indexes_out) {
    //using LeGO-LOAM's functions

    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr pc_in_const = pc_in;

    LeGoLoamPointSelection lego_loam;
    lego_loam.cloudHandler(pc_in_const, pc_out, edge_or_surface, selected_indexes_out);

    pcl::io::savePCDFile("raw_input.pcd", *pc_in);
    pcl::io::savePCDFile("loam_pointselection.pcd", *pc_out);
  }

  void LidarPointSelector::legoloam_point_selector(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in,
                                                  const std::vector<int> & semantic_in,
                                                  pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out,
                                                  std::vector <float> & edge_or_surface,
                                                  std::vector <int> & selected_indexes_out,
                                                  std::vector<int> & semantic_out) {
    //using LeGO-LOAM's functions

    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr pc_in_const = pc_in;
    
    std::vector<int> selected_indexes;
    LeGoLoamPointSelection lego_loam;
    lego_loam.cloudHandler(pc_in_const, pc_out, edge_or_surface, selected_indexes);
    // pcl::PointCloud<pcl::PointXYZI>::Ptr pc_selected (new pcl::PointCloud<pcl::PointXYZI>);


    for(int i = 0; i< selected_indexes.size(); i++) {
      semantic_out.push_back(semantic_in[selected_indexes[i]]);
      selected_indexes_out.push_back(selected_indexes[i]);
      // pc_selected->push_back(pc_in->points[selected_indexes[i]]);
    }

    // pcl::io::savePCDFile("raw_input.pcd", *laserCloudIn);
    // pcl::io::savePCDFile("loam_pointselection.pcd", *pc_out);
    // pcl::io::savePCDFile("pc_selected.pcd", *pc_selected);
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
