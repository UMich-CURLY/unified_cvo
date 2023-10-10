#pragma once
#include <iostream>
#include "utils/LidarPointSelector.hpp"
#include "utils/LidarPointType.hpp"
#include "utils/VoxelMap.hpp"
#include "utils/data_type.hpp"
#include <unordered_set>
//#include "utils/PointXYZIL.hpp"

namespace cvo {

  
  std::shared_ptr<cvo::CvoPointCloud> downsample_lidar_points(bool is_edge_only,
                                                              pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in,
                                                              float leaf_size,
                                                              const std::vector<int> & semantics_vec ){




    /*
    // Running edge detection + lego loam point selection
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_edge (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_surface (new pcl::PointCloud<pcl::PointXYZI>);
    std::vector<int> selected_edge_inds, selected_loam_inds;
    lps.edge_detection(pc, pc_out_edge, output_depth_grad, output_intenstity_grad, selected_edge_inds);
  
    lps.legoloam_point_selector(pc, pc_out_surface, edge_or_surface, selected_loam_inds);    
    // *pc_out += *pc_out_edge;
    // *pc_out += *pc_out_surface;
    //
    num_points_ = selected_indexes.size();
    */

    if (is_edge_only) {
      cvo::VoxelMap<pcl::PointXYZI> full_voxel(leaf_size);
      for (int k = 0; k < pc_in->size(); k++) {
        full_voxel.insert_point(&pc_in->points[k]);
      }
      std::vector<pcl::PointXYZI*> downsampled_results = full_voxel.sample_points();
      pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZI>);
      for (int k = 0; k < downsampled_results.size(); k++)
        downsampled->push_back(*downsampled_results[k]);
      std::shared_ptr<cvo::CvoPointCloud>  ret(new cvo::CvoPointCloud(downsampled, 5000, 64, cvo::CvoPointCloud::PointSelectionMethod::FULL));
      return ret;
    } else {
      int expected_points = 10000;
      double intensity_bound = 0.4;
      double depth_bound = 4.0;
      double distance_bound = 40.0;
      int kitti_beam_num = 64;
      cvo::LidarPointSelector lps(expected_points, intensity_bound, depth_bound, distance_bound, kitti_beam_num);
    
      /// edge points
      pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_edge (new pcl::PointCloud<pcl::PointXYZI>);
      std::vector<int> selected_edge_inds;
      std::unordered_map<pcl::PointXYZI*, int> edge_pt_to_ind, surface_pt_to_ind;
      std::vector <double> output_depth_grad;
      std::vector <double> output_intenstity_grad;
      //std::vector<int> edge_semantics;
      //if (semantics_vec.size())
      //  lps.edge_detection(pc_in, semantics_vec, pc_out_edge, output_depth_grad, output_intenstity_grad, selected_edge_inds, edge_semantics);  
      //else
      lps.edge_detection(pc_in, pc_out_edge, output_depth_grad, output_intenstity_grad, selected_edge_inds);
      std::unordered_set<int> edge_inds;
      for (int l = 0; l < selected_edge_inds.size(); l++){//(auto && j : selected_edge_inds) {
        int j = selected_edge_inds[l];
        edge_inds.insert(j);
        edge_pt_to_ind.insert(std::make_pair(&(pc_out_edge->points[l]), j));
      }

      /// surface points
      std::vector<float> edge_or_surface;
      std::vector<int> selected_loam_inds, loam_semantics;
      pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_loam (new pcl::PointCloud<pcl::PointXYZI>);
      //if (semantics_vec.size())
      //  lps.legoloam_point_selector(pc_in, semantics_vec, pc_out_loam, edge_or_surface, selected_loam_inds, loam_semantics);
      //else         
        lps.loam_point_selector(pc_in, pc_out_loam, edge_or_surface, selected_loam_inds);
      std::unordered_set<int> loam_inds;
      for (int l = 0; l < selected_loam_inds.size(); l++) { //auto && j : selected_loam_inds) {
        //loam_inds.insert(j);
        int j = selected_loam_inds[l];
        surface_pt_to_ind.insert(std::make_pair(&(pc_out_loam->points[l]), j));
      }      

      /// declare voxel map
      cvo::VoxelMap<pcl::PointXYZI> edge_voxel(leaf_size / 4); 
      cvo::VoxelMap<pcl::PointXYZI> surface_voxel(leaf_size );

      /// edge and surface downsample
      for (int k = 0; k < pc_out_edge->size(); k++) 
        edge_voxel.insert_point(&pc_out_edge->points[k]);
      std::vector<pcl::PointXYZI*> edge_results = edge_voxel.sample_points();
      for (int k = 0; k < pc_out_loam->size(); k++)  {
        if (edge_or_surface[k] > 0 &&
            edge_inds.find(selected_loam_inds[k]) == edge_inds.end()) {
          surface_voxel.insert_point(&pc_out_loam->points[k]);
        }
      }
      std::vector<pcl::PointXYZI*> surface_results = surface_voxel.sample_points();
      int total_selected_pts_num = edge_results.size() + surface_results.size();    
      std::shared_ptr<cvo::CvoPointCloud> ret(new cvo::CvoPointCloud(1, NUM_CLASSES));
      //ret->reserve(total_selected_pts_num, 1, NUM_CLASSES);
      std::cout<<"edge voxel selected points "<<edge_results.size()<<std::endl;
      std::cout<<"surface voxel selected points "<<surface_results.size()<<std::endl;    

      /// push
      for (int k = 0; k < edge_results.size(); k++) {
        cvo::CvoPoint pt;
        pt.getVector3fMap() = edge_results[k]->getVector3fMap();
        pt.features[0] = edge_results[k]->intensity;
        if (semantics_vec.size()){
          int index = semantics_vec[edge_pt_to_ind[edge_results[k]]];
          if (index == -1)
            continue;
          pt.label_distribution[ index] = 1; 
        }
        pt.geometric_type[0] = 1.0;
        pt.geometric_type[1] = 0.0;
        ret->push_back(pt);
      }
      /// surface downsample
      for (int k = 0; k < surface_results.size(); k++) {
        cvo::CvoPoint pt;
        pt.getVector3fMap() = surface_results[k]->getVector3fMap();
        pt.features[0] = surface_results[k]->intensity;
        if (semantics_vec.size()){
          int index = semantics_vec[surface_pt_to_ind[surface_results[k]]];
          if (index == -1)
            continue;
          pt.label_distribution[index] = 1; 
        }
        pt.geometric_type[0] = 0.0;
        pt.geometric_type[1] = 1.0;
        ret->push_back(pt);
        
      }
      return ret;

    }

  }



  void read_and_downsample_lidar_pc(const std::set<int> & result_selected_frames,
                                    DatasetHandler & dataset,
 
                                    const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> & tracking_poses,                                    
                                    int num_merging_sequential_frames,
                                    float voxel_size,
                                    int is_edge_only,
                                    int is_semantic,
                                    std::map<int, std::shared_ptr<cvo::CvoPointCloud>> & pcs) {
    for (auto i : result_selected_frames) {
      //for (int i = 0; i<gt_poses.size(); i++) {
      
      pcl::PointCloud<pcl::PointXYZI>::Ptr pc_local(new pcl::PointCloud<pcl::PointXYZI>);
      std::vector<int> semantics_local;      
      for (int j = 0; j < 1+num_merging_sequential_frames; j++){
        dataset.set_start_index(i+j);
        pcl::PointCloud<pcl::PointXYZI>::Ptr pc_pcl(new pcl::PointCloud<pcl::PointXYZI>);
        std::vector<int> semantics_single;
        if (is_semantic) {
          if (-1 == dataset.read_next_lidar(pc_pcl, semantics_single))
            break;
        } else {
          if (-1 == dataset.read_next_lidar(pc_pcl)) 
            break;
        }
        if (j > 0) {
          Eigen::Matrix4f pose_fi_to_fj = (tracking_poses[i].inverse() * tracking_poses[j+i]).cast<float>();
          #pragma omp parallel for 
          for (int k = 0; k < pc_pcl->size(); k++) {
            auto & p = pc_pcl->at(k);
            p.getVector3fMap() = pose_fi_to_fj.block(0,0,3,3) * p.getVector3fMap() + pose_fi_to_fj.block(0,3,3,1);
          }
        }
        *pc_local += *pc_pcl;
        semantics_local.insert(semantics_local.end(), semantics_single.begin(), semantics_single.end());
      }
      //if (i == 0)
      //  pcl::io::savePCDFileASCII(std::to_string(i) + ".pcd", *pc_local);
      //pc_local->write_to_pcd("0.pcd");
      float leaf_size = voxel_size;

      if (pc_local->size()) {
        std::shared_ptr<cvo::CvoPointCloud> pc = cvo::downsample_lidar_points(is_edge_only,
                                                                              pc_local,
                                                                              leaf_size,
                                                                              semantics_local);
        std::cout<<"new frame "<<i<<" downsampled from  "<<pc_local->size()<<" to "<<pc->size()<<"\n";
        pcs.insert(std::make_pair(i, pc));
        if (i == 0) {
          std::cout<<"is_semantic="<<is_semantic<<"\n";
          if (is_semantic) {
            cvo::CvoPointCloud pc_full(pc_local, semantics_local, NUM_CLASSES, 5000, 64, cvo::CvoPointCloud::PointSelectionMethod::FULL);
            pc_full.write_to_label_pcd("0_full.pcd");
            pc->write_to_label_pcd("0.pcd");
          } else
            pc->write_to_pcd("0.pcd");
        }
      }
    }
    
  }
  

}
