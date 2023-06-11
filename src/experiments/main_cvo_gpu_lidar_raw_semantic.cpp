#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
//#include <opencv2/opencv.hpp>
#include "dataset_handler/KittiHandler.hpp"
//#include "graph_optimizer/Frame.hpp"
#include "dataset_handler/PoseLoader.hpp"
#include "utils/LidarPointSelector.hpp"
#include "utils/LidarPointType.hpp"
#include "utils/ImageRGBD.hpp"
#include "utils/Calibration.hpp"
#include "utils/SymbolHash.hpp"
#include "utils/VoxelMap.hpp"
#include "utils/data_type.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoGPU.hpp"
//#include "cvo/Cvo.hpp"
using namespace std;
using namespace boost::filesystem;


std::shared_ptr<cvo::CvoPointCloud> downsample_lidar_points(bool is_edge_only,
                                                            pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in,
                                                            std::vector<int> semantics,
                                                            float leaf_size) {


    int expected_points = 5000;
    double intensity_bound = 0.4;
    double depth_bound = 4.0;
    double distance_bound = 40.0;
    int kitti_beam_num = 64;
    cvo::LidarPointSelector lps(expected_points, intensity_bound, depth_bound, distance_bound, kitti_beam_num);

  if (is_edge_only) {
    cvo::VoxelMap<pcl::PointXYZI> full_voxel(leaf_size);
    std::unordered_map<pcl::PointXYZI*, int> ptr_to_ind;
    for (int k = 0; k < pc_in->size(); k++) {
      full_voxel.insert_point(&pc_in->points[k]);
      ptr_to_ind[&pc_in->points[k]] = k;
    }
    std::vector<pcl::PointXYZI*> downsampled_results = full_voxel.sample_points();
    std::vector<int> semantics_downsampled;
    pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZI>);
    for (int k = 0; k < downsampled_results.size(); k++) {
      downsampled->push_back(*downsampled_results[k]);
      int ind = ptr_to_ind[downsampled_results[k]];
      semantics_downsampled.push_back(semantics[ind]);
    }
    std::shared_ptr<cvo::CvoPointCloud>  ret(new cvo::CvoPointCloud(downsampled, semantics_downsampled, NUM_CLASSES,  5000, 64, cvo::CvoPointCloud::PointSelectionMethod::FULL));
     return ret;
  } else {
    
    /// edge points
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_edge (new pcl::PointCloud<pcl::PointXYZI>);  
    std::vector<int> selected_edge_inds;
    std::vector <double> output_depth_grad;
    std::vector <double> output_intenstity_grad;
    lps.edge_detection(pc_in, pc_out_edge, output_depth_grad, output_intenstity_grad, selected_edge_inds);
    std::unordered_set<int> edge_inds;
    for (auto && j : selected_edge_inds) edge_inds.insert(j);

    /// surface points
    std::vector<float> edge_or_surface;
    std::vector<int> selected_loam_inds;
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_loam (new pcl::PointCloud<pcl::PointXYZI>);        
    lps.loam_point_selector(pc_in, pc_out_loam, edge_or_surface, selected_loam_inds);

    /// declare voxel map
    cvo::VoxelMap<pcl::PointXYZI> edge_voxel(leaf_size); 
    cvo::VoxelMap<pcl::PointXYZI> surface_voxel(leaf_size);

    /// edge and surface downsample
    for (int k = 0; k < pc_out_edge->size(); k++) 
      edge_voxel.insert_point(&pc_out_edge->points[k]);
    std::vector<pcl::PointXYZI*> edge_results = edge_voxel.sample_points();
    for (int k = 0; k < pc_out_loam->size(); k++)  {
      if (edge_or_surface[k] > 0 &&
          edge_inds.find(selected_loam_inds[k]) == edge_inds.end())
        surface_voxel.insert_point(&pc_out_loam->points[k]);
    }
    std::vector<pcl::PointXYZI*> surface_results = surface_voxel.sample_points();
    int total_selected_pts_num = edge_results.size() + surface_results.size();    
    std::shared_ptr<cvo::CvoPointCloud> ret(new cvo::CvoPointCloud(1, NUM_CLASSES));
    ret->reserve(total_selected_pts_num, 1, NUM_CLASSES);
    std::cout<<"edge voxel selected points "<<edge_results.size()<<std::endl;
    std::cout<<"surface voxel selected points "<<surface_results.size()<<std::endl;    

    /// push
    for (int k = 0; k < edge_results.size(); k++) {
      Eigen::VectorXf feat(1);
      feat(0) = edge_results[k]->intensity;
      Eigen::VectorXf semantics = Eigen::VectorXf::Zero(NUM_CLASSES);
      Eigen::VectorXf geo_t(2);
      geo_t << 1.0, 0;
      ret->add_point(k, edge_results[k]->getVector3fMap(),  feat, semantics, geo_t);
    }
    /// surface downsample
    for (int k = 0; k < surface_results.size(); k++) {
      // surface_pcl.push_back(*surface_results[k]);
      Eigen::VectorXf feat(1);
      feat(0) = surface_results[k]->intensity;
      Eigen::VectorXf semantics = Eigen::VectorXf::Zero(NUM_CLASSES);
      Eigen::VectorXf geo_t(2);
      geo_t << 0, 1.0;
      ret->add_point(k+edge_results.size(), surface_results[k]->getVector3fMap(), feat,
                     semantics, geo_t);
    }
    return ret;

  }

}

int main(int argc, char *argv[]) {
  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  cvo::KittiHandler kitti(argv[1], 1);
  int total_iters = kitti.get_total_number();
  string cvo_param_file(argv[2]);
  string calib_file;
  calib_file = string(argv[1] ) +"/cvo_calib.txt"; 
  cvo::Calibration calib(calib_file);
  std::ofstream accum_output(argv[3]);
  int start_frame = std::stoi(argv[4]);
  kitti.set_start_index(start_frame);
  int max_num = std::stoi(argv[5]);

  accum_output <<"1 0 0 0 0 1 0 0 0 0 1 0\n";

  
  cvo::CvoGPU cvo_align(cvo_param_file );
  cvo::CvoParams & init_param = cvo_align.get_params();
  float ell_init = init_param.ell_init;
  float ell_decay_rate = init_param.ell_decay_rate;
  int ell_decay_start = init_param.ell_decay_start;
  init_param.ell_init = init_param.ell_init_first_frame;
  init_param.ell_decay_rate = init_param.ell_decay_rate_first_frame;
  init_param.ell_decay_start  = init_param.ell_decay_start_first_frame;  
  cvo_align.write_params(&init_param);
  
  Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();  // from source frame to the target frame
  init_guess(2,3)=0;
  Eigen::Affine3f init_guess_cpu = Eigen::Affine3f::Identity();
  init_guess_cpu.matrix()(2,3)=0;
  Eigen::Matrix4f accum_mat = Eigen::Matrix4f::Identity();
  // start the iteration

  pcl::PointCloud<pcl::PointXYZI>::Ptr source_pc(new pcl::PointCloud<pcl::PointXYZI>);
  std::vector<int> semantics_source;
  kitti.read_next_lidar(source_pc,  semantics_source);

  std::shared_ptr<cvo::CvoPointCloud> source(new cvo::CvoPointCloud(source_pc, semantics_source, NUM_CLASSES, 5000, 64)); 
  
  //kitti.read_next_stereo(source_left, source_right);
  //std::shared_ptr<cvo::Frame> source(new cvo::Frame(start_frame, source_pc,
  //                                                  semantics_source, 
  //                                                 calib));
  //0.2));
  double total_time = 0;
  int i = start_frame;
  for (; i<min(total_iters, start_frame+max_num)-1 ; i++) {
    
    // calculate initial guess
    std::cout<<"\n\n\n\n============================================="<<std::endl;
    std::cout<<"Aligning "<<i<<" and "<<i+1<<" with GPU "<<std::endl;

    kitti.next_frame_index();
    pcl::PointCloud<pcl::PointXYZI>::Ptr target_pc(new pcl::PointCloud<pcl::PointXYZI>);
    std::vector<int> semantics_target;
    if (kitti.read_next_lidar(target_pc, semantics_target) != 0) {
      std::cout<<"finish all files\n";
      break;
    }

    //std::shared_ptr<cvo::Frame> target(new cvo::Frame(i+1, target_pc, semantics_target, calib));
    std::shared_ptr<cvo::CvoPointCloud> target(new cvo::CvoPointCloud(target_pc, semantics_target, NUM_CLASSES, 5000, 64));

    // std::cout<<"reading "<<files[cur_kf]<<std::endl;
    std::cout<<"NUm of source pts is "<<source->num_points()<<"\n";
    std::cout<<"NUm of target pts is "<<target->num_points()<<"\n";

    Eigen::Matrix4f result, init_guess_inv;
    init_guess_inv = init_guess.inverse();
    double this_time = 0;
    cvo_align.align(*source, *target, init_guess_inv, result, &this_time);
    total_time += this_time;
    
    // get tf and inner product from cvo getter
    //double in_product = cvo_align.inner_product(*source, *target, result);
    //double in_product_normalized = cvo_align.inner_product_normalized();
    //int non_zeros_in_A = cvo_align.number_of_non_zeros_in_A();
    //std::cout<<"The gpu inner product between "<<i-1 <<" and "<< i <<" is "<<in_product<<"\n";
    //std::cout<<"The normalized inner product between "<<i-1 <<" and "<< i <<" is "<<in_product_normalized<<"\n";
    std::cout<<"Transform is "<<result <<"\n\n";

    // append accum_tf_list for future initialization
    init_guess = result;
    accum_mat = accum_mat * result;
    std::cout<<"accum tf: \n"<<accum_mat<<std::endl;
    
    
    // log accumulated pose

    accum_output << accum_mat(0,0)<<" "<<accum_mat(0,1)<<" "<<accum_mat(0,2)<<" "<<accum_mat(0,3)<<" "
                <<accum_mat(1,0)<<" " <<accum_mat(1,1)<<" "<<accum_mat(1,2)<<" "<<accum_mat(1,3)<<" "
                <<accum_mat(2,0)<<" " <<accum_mat(2,1)<<" "<<accum_mat(2,2)<<" "<<accum_mat(2,3);
    accum_output<<"\n";
    accum_output<<std::flush;
    
    std::cout<<"\n\n===========next frame=============\n\n";
   
    source = target;
    if (i == start_frame) {
      init_param.ell_init = ell_init;
      init_param.ell_decay_start = ell_decay_rate;
      init_param.ell_decay_start = ell_decay_start;            
      cvo_align.write_params(&init_param);
      
    } //else if (i < start_frame + 20)  {
      //init_param.ell_init =  1.0;
      //init_param.ell_max = 1.0;
      //cvo_align.write_params(&init_param);

      
    //}

  }

  std::cout<<"time per frame is "<<total_time / double(i - start_frame + 1)<<std::endl;
  accum_output.close();

  return 0;
}
