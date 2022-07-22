#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "cvo/CvoFrameGPU.hpp"
#include "cvo/CvoFrame.hpp"
#include "utils/Calibration.hpp"
#include "utils/VoxelMap.hpp"
#include "utils/ImageStereo.hpp"
#include "cvo/CvoGPU.hpp"
#include "utils/CvoPointCloud.hpp"
#include "dataset_handler/KittiHandler.hpp"
#include "utils/CvoPoint.hpp"
#include "utils/VoxelMap_impl.hpp"



void read_kitti_pose_file(const std::string & tracking_fname,
                    std::vector<int> & frame_inds,
                          std::vector<Eigen::Matrix4f,
                          Eigen::aligned_allocator<Eigen::Matrix4f>> & poses_all) {

  poses_all.resize(frame_inds.size());
  std::ifstream gt_file(tracking_fname);

  std::string line;
  int line_ind = 0, curr_frame_ind = 0;

  //std::string gt_file_subset(selected_pose_fname);
  //ofstream outfile(gt_file_subset);

  while (std::getline(gt_file, line)) {
    
    if (line_ind < frame_inds[curr_frame_ind]) {
      line_ind ++;
      continue;
    }

    // outfile<< line<<std::endl;
    
    std::stringstream line_stream(line);
    std::string substr;
    float pose_v[12];
    int pose_counter = 0;
    while (std::getline(line_stream,substr, ' ')) {
      pose_v[pose_counter] = std::stof(substr);
      pose_counter++;
    }
    Eigen::Map<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>> pose(pose_v);

    //Eigen::Map<cvo::Mat34d_row> pose(pose_v);
    poses_all[curr_frame_ind] = Eigen::Matrix4f::Identity();
    Eigen::Matrix<float, 3, 4> pose_c = pose;
    poses_all[curr_frame_ind].block<3,4>(0,0) = pose_c;    
    //Eigen::Matrix<double, 4,4, Eigen::RowMajor> pose_id = Eigen::Matrix<double, 4,4, Eigen::RowMajor>::Identity();
    //poses_all[curr_frame_ind] = pose_id.block<3,4>(0,0);    
    //if (curr_frame_ind == 2) {
    //  std::cout<<"read: line "<<frame_inds[curr_frame_ind]<<" pose is "<<poses_all[curr_frame_ind]<<std::endl;
    //}
    
    line_ind ++;
    curr_frame_ind++;
    //if (line_ind == frame_inds.size())
    if (curr_frame_ind == frame_inds.size())
      break;
  }

  ///  outfile.close();
  gt_file.close();
}


int main(int argc, char ** argv) {

  omp_set_num_threads(24);

  cvo::KittiHandler kitti(argv[1], 0);
  std::string cvo_param_file(argv[2]);    
  //std::string graph_file_name(argv[3]);
  std::string tracking_fname(argv[3]);
  int start_ind = std::stoi(argv[4]);
  int total_inds = std::stoi(argv[5]);

  float depth_dir_ell = std::stof(argv[6]);
  float depth_normal_ell = std::stof(argv[7]);

  
  int total_iters = kitti.get_total_number();

  cvo::CvoGPU cvo_align(cvo_param_file);
  std::string calib_file;
  calib_file = std::string(argv[1] ) +"/cvo_calib.txt"; 
  cvo::Calibration calib(calib_file, cvo::Calibration::STEREO);

  std::vector<int> frame_inds;
  for (int i = 0; i < total_inds; i++) {
    frame_inds.push_back(i + start_ind);
  }
  //std::vector<cvo::Mat34d_row, Eigen::aligned_allocator<cvo::Mat34d_row>> poses;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> poses;
  read_kitti_pose_file(tracking_fname, frame_inds, poses);

  std::vector<std::shared_ptr<cvo::CvoPointCloud>> pcs;
  std::vector<std::shared_ptr<cvo::CvoPointCloud>> pcs_full;
  std::vector<std::vector<float>> depths;
  std::vector<std::vector<float>> weights;
  
  for (int i = 0; i<1; i++) {

    int curr_frame_id = frame_inds[i];
    
    kitti.set_start_index(curr_frame_id);
    cv::Mat left, right;
    kitti.read_next_stereo(left, right);
    std::shared_ptr<cvo::ImageStereo> raw(new cvo::ImageStereo(left, right));

    std::shared_ptr<cvo::CvoPointCloud> pc_full(new cvo::CvoPointCloud(*raw,  calib, cvo::CvoPointCloud::FULL));
    std::shared_ptr<cvo::CvoPointCloud> pc_edge_raw(new cvo::CvoPointCloud(*raw, calib, cvo::CvoPointCloud::DSO_EDGES));
    //std::shared_ptr<cvo::CvoPointCloud> pc(new cvo::CvoPointCloud(*raw, calib, cvo::CvoPointCloud::DSO_EDGES));
    
    pcl::PointCloud<cvo::CvoPoint>::Ptr raw_pcd_edge(new pcl::PointCloud<cvo::CvoPoint>);
    pc_edge_raw->export_to_pcd<cvo::CvoPoint>(*raw_pcd_edge);
    
    pcl::PointCloud<cvo::CvoPoint>::Ptr raw_pcd_surface(new pcl::PointCloud<cvo::CvoPoint>);    
    pc_full->export_to_pcd<cvo::CvoPoint>(*raw_pcd_surface);
    
    pcl::PointCloud<cvo::CvoPoint> edge_pcl;
    pcl::PointCloud<cvo::CvoPoint> surface_pcl;

    float leaf_size = cvo_align.get_params().multiframe_downsample_voxel_size;
    while (true) {
      std::cout<<"Current leaf size is "<<leaf_size<<std::endl;
      edge_pcl.clear();
      surface_pcl.clear();
      cvo::VoxelMap<cvo::CvoPoint> edge_voxel(leaf_size / 4); // /10
      cvo::VoxelMap<cvo::CvoPoint> surface_voxel(leaf_size);

      for (int k = 0; k < raw_pcd_edge->size(); k++) {
        edge_voxel.insert_point(&raw_pcd_edge->points[k]);
      }
      std::vector<cvo::CvoPoint*> edge_results = edge_voxel.sample_points();

      for (int k = 0; k < edge_results.size(); k++)
        edge_pcl.push_back(*edge_results[k]);
      std::cout<<"edge voxel selected points "<<edge_pcl.size()<<std::endl;
      for (int k = 0; k < raw_pcd_surface->size(); k++) {
        surface_voxel.insert_point(&raw_pcd_surface->points[k]);
      }
      std::vector<cvo::CvoPoint*> surface_results = surface_voxel.sample_points();
      for (int k = 0; k < surface_results.size(); k++)
        surface_pcl.push_back(*surface_results[k]);
      std::cout<<"surface voxel selected points "<<surface_pcl.size()<<std::endl;

      int total_selected_pts_num = edge_results.size() + surface_results.size();

      break;
    
    }
    // std::cout<<"start voxel filtering...\n"<<std::flush;
    //sor.setInputCloud (raw_pcd);

    //sor.setLeafSize (leaf_size, leaf_size, leaf_size);
    //sor.filter (*raw_pcd);
    //std::cout<<"construct filtered cvo points with voxel size "<<leaf_size<<"\n"<<std::flush;
    std::shared_ptr<cvo::CvoPointCloud> pc_edge(new cvo::CvoPointCloud(edge_pcl, cvo::CvoPointCloud::GeometryType::EDGE));
    std::shared_ptr<cvo::CvoPointCloud> pc_surface(new cvo::CvoPointCloud(surface_pcl, cvo::CvoPointCloud::GeometryType::SURFACE));
    std::shared_ptr<cvo::CvoPointCloud> pc(new cvo::CvoPointCloud);
    *pc = *pc_edge + *pc_surface;

    std::cout<<"Voxel number points is "<<pc->num_points()<<std::endl;

    pcl::PointCloud<cvo::CvoPoint> pcd_to_save;
    pc->write_to_color_pcd(std::to_string(curr_frame_id)+".pcd");
    
    
    std::cout<<"Load "<<curr_frame_id<<", "<<pc->positions().size()<<" number of points\n"<<std::flush;
    pcs.push_back(pc);
    pcs_full.push_back(pc_full);

  }


  for (int i = 1; i<frame_inds.size(); i++) {

    int curr_frame_id = frame_inds[i];
    
    kitti.set_start_index(curr_frame_id);
    cv::Mat left, right;
    kitti.read_next_stereo(left, right);
    std::shared_ptr<cvo::ImageStereo> raw(new cvo::ImageStereo(left, right));

    std::shared_ptr<cvo::CvoPointCloud> pc_full(new cvo::CvoPointCloud(*raw,  calib, cvo::CvoPointCloud::FULL));
    pc_full->write_to_color_pcd(std::to_string(curr_frame_id)+".pcd");
    
    pcs.push_back(pc_full);
  }

  

  std::cout<<"\n\nFinish loading temporal frames, the number of of points for each frame is ";
  for (int i = 0; i < frame_inds.size(); i++)
    std::cout<<pcs[i]->size()<<" ";
  std::cout<<"\n\n";


  // start depth filtering
  const cvo::CvoPointCloud & kf = *pcs[0];
  Eigen::Matrix3f non_isotropic_kernel= Eigen::Matrix3f::Identity();
  non_isotropic_kernel(0,0) = depth_normal_ell;
  non_isotropic_kernel(2,2) = depth_dir_ell;  
  non_isotropic_kernel(1,1) = depth_normal_ell;
  std::cout<<"kernel is "<<non_isotropic_kernel<<std::endl;
  depths.resize(kf.size());
  weights.resize(kf.size());
  for (int i = 1; i < total_inds; i ++) {

    Eigen::Matrix4f T_s = poses[0];
    Eigen::Matrix4f T_t = poses[i];
    Eigen::Matrix4f T_t2s = T_t.inverse() * T_s;
    Eigen::Matrix4f T_s2t = T_s.inverse() * T_t;
    std::cout<<"\nT_s\n"<<T_s
             <<"\nT_t\n"<<T_t
             <<"\nT_t2s\n"<<T_t2s
             <<"\nT_s2t\n"<<T_s2t<<std::endl;

    cvo::Association association;
    cvo_align.compute_association_gpu(kf,
                                      *pcs[i],
                                      T_t2s,
                                      non_isotropic_kernel,
                                      association
                                      );
    std::cout<<" non kf "<<i<<" has nonzeros "<<association.pairs.nonZeros()<<std::endl;

    cvo::CvoPointCloud pc_t_in_s(pcs[0]->num_features(),
                                 pcs[0]->num_classes()
                                 );
    cvo::CvoPointCloud::transform(T_s2t,
                                  *pcs[i],
                                  pc_t_in_s
                                  );

    for (int k=0; k<association.pairs.outerSize(); ++k)
    {
      for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(association.pairs,k); it; ++it) {

        int idx1 = it.row();
        int idx2 = it.col();
        float val = it.value();
        depths[idx1].push_back(pc_t_in_s.at(idx2)(2));
        weights[idx1].push_back(val);
        if (idx1 == 2349)
          std::cout<<"j="<<idx2<<", weght is "<<weights[idx1][weights[idx1].size()-1]<<std::endl;
      }
    }
  }

  cvo::CvoPointCloud kf_filtered(kf.num_features(), kf.num_classes());
  int total_pts = 0;
  for (int i = 0; i < kf.size(); i++)
    if (depths[i].size() > 3)
      total_pts ++;
  kf_filtered.reserve(total_pts, kf.num_features(), kf.num_classes());
  int index = 0;
  for (int i = 0; i < kf.size(); i++){
    if (depths[i].size() > 3) {
      float depth = 0; //kf.at(i)(2);
      float weight = 0;
      for (int k = 0; k < depths[i].size(); k++) {
        depth += depths[i][k] * weights[i][k];
        weight += weights[i][k];
        
        //std::cout<<"depth is "<< depths[i][k] * weights[i][k]<<", weight is "<<weights[i][k]<<"\n";
      }
      depth += kf.at(i)(2) * (weight / depths[i].size());      
      weight += weight / depths[i].size();

      depth /= weight;

      std::cout<<"depth before "<<kf.at(i)(2)<<", after "<<depth<<std::endl;
      
      Eigen::Vector3f xyz = kf.at(i);
      xyz = (xyz / xyz(2)).eval();
      xyz = (xyz * depth).eval();
      Eigen::VectorXf feature =  kf.feature_at(i);
      Eigen::VectorXf label = kf.label_at(i);
      Eigen::VectorXf gtype = kf.geometry_type_at(i);
      kf_filtered.add_point(index, xyz,feature,label,
                            gtype);
      index ++;
    } 
  }

  std::cout<<"total pts after depth filtering is "<<index<<std::endl;
  kf.write_to_color_pcd("before_depth_filtering.pcd");  
  kf_filtered.write_to_color_pcd("after_depth_filtering.pcd");

  return 0;
}
