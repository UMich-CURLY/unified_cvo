#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <boost/filesystem.hpp>
#include "utils/PoseLoader.hpp"
#include "dataset_handler/TartanAirHandler.hpp"
#include "utils/Calibration.hpp"
#include "utils/RawImage.hpp"
#include "utils/ImageRGBD.hpp"
#include "utils/Calibration.hpp"
#include "utils/CvoPointCloud.hpp"
#include "utils/PointSegmentedDistribution.hpp"

#include "utils/PointCloudIO.hpp"


#include "utils/viewer.hpp"
#include <Eigen/Dense>
using Mat34d_row = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;

void read_graph_file(std::string &graph_file_path,
                     std::vector<int> & frame_inds,
                     std::vector<std::pair<int, int>> & edges) {
  std::ifstream graph_file(graph_file_path);
  
  int num_frames, num_edges;
  graph_file>>num_frames >> num_edges;
  frame_inds.resize(num_frames);
  std::cout<<"Frame indices include ";
  for (int i = 0; i < num_frames; i++) {
    graph_file >> frame_inds[i];
    std::cout<<frame_inds[i]<<", ";
  }
  std::cout<<"\nEdges include ";

  for (int i =0; i < num_edges; i++ ) {
    std::pair<int, int> p;
    graph_file >> p.first >> p.second;
    edges.push_back(p);
    std::cout<<"("<<p.first<<", "<<p.second <<"), ";
  }
  std::cout<<"\n";
  graph_file.close();  
}
void read_graph_file(std::string &graph_file_path,
                     std::vector<int> & frame_inds,
                     std::vector<std::pair<int, int>> & edges,
                     // optional
                     std::vector<Mat34d_row,
                      Eigen::aligned_allocator<Mat34d_row>> & poses_all) {
  std::ifstream graph_file(graph_file_path);
  
  int num_frames, num_edges;
  graph_file>>num_frames >> num_edges;
  frame_inds.resize(num_frames);
  std::cout<<"Frame indices include ";
  for (int i = 0; i < num_frames; i++) {
    graph_file >> frame_inds[i];
    std::cout<<frame_inds[i]<<", ";
  }
  std::cout<<"\nEdges include ";

  for (int i =0; i < num_edges; i++ ) {
    std::pair<int, int> p;
    graph_file >> p.first >> p.second;
    edges.push_back(p);
    std::cout<<"("<<p.first<<", "<<p.second <<"), ";
  }
  std::cout<<"\n";
  if (graph_file.eof() == false){
    std::cout<<"poses included in the graph file\n";
    poses_all.resize(num_frames);
    for (int i = 0; i < num_frames; i++) {
      double pose_vec[12];
      for (int j = 0; j < 12; j++) {
        graph_file>>pose_vec[j];
      }
      poses_all[i]  << pose_vec[0] , pose_vec[1], pose_vec[2], pose_vec[3],
        pose_vec[4], pose_vec[5], pose_vec[6], pose_vec[7],
        pose_vec[8], pose_vec[9], pose_vec[10], pose_vec[11];
      std::cout<<"read pose["<<i<<"] as \n"<<poses_all[i]<<"\n";
    }
  }
  
  graph_file.close();  
}


/*
void read_pose_file(std::string & gt_fname,
                       std::vector<int> & frame_inds,
                       std::vector<cvo::Mat34d_row, Eigen::aligned_allocator<cvo::Mat34d_row>> & poses_all) {

  poses_all.resize(frame_inds.size());
  
  std::ifstream gt_file(gt_fname);

  std::string line;
  int line_ind = 0, curr_frame_ind = 0;
  
  while (std::getline(gt_file, line)) {
    
    if (line_ind < frame_inds[curr_frame_ind]) {
      line_ind ++;
      continue;
    }
    
    std::stringstream line_stream(line);
    std::string substr;
    double pose_v[12];
    int pose_counter = 0;
    while (std::getline(line_stream,substr, ' ')) {
      pose_v[pose_counter] = std::stod(substr);
      pose_counter++;
    }
    Eigen::Map<cvo::Mat34d_row> pose(pose_v);
    poses_all[curr_frame_ind] = pose;
    //if (curr_frame_ind == 2) {
    //  std::cout<<"read: line "<<frame_inds[curr_frame_ind]<<" pose is "<<poses_all[curr_frame_ind]<<std::endl;
    //}
    
    line_ind ++;
    curr_frame_ind++;
    //if (line_ind == frame_inds.size())
    if (curr_frame_ind == frame_inds.size())
      break;
  }


  gt_file.close();
}


void read_pose_file(std::string & gt_fname,
                    std::string & selected_pose_fname,
                    std::vector<int> & frame_inds,
                    std::vector<string> & timestamps,
                    std::vector<cvo::Mat34d_row,
                      Eigen::aligned_allocator<cvo::Mat34d_row>> & poses_all) {

  poses_all.resize(frame_inds.size());
  timestamps.resize(frame_inds.size());
  std::ifstream gt_file(gt_fname);

  std::string line;
  int line_ind = 0, curr_frame_ind = 0;

  std::string gt_file_subset(selected_pose_fname);
  ofstream outfile(gt_file_subset);
  
  while (std::getline(gt_file, line)) {
    
    if (line_ind < frame_inds[curr_frame_ind]) {
      line_ind ++;
      continue;
    }

    outfile<< line<<std::endl;
    
    std::stringstream line_stream(line);
    std::string timestamp;
    double xyz[3];
    double q[4]; // x y z w
    int pose_counter = 0;

    line_stream >> timestamp;
    std::string xyz_str[3];
    line_stream >> xyz_str[0] >> xyz_str[1] >> xyz_str[2];
    xyz[0] = std::stod(xyz_str[0]);
    xyz[1] = std::stod(xyz_str[1]);
    xyz[2] = std::stod(xyz_str[2]);
    std::string q_str[4];
    line_stream >> q_str[0] >> q_str[1] >> q_str[2] >> q_str[3];
    q[0] = stod(q_str[0]);
    q[1] = stod(q_str[1]);
    q[2] = stod(q_str[2]);
    q[3] = stod(q_str[3]);
    Eigen::Quaterniond q_eigen(q[3], q[0], q[1], q[2]);
    Sophus::SO3d quat(q_eigen);
    Eigen::Vector3d trans = Eigen::Map<Eigen::Vector3d>(xyz);
    Sophus::SE3d pose_sophus(quat, trans);

    cvo::Mat34d_row pose = pose_sophus.matrix().block<3,4>(0,0);
    
    //Eigen::Map<cvo::Mat34d_row> pose(pose_v);
    poses_all[curr_frame_ind] = pose;    
    //Eigen::Matrix<double, 4,4, Eigen::RowMajor> pose_id = Eigen::Matrix<double, 4,4, Eigen::RowMajor>::Identity();
    //poses_all[curr_frame_ind] = pose_id.block<3,4>(0,0);    
    timestamps[curr_frame_ind] = timestamp;
    //if (curr_frame_ind == 2) {
    //  std::cout<<"read: line "<<frame_inds[curr_frame_ind]<<" pose is "<<poses_all[curr_frame_ind]<<std::endl;
    //}
    
    line_ind ++;
    curr_frame_ind++;
    //if (line_ind == frame_inds.size())
    if (curr_frame_ind == frame_inds.size())
      break;
  }

  outfile.close();
  gt_file.close();
}



void write_transformed_pc(std::vector<cvo::CvoFrame::Ptr> & frames, std::string & fname) {
  pcl::PointCloud<pcl::PointXYZRGB> pc_all;
  for (auto ptr : frames) {
    cvo::CvoPointCloud new_pc;
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3,4>(0,0) = Eigen::Map<cvo::Mat34d_row>(ptr->pose_vec);
    
    Eigen::Matrix4f pose_f = pose.cast<float>();
    cvo::CvoPointCloud::transform(pose_f, *ptr->points, new_pc);

    pcl::PointCloud<pcl::PointXYZRGB> pc_curr;
    new_pc.export_to_pcd(pc_curr);

    pc_all += pc_curr;
  }
  pcl::io::savePCDFileASCII(fname, pc_all);
}
*/

int main(int argc, char** argv) {

  omp_set_num_threads(24);

  std::string pcd_folder(argv[1]);
  std::string ba_pose_folder(argv[2]);
  //int start_ind = std::stoi(std::string(argv[3]));
  int is_auto_preceed = std::stoi(std::string(argv[3]));
  int step_size = std::stoi(std::string(argv[4]));
  
  //std::string pose_file_format;
  //std::vector<Eigen::Matrix4d,
  //            Eigen::aligned_allocator<Eigen::Matrix4d>> all_poses;

  //std::unique_ptr<cvo::TartanAirHandler> reader;
  using PCLType = CvoPointToPCL<cvo::CvoPoint>::type;
  std::unique_ptr<cvo::Viewer<PCLType>> viewer = std::make_unique<cvo::Viewer<CvoPointToPCL<cvo::CvoPoint>::type>>(true, "./");
  //}
  
  //std::string graph_file_name(argv[2]);

  std::unordered_map<int, std::string> idx_to_viewerid;
  std::unordered_map<int, pcl::PointCloud<CvoPointToPCL<cvo::CvoPoint>::type>::Ptr> pcs_local_frame;
  std::string s("title");
  viewer->addOrUpdateText (s,
                           0,
                           0,
                           "title");

  int start_ind = 0;
  int last_ind = -1;
  int last_traj_ind = -1;

  /// read point clouds
  int pcd_count = std::distance(std::filesystem::directory_iterator(pcd_folder),
                                std::filesystem::directory_iterator{});
  for (int f = 0; f < pcd_count; f++) {
    
    int curr_frame_id = f * step_size;
    std::string pcd_fname = std::to_string(curr_frame_id)+".pcd";
    
    if ( !boost::filesystem::exists(pcd_folder + "/" + pcd_fname ) ) {
      std::cout<<pcd_fname<<" doesn't exist!\n";
      continue;
    } else
      std::cout<<"Read "<<pcd_folder + "/" + pcd_fname<<std::endl;

    std::string viewer_id = std::to_string(curr_frame_id);
    pcl::PointCloud<CvoPointToPCL<cvo::CvoPoint>::type>::Ptr cloud(new pcl::PointCloud<CvoPointToPCL<cvo::CvoPoint>::type>);
    pcl::io::loadPCDFile<CvoPointToPCL<cvo::CvoPoint>::type>(pcd_folder + "/" + pcd_fname, *cloud);
    pcs_local_frame.insert(std::make_pair(curr_frame_id, cloud));
    if (cloud->size() > 0 ) {
      viewer->updatePointCloud(*cloud, viewer_id);      
      if ( idx_to_viewerid.find(curr_frame_id) == idx_to_viewerid.end()) {
        idx_to_viewerid.insert(std::make_pair(f, std::to_string(curr_frame_id)));
      }
    }
    
    //last_ind = frame_inds[1];
  }
  
  int pose_iter_count = std::distance(std::filesystem::directory_iterator(ba_pose_folder),
                                      std::filesystem::directory_iterator{});
  std::vector<Eigen::Matrix4d,
              Eigen::aligned_allocator<Eigen::Matrix4d>>  last_poses(pcs_local_frame.size(), Eigen::Matrix4d::Identity());
  std::cout<<"Total pose iteration num is "<<pose_iter_count<<"\n";
  for (int j = start_ind; j < pose_iter_count; j++ ) {
    
    if (j > start_ind && is_auto_preceed)
      std::this_thread::sleep_for(std::chrono::microseconds(500000));
    else {
      std::cout <<"Just rendered "<<j<<"th iter, Press Enter to Continue";
      std::cin.ignore();
    }

    std::string pose_file = ba_pose_folder + "/" + std::string("rkhs_irls_iter_") + std::to_string(j) + ".txt";
    if ( !boost::filesystem::exists(pose_file ) ) {
      std::cout<<pose_file<<" doesn't exist!\n";
      continue;
    } else
      std::cout<<"Read "<<pose_file<<std::endl;
    
    std::vector<Eigen::Matrix4d,
                Eigen::aligned_allocator<Eigen::Matrix4d>>  poses(pcs_local_frame.size(), Eigen::Matrix4d::Identity());    
    cvo::read_pose_file_tartan_format(pose_file,
                                     0, //start_ind,
                                     pcs_local_frame.size()-1,
                                     poses
                                     );
    std::cout<<"Finish reading "<<pose_file<<"\n";

    pcl::PointCloud<PCLType> stacked;
    
    for (int k = 0; k < poses.size(); k++) {
      Eigen::Matrix4d transform = poses[k];
      //std::cout<<"Tranform k is "<<transform<<"\n";
      auto cloud = pcs_local_frame[std::stoi(idx_to_viewerid[k])];

      viewer->addOrUpdateText ( "iter: "+std::to_string(j),
                                0,
                                0,
                                "title");

      if (k == 0 && j == 0) 
        pcl::io::savePCDFileASCII("raw" + std::to_string(j)+"_"+std::to_string(k)+".pcd", *cloud);          
      
      //#pragma omp parallel
      for (int l = 0; l < cloud->size(); l++ ) {
        CvoPointToPCL<cvo::CvoPoint>::type p = cloud->at(l);
        Eigen::Vector4d point;
        point << (double)p.x, (double)p.y, (double) p.z, 1.0;
        Eigen::Vector3f p_t = (transform * (last_poses[k].inverse()) * point).cast<float>().head<3>();
        PCLType p_new = p;
        p_new.getVector3fMap() = p_t;
        (*cloud)[l] = (p_new);
      }
      last_poses[k] = poses[k];

      //stacked += *cloud;

      //if (k == 0) 
      //  pcl::io::savePCDFileASCII("stacked" + std::to_string(j)+".pcd", stacked);        

      viewer->updatePointCloud(*cloud, idx_to_viewerid[k]);

      //saveCameraParameters 
    }


    //std::string f_name("before_BA.pcd");
    //write_transformed_pc(frames, f_name);
  }

  
  while (!viewer->wasStopped()) {
    std::this_thread::sleep_for(std::chrono::microseconds(1000000));
  }


  return 0;
}

