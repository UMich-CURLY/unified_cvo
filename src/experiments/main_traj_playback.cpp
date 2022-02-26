#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <string>
#include "viewer/viewer.h"
#include <Eigen/Dense>
#include <boost/filesystem.hpp>

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

void read_trajectory_file(std::string &traj_file_path,
                          std::vector<Mat34d_row,
                            Eigen::aligned_allocator<Mat34d_row>> & poses_all) {

    std::ifstream traj_file(traj_file_path);
    std::string line;
    while(!traj_file.eof()) {
      double pose_vec[7]; // tx ty tz qx qy qz qw
      for (int i = 0; i < 7; i++)
        traj_file >> pose_vec[i];
      Eigen::Quaternion<double> quat(pose_vec[6], pose_vec[3], pose_vec[4], pose_vec[5]);
      Eigen::Matrix3d R = quat.normalized().toRotationMatrix();
      Mat34d_row pose_mat;
      pose_mat.block<3, 3>(0, 0) = R;
      pose_mat(0, 3) = pose_vec[0];
      pose_mat(1, 3) = pose_vec[1];
      pose_mat(2, 3) = pose_vec[2];
      poses_all.push_back(pose_mat);
    }
    traj_file.close();

    // dbg
    // for (Mat34d_row& p : poses_all)
    //   std::cout << p << std::endl;
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

  // string cvo_param_file(argv[1]);  
  //cvo::CvoGPU cvo_align(cvo_param_file);
  std::string pcd_folder(argv[1]);
  std::string total_graphs_arg(argv[4]);
  std::string traj_file(argv[2]);
  int start_ind = std::stoi(std::string(argv[5]));
  int is_auto_preceed = std::stoi(std::string(argv[6]));

  std::unique_ptr<perl_registration::Viewer> viewer;
  if (argc == 7) {
      std::string screenDir(argv[3]);
      std::cout << "Screenshots will be saved to: " << screenDir << std::endl;
      viewer = std::make_unique<perl_registration::Viewer>(true, screenDir);
  } else {
      std::cout << "Screenshots will not be saved\n";
      
  }

  std::vector<Mat34d_row, Eigen::aligned_allocator<Mat34d_row>> poses_all;
  read_trajectory_file(traj_file, poses_all);
  
  
  //std::string graph_file_name(argv[2]);

  std::unordered_map<int, std::string> idx_to_viewerid;

  std::string s("title");
  viewer->addOrUpdateText (s,
                           0,
                           0,
                           "title");
  
  // gather all .pcd files
//   std::vector<boost::filesystem::path> all_files;
//   std::copy(boost::filesystem::directory_iterator(pcd_folder), boost::filesystem::directory_iterator(), std::back_inserter(all_files));
//   sort(all_files.begin(), all_files.end());
//   std::vector<int> available_frames;
//   for (auto& file : all_files) {
//       std::string filename = file.string().find;
//       std::size_t pos = filename.find(".pcd");
//       if (pos == std::string::npos) continue;
//       std::cout << filename.substr(0, pos) << std::endl;
//     //   available_frames.push_back(std::stoi(filename.substr(pos)));
//   }
  for (int i = start_ind; i < poses_all.size(); i++) {
    // visualize trajectory
    viewer->drawTrajectory(poses_all[i]);
    // visualize pcd if available
    std::string p_file = pcd_folder + "/" + std::to_string(i) + ".pcd";
    if (!boost::filesystem::exists(p_file)) continue;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPCDFile<pcl::PointXYZRGB> (p_file, *cloud);
    for (auto && p : *cloud) {
        Mat34d_row transform = Eigen::Map<Mat34d_row>(poses_all[i].data());
        Eigen::Vector4d point;
        point << (double)p.x, (double)p.y, (double) p.z, 1.0;
        Eigen::Vector3f p_t = (transform * point).cast<float>();
        p.getVector3fMap() = p_t;
    }
    viewer->addColorPointCloud(*cloud, std::to_string(i) + "pc");
    if (is_auto_preceed)
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    else {
        std::cout <<"Just rendered frame id "<< i << ", Press Enter to Continue";
        // std::cout<<"read pose["<<i<<"] as \n"<<poses_all[i]<<"\n";
        std::cin.ignore();
    }
  }

  /*
  std::list<std::pair<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr>> edges;
  for (int i = 0; i < edge_inds.size(); i++) {
    int first_ind = id_to_index[edge_inds[i].first];
    int second_ind = id_to_index[edge_inds[i].second];
    std::cout<<"first ind "<<first_ind<<", second ind "<<second_ind<<std::endl;
    std::pair<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr> p(frames[first_ind], frames[second_ind]);
    edges.push_back(p);
  }

  double time = 0;
  std::vector<bool> const_flags(frames.size(), false);
  const_flags[0] = true;  

  std::cout<<"Align ends. Total time is "<<time<<std::endl;
  */
  //f_name="after_BA.pcd";
  //write_transformed_pc(frames, f_name);
  
  while (!viewer->wasStopped()) {
    std::this_thread::sleep_for(std::chrono::microseconds(1000000));
  }


  return 0;
}

