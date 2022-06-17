#include <iostream>
#include <list>
#include <vector>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <set>
#include <Eigen/Dense>
#include <Eigen/Geometry> 
#include "cvo/CvoGPU.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoFrame.hpp"
#include "dataset_handler/TumHandler.hpp"
#include "utils/ImageRGBD.hpp"
#include "utils/Calibration.hpp"
#include "utils/VoxelMap.hpp"
#include "cvo/IRLS_State_CPU.hpp"
#include "cvo/IRLS_State_GPU.hpp"
#include "cvo/IRLS_State.hpp"
#include "cvo/CvoFrame.hpp"
#include "cvo/CvoFrameGPU.hpp"

using namespace std;


void read_graph_file(std::string &graph_file_path,
                     std::vector<int> & frame_inds,
                     std::vector<std::pair<int, int>> & edges,
                     // optional
                     std::vector<cvo::Mat34d_row,
                     Eigen::aligned_allocator<cvo::Mat34d_row>> & poses_all){

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

void write_traj_file(std::string & fname,
                     std::vector<std::string> & timestamps,
                     std::vector<cvo::CvoFrame::Ptr> & frames ) {
  std::ofstream outfile(fname);
  for (int i = 0; i< frames.size(); i++) {
    cvo::CvoFrame::Ptr ptr = frames[i];
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3,4>(0,0) = Eigen::Map<cvo::Mat34d_row>(ptr->pose_vec);
    Sophus::SO3d q(pose.block<3,3>(0,0));
    auto q_eigen = q.unit_quaternion().coeffs();
    Eigen::Vector3d t(pose.block<3,1>(0,3));
    outfile <<timestamps[i]<<" "<< t(0) <<" "<< t(1)<<" "<<t(2)<<" "
            <<q_eigen[0]<<" "<<q_eigen[1]<<" "<<q_eigen[2]<<" "<<q_eigen[3]<<std::endl;
    
  }
  outfile.close();
}

void write_transformed_pc(std::vector<cvo::CvoFrame::Ptr> & frames, std::string & fname, int max_frames=-1) {
  pcl::PointCloud<pcl::PointXYZRGB> pc_all;
  int counter = 0;
  for (auto ptr : frames) {
    if (counter == max_frames) break;
    cvo::CvoPointCloud new_pc;
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3,4>(0,0) = Eigen::Map<cvo::Mat34d_row>(ptr->pose_vec);
    
    Eigen::Matrix4f pose_f = pose.cast<float>();
    cvo::CvoPointCloud::transform(pose_f, *ptr->points, new_pc);

    pcl::PointCloud<pcl::PointXYZRGB> pc_curr;
    new_pc.export_to_pcd(pc_curr);

    pc_all += pc_curr;
    counter++;
  }
  pcl::io::savePCDFileASCII(fname, pc_all);
}

void pose_to_tum(std::string & tum_time,
                 const cvo::Mat34d_row & pose,
                 std::string & tum_pose_out) {
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> rot_row = pose.block<3,3>(0,0);
  Eigen::Matrix3d rot = rot_row;
  Eigen::Quaterniond rot_q(rot);

  double xyzw[4];
  xyzw[0] = rot_q.x();
  xyzw[1] = rot_q.y();
  xyzw[2] = rot_q.z();
  xyzw[3] = rot_q.w();

  tum_pose_out = tum_time + " " +
    std::to_string(pose(0,3)) + " " +
    std::to_string(pose(1,3)) + " " +
    std::to_string(pose(2,3)) + " " +
    std::to_string(xyzw[0]) + " " +
    std::to_string(xyzw[1]) + " " +
    std::to_string(xyzw[2]) + " " +
    std::to_string(xyzw[3]);

  std::cout<<"new tum pose line: "<<tum_pose_out<<"\n";
  return;
  
}


int main(int argc, char** argv) {

  omp_set_num_threads(24);

  cvo::TumHandler tum(argv[1]);
  string cvo_param_file(argv[2]);    
  std::string graph_file_name(argv[3]);
  //std::string tracking_fname(argv[4]);

  int num_const_frames = 1;
  num_const_frames = std::stoi(std::string(argv[4]));

  std::string graph_file_folder;
  //if (argc > 5)
  graph_file_folder = std::string(argv[5]);
  
  std::string covisMapFile;
  if (argc > 6)
    covisMapFile = std::string(argv[6]);
  
  int total_iters = tum.get_total_number();
  vector<string> vstrRGBName = tum.get_rgb_name_list();


  cvo::CvoGPU cvo_align(cvo_param_file);
  string calib_file;
  calib_file = string(argv[1] ) +"/cvo_calib.txt"; 
  cvo::Calibration calib(calib_file, cvo::Calibration::RGBD);

  

  std::vector<int> frame_inds;
  std::vector<std::pair<int, int>> edge_inds;
  std::vector<cvo::Mat34d_row, Eigen::aligned_allocator<cvo::Mat34d_row>> BA_poses;
  read_graph_file(graph_file_name, frame_inds, edge_inds, BA_poses);

  std::vector<cvo::Mat34d_row, Eigen::aligned_allocator<cvo::Mat34d_row>> gt_poses;
  std::vector<cvo::Mat34d_row, Eigen::aligned_allocator<cvo::Mat34d_row>> tracking_poses;
  std::vector<std::string> timestamps;

  std::string tracking_subset_poses_fname("cvo_track_poses.txt");
  std::ofstream tracking_subset_poses_f(tracking_subset_poses_fname);
  //read_pose_file(tracking_fname, tracking_subset_poses_fname , frame_inds, timestamps, tracking_poses);

  // read point cloud
  std::vector<cvo::CvoFrame::Ptr> frames;
  std::vector<std::shared_ptr<cvo::CvoPointCloud>> pcs;
  std::vector<std::shared_ptr<cvo::CvoPointCloud>> pcs_full;
  std::vector<std::shared_ptr<cvo::CvoFrame>> frames_full;
  std::unordered_map<int, int> id_to_index;
  for (int i = 0; i<frame_inds.size(); i++) {

    int curr_frame_id = frame_inds[i];

    std::string tum_format_tracking_pose;
    pose_to_tum(vstrRGBName[curr_frame_id],
                BA_poses[i],
                tum_format_tracking_pose);
    tracking_subset_poses_f << tum_format_tracking_pose << std::endl;
    
    tum.set_start_index(curr_frame_id);
    timestamps.push_back(vstrRGBName[curr_frame_id]);
    cv::Mat rgb, depth;
    tum.read_next_rgbd(rgb, depth);
    vector<uint16_t> depth_data(depth.begin<uint16_t>(), depth.end<uint16_t>());
    std::cout<<"Just read image "<<curr_frame_id<<" with size "<<rgb.total()<<"\n";
    std::shared_ptr<cvo::ImageRGBD<uint16_t>> raw(new cvo::ImageRGBD<uint16_t>(rgb, depth_data));

    
    std::shared_ptr<cvo::CvoPointCloud> pc_full(new cvo::CvoPointCloud(*raw,  calib, cvo::CvoPointCloud::FULL));
    std::shared_ptr<cvo::CvoPointCloud> pc_edge_raw(new cvo::CvoPointCloud(*raw, calib, cvo::CvoPointCloud::DSO_EDGES));
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr raw_pcd_edge(new pcl::PointCloud<pcl::PointXYZRGB>);
    pc_edge_raw->export_to_pcd<pcl::PointXYZRGB>(*raw_pcd_edge);
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr raw_pcd_surface(new pcl::PointCloud<pcl::PointXYZRGB>);    
    pc_full->export_to_pcd<pcl::PointXYZRGB>(*raw_pcd_surface);
    
    pcl::PointCloud<pcl::PointXYZRGB> edge_pcl;
    pcl::PointCloud<pcl::PointXYZRGB> surface_pcl;

    float leaf_size = cvo_align.get_params().multiframe_downsample_voxel_size;
    while (true) {
      std::cout<<"Current leaf size is "<<leaf_size<<std::endl;
      edge_pcl.clear();
      surface_pcl.clear();
      cvo::VoxelMap<pcl::PointXYZRGB> edge_voxel(leaf_size / 4); // /10
      cvo::VoxelMap<pcl::PointXYZRGB> surface_voxel(leaf_size);

      
    
      for (int k = 0; k < raw_pcd_edge->size(); k++) {
        edge_voxel.insert_point(&raw_pcd_edge->points[k]);
      }
      std::vector<pcl::PointXYZRGB*> edge_results = edge_voxel.sample_points();

      for (int k = 0; k < edge_results.size(); k++)
        edge_pcl.push_back(*edge_results[k]);
      std::cout<<"edge voxel selected points "<<edge_pcl.size()<<std::endl;
      for (int k = 0; k < raw_pcd_surface->size(); k++) {
        surface_voxel.insert_point(&raw_pcd_surface->points[k]);
      }
      std::vector<pcl::PointXYZRGB*> surface_results = surface_voxel.sample_points();
      for (int k = 0; k < surface_results.size(); k++)
        surface_pcl.push_back(*surface_results[k]);
      std::cout<<"surface voxel selected points "<<surface_pcl.size()<<std::endl;

      int total_selected_pts_num = edge_results.size() + surface_results.size();
      /*if (total_selected_pts_num > 1500)
        leaf_size = leaf_size * 1.2;
      else if (total_selected_pts_num < 500)
        leaf_size = leaf_size * 0.8;
      else
      */
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

    pcl::PointCloud<pcl::PointXYZRGB> pcd_to_save;
    pc->write_to_color_pcd(std::to_string(curr_frame_id)+".pcd");


    
    
    std::cout<<"Load "<<curr_frame_id<<", "<<pc->positions().size()<<" number of points\n"<<std::flush;
    pcs.push_back(pc);
    pcs_full.push_back(pc_full);

    double * poses_data = nullptr;
    //if (BA_poses.size())

    Eigen::Matrix4d id_mat = Eigen::Matrix4d::Identity();
    if (BA_poses.size() == frame_inds.size())
      poses_data = BA_poses[i].data();
    else 
      poses_data = id_mat.data();
    
    cvo::CvoFrame::Ptr new_frame(new cvo::CvoFrameGPU(pc.get(), poses_data));
    cvo::CvoFrame::Ptr new_full_frame(new cvo::CvoFrameGPU(pc_full.get(), poses_data));
    frames.push_back(new_frame);
    frames_full.push_back(new_full_frame);
    id_to_index[curr_frame_id] = i;
  }

  std::cout<<"\n\nFinish loading temporal frames, the number of of points for each frame is ";
  for (int i = 0; i < frames.size(); i++)
    std::cout<<frames[i]->points->size()<<" ";
  std::cout<<"\n\n";


  if (covisMapFile.size() > 0) {
    std::cout<<"Read covis Map "<<covisMapFile<<std::endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr covis_pcd(new pcl::PointCloud<pcl::PointXYZRGB>);
    std::cout<<"Start loading covis pcd...\n";
    pcl::io::loadPCDFile(graph_file_folder + "/" + covisMapFile,
                         *covis_pcd);
    std::cout<<"Just loaded pcd\n";
    std::shared_ptr<cvo::CvoPointCloud> pc(new cvo::CvoPointCloud(*covis_pcd));
    pcs.push_back(pc);
    std::cout<<"Just construct cvopointcloud\n";
    cvo::CvoFrame::Ptr new_frame(new cvo::CvoFrameGPU(pc.get(), frames[0]->pose_vec));
    cvo::CvoFrame::Ptr new_full_frame(new cvo::CvoFrameGPU(pc.get(), frames[0]->pose_vec));
    frames.push_back(new_frame);
    frames_full.push_back(new_frame);      
    std::cout<<"read number points is "<<pc->num_points()<<std::endl<<std::endl;

  }

  
  std::string f_name_full("before_BA_full.pcd");
  std::string f_name("before_BA.pcd");
  write_transformed_pc(frames_full, f_name_full);
  write_transformed_pc(frames, f_name);

  // read edges to construct graph
  std::list<std::pair<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr>> edges;
  std::list<cvo::BinaryState::Ptr> edge_states;
  std::list<cvo::BinaryState::Ptr> edge_states_cpu;
  for (int i = 0; i < edge_inds.size(); i++) {
    int first_ind = id_to_index[edge_inds[i].first];
    int second_ind = id_to_index[edge_inds[i].second];
    std::cout<<"first ind "<<first_ind<<", second ind "<<second_ind<<std::endl;
    std::pair<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr> p(frames[first_ind], frames[second_ind]);
    edges.push_back(p);

    Eigen::Vector3f p_mean_covis = Eigen::Vector3f::Zero();
    Eigen::Vector3f p_mean_tmp = Eigen::Vector3f::Zero();
    for (int k = 0; k < frames[first_ind]->points->num_points(); k++)
      p_mean_tmp = (p_mean_tmp + frames[first_ind]->points->positions()[k]).eval();
    p_mean_tmp = (p_mean_tmp) / frames[first_ind]->points->num_points();    
    for (int k = 0; k <  frames[second_ind]->points->num_points(); k++)
      p_mean_covis = (p_mean_covis + frames[second_ind]->points->positions()[k]).eval();
    p_mean_covis = (p_mean_covis) / frames[second_ind]->points->num_points();
    float dist = (p_mean_covis - p_mean_tmp).norm();
    std::cout<<"dist between tmp and covis is "<<dist<<"\n";
    
    
    const cvo::CvoParams & params = cvo_align.get_params();
    cvo::BinaryStateGPU::Ptr edge_state(new cvo::BinaryStateGPU(std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[first_ind]),
                                                                std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[second_ind]),
                                                             &params,
                                                             cvo_align.get_params_gpu(),
                                                             params.multiframe_num_neighbors,
                                                             params.multiframe_ell_init
                                                             //dist / 3
                                                             ));
    edge_states.push_back((edge_state));

    cvo::BinaryState::Ptr edge_state_cpu(new cvo::BinaryStateCPU((frames[first_ind]),
                                                             (frames[second_ind]),
                                                             &params
                                                                 // ,
                                                              //cvo_align.get_params_gpu(),
                                                                 //params.multiframe_num_neighbors,
                                                                 //params.multiframe_ell_init
                                                             //dist / 3
                                                             ));
    edge_states_cpu.push_back(edge_state_cpu);

    
    Eigen::Matrix<double, 4,4, Eigen::RowMajor > T_source_frame =  Eigen::Matrix<double, 4,4, Eigen::RowMajor >::Identity();
    T_source_frame.block<3,4>(0,0) = Eigen::Map<Eigen::Matrix<double, 3,4, Eigen::RowMajor>>(frames[first_ind]->pose_vec);
    Eigen::Matrix<double, 4,4, Eigen::RowMajor > T_target_frame =  Eigen::Matrix<double, 4,4, Eigen::RowMajor >::Identity();
    T_target_frame.block<3,4>(0,0) = Eigen::Map<Eigen::Matrix<double, 3,4, Eigen::RowMajor>>(frames[second_ind]->pose_vec);
    Eigen::Matrix4f TT = (T_target_frame.inverse() * T_source_frame).cast<float>();
    std::cout<<"cos between "<<first_ind<<" and "<<second_ind<<" is "<<cvo_align.function_angle(
										      *frames[first_ind]->points,
										      *frames[second_ind]->points,
										      

										      TT,
                                                                                      1.0,
										      false
										      
										      
												)<<std::endl;
    
  }

  if (covisMapFile.size()) {
    for (int i = 0; i < frames.size() - 1; i++) {
      std::pair<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr> p(frames[i], frames[frames.size()-1]);
      edges.push_back(p);

      Eigen::Vector3f p_mean_covis = Eigen::Vector3f::Zero();
      Eigen::Vector3f p_mean_tmp = Eigen::Vector3f::Zero();
      for (int k = 0; k < frames[i]->points->num_points(); k++)
        p_mean_tmp = (p_mean_tmp + frames[i]->points->positions()[k]).eval();
      p_mean_tmp = (p_mean_tmp) / frames[i]->points->num_points();    
      for (int k = 0; k <  frames[frames.size()-1]->points->num_points(); k++)
        p_mean_covis = (p_mean_covis + frames[frames.size()-1]->points->positions()[k]).eval();
      p_mean_covis = (p_mean_covis) / frames[frames.size()-1]->points->num_points();
      float dist = (p_mean_covis - p_mean_tmp).norm();
      std::cout<<"dist between tmp and covis is "<<dist<<"\n";
      
      const cvo::CvoParams & params = cvo_align.get_params();
      cvo::BinaryStateGPU::Ptr edge_state(new cvo::BinaryStateGPU(std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[i]),
                                                                  std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[frames.size()-1]),
                                                                  &params,
                                                                  cvo_align.get_params_gpu(),
                                                                  params.multiframe_num_neighbors,
                                                                  params.multiframe_ell_init * 4
                                                                  //dist / 4
                                                                  ));
      edge_states.push_back((edge_state));
      
      Eigen::Matrix<double, 4,4, Eigen::RowMajor > T_source_frame =  Eigen::Matrix<double, 4,4, Eigen::RowMajor >::Identity();
      T_source_frame.block<3,4>(0,0) = Eigen::Map<Eigen::Matrix<double, 3,4, Eigen::RowMajor>>(frames[i]->pose_vec);
      Eigen::Matrix<double, 4,4, Eigen::RowMajor > T_target_frame =  Eigen::Matrix<double, 4,4, Eigen::RowMajor >::Identity();
      T_target_frame.block<3,4>(0,0) = Eigen::Map<Eigen::Matrix<double, 3,4, Eigen::RowMajor>>(frames[frames.size()-1]->pose_vec);
      Eigen::Matrix4f TT = (T_target_frame.inverse() * T_source_frame).cast<float>();
      std::cout<<"cos between "<<i<<" and covidMap is "<<cvo_align.function_angle(
												  *frames[i]->points,
												  *frames[frames.size()-1]->points,
												  

												  TT,
                                                                                                  1.0,
												  false
										      
										      
										  )<<std::endl;
      
      
    }
  }
  

  double time = 0;
  std::vector<bool> const_flags(frames.size(), false);
  std::cout<<"Const frames include ";
  if (covisMapFile.size())
    //const_flags[const_flags.size()-1] = true;
    const_flags[0] = true;
  else {
    for (int i = 0; i < num_const_frames; i++) {
      const_flags[i] = true;
      std::cout<<frame_inds[i]<<", ";
    }
    std::cout<<std::endl;
  }


  cvo_align.align(frames, const_flags,
                  edges, &time);

  std::cout<<"Align ends. Total time is "<<time<<" seconds."<<std::endl;
  f_name="after_BA.pcd";
  f_name_full="after_BA_full.pcd";
  for (int i = 0; i < frames.size(); i++) {
    memcpy(frames_full[i]->pose_vec, frames[i]->pose_vec, sizeof(double)*12);
  }
  write_transformed_pc(frames_full, f_name_full);
  write_transformed_pc(frames, f_name);

  f_name="const_BA.pcd";
  f_name_full="const_BA_full.pcd";
  write_transformed_pc(frames_full, f_name_full, num_const_frames);
  write_transformed_pc(frames, f_name, num_const_frames);

  tracking_subset_poses_f.close();

  std::string traj_out("traj_out.txt");
  write_traj_file(traj_out,
                  timestamps,
                  frames );
 
  
  return 0;
}
