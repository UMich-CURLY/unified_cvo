#include <ios>
#include <iostream>
#include <list>
#include <vector>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <set>
#include <Eigen/Dense>
#include "cvo/CvoGPU.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoFrame.hpp"
#include "dataset_handler/KittiHandler.hpp"
#include "utils/ImageStereo.hpp"
#include "utils/Calibration.hpp"
#include "utils/VoxelMap.hpp"
#include "cvo/IRLS_State_CPU.hpp"
#include "cvo/IRLS_State_GPU.hpp"
#include "cvo/IRLS_State.hpp"
#include "cvo/IRLS.hpp"
#include "cvo/CvoFrame.hpp"
#include "cvo/CvoFrameGPU.hpp"
#include "utils/ImageDownsampler.hpp"
#include "utils/LidarPointDownsampler.hpp"

//using namespace std;

template class cvo::VoxelMap<const cvo::CvoPoint>;

void write_traj_file(std::string & fname,
                     std::vector<cvo::CvoFrame::Ptr> & frames ) {
  std::ofstream outfile(fname);
  for (int i = 0; i< frames.size(); i++) {
    cvo::CvoFrame::Ptr ptr = frames[i];
    outfile << ptr->pose_vec[0] <<" "
            << ptr->pose_vec[1] <<" "
            << ptr->pose_vec[2] <<" "
            << ptr->pose_vec[3] <<" "
            << ptr->pose_vec[4] <<" "
            << ptr->pose_vec[5] <<" "
            << ptr->pose_vec[6] <<" "
            << ptr->pose_vec[7] <<" "
            << ptr->pose_vec[8] <<" "
            << ptr->pose_vec[9] <<" "
            << ptr->pose_vec[10] <<" "
            << ptr->pose_vec[11] <<"\n";

  }
  outfile.close();
}


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



void read_graph_file(std::string &graph_file_path,
                     std::vector<int> & frame_inds,
                     std::vector<std::pair<int, int>> & edges) {
  std::ifstream graph_file(graph_file_path);
  
  int num_frames;
  graph_file>>num_frames;
  //frame_inds.resize(num_frames);
  std::cout<<"Frame indices include ";
  for (int i = 0; i < num_frames; i++) {
    graph_file >> frame_inds[i];
    std::cout<<frame_inds[i]<<", ";
  }
  std::cout<<"\nEdges include ";

  while (!graph_file.eof()) {
    std::pair<int, int> p;
    graph_file >> p.first >> p.second;
    edges.push_back(p);
    std::cout<<"("<<p.first<<", "<<p.second <<"), ";
  }
  std::cout<<"\n";
  graph_file.close();  
}



void read_pose_file(std::string & gt_fname,
                    std::vector<int> & frame_inds,
                    std::vector<Eigen::Matrix4d,
                    Eigen::aligned_allocator<Eigen::Matrix4d>> & poses_all) {

  poses_all.clear();//reserve(frame_inds.size());
  
  std::ifstream gt_file(gt_fname);

  std::cout<<"Read "<<gt_fname<<std::endl;

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
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> Tr(pose_v);
    Eigen::Matrix<double, 3, 4, Eigen::ColMajor> T34 = Tr;
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,4>(0,0) = T34;
    //poses.push_back(T);

    poses_all.push_back(T);        

    //std::cout<<"Read pose "<<pose<<std::endl;
    //if (curr_frame_ind == 2) {
      std::cout<<"read: line "<<frame_inds[curr_frame_ind]<<" pose is "<<poses_all[curr_frame_ind]<<std::endl;
      //}

    
    line_ind ++;
    curr_frame_ind++;
    if (curr_frame_ind == frame_inds.size()) break;
  }


  gt_file.close();
}


void write_transformed_pc(std::vector<cvo::CvoFrame::Ptr> & frames, std::string & fname) {
  pcl::PointCloud<pcl::PointXYZ> pc_all;
  for (auto ptr : frames) {
    cvo::CvoPointCloud new_pc;
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3,4>(0,0) = Eigen::Map<cvo::Mat34d_row>(ptr->pose_vec);
    
    Eigen::Matrix4f pose_f = pose.cast<float>();
    cvo::CvoPointCloud::transform(pose_f, *ptr->points, new_pc);

    pcl::PointCloud<pcl::PointXYZ> pc_curr;
    new_pc.export_to_pcd(pc_curr);

    pc_all += pc_curr;
  }
  pcl::io::savePCDFileASCII(fname, pc_all);
}

int main(int argc, char** argv) {

  std::string kitti_type(argv[1]);
  std::string kitti_path(argv[2]);
  
  std::unique_ptr<cvo::KittiHandler> kitti;
  cvo::KittiHandler::DataType dtype;
  if (std::strcmp(kitti_type.c_str(), "stereo") == 0) {
    dtype = cvo::KittiHandler::DataType::STEREO;
  } else if (std::strcmp(kitti_type.c_str(), "lidar") == 0) {
    dtype = cvo::KittiHandler::DataType::LIDAR;
  } else {
    ASSERT(false, "unknown dtype");
  }
  std::cout<<"dtype is "<<dtype<<"\n";
     
  kitti.reset(new cvo::KittiHandler(kitti_path, dtype));
  std::string cvo_param_file(argv[3]);  
  std::string graph_file_name(argv[4]);
  std::string tracking_fname(argv[5]);
  std::string gt_fname(argv[6]);

  int total_iters = kitti->get_total_number();
  std::string calib_file;
  calib_file =  kitti_path +"/cvo_calib.txt"; 
  cvo::Calibration calib(calib_file, cvo::Calibration::STEREO);

  cvo::CvoGPU cvo_align(cvo_param_file );
  
  std::vector<int> frame_inds;
  std::vector<std::pair<int, int>> edge_inds;
  std::vector<cvo::Mat34d_row, Eigen::aligned_allocator<cvo::Mat34d_row>> BA_poses; 
  read_graph_file(graph_file_name, frame_inds, edge_inds, BA_poses);


  std::vector<cvo::Mat34d_row, Eigen::aligned_allocator<cvo::Mat34d_row>> tracking_poses;
  //std::vector<cvo::Mat34d_row, Eigen::aligned_allocator<cvo::Mat34d_row>> gt_poses;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> gt_poses;
  //cvo::read_pose_file_kitti_format(gt_fname, int start_frame, int last_frame, int)
  read_pose_file(gt_fname, frame_inds, gt_poses);
  std::ofstream gt_subset_poses_f("gt_poses.txt");
  for (auto && gt_p : gt_poses) {
    std::cout<<"gt p" << gt_p<<std::endl;
    gt_subset_poses_f << gt_p(0,0) << " "<< gt_p(0,1) << " "<< gt_p(0,2) << " "<< gt_p(0,3) << " "<< gt_p(1,0) << " "<< gt_p(1,1) << " "<< gt_p(1,2) << " "<< gt_p(1,3) << " "<< gt_p(2,0) << " "<< gt_p(2,1) << " "<< gt_p(2,2) << " "<< gt_p(2,3) << "\n";
  }
  gt_subset_poses_f.close();

  // read point cloud
  std::vector<cvo::CvoFrame::Ptr> frames;
  //std::vector<cvo::CvoFrame::Ptr> frames_full;  
  std::vector<std::shared_ptr<cvo::CvoPointCloud>> pcs;
  //std::vector<std::shared_ptr<cvo::CvoPointCloud>> pcs_full;  
  std::unordered_map<int, int> id_to_index;
  for (int i = 0; i<frame_inds.size(); i++) {

    int curr_frame_id = frame_inds[i];
    kitti->set_start_index(curr_frame_id);

    std::shared_ptr<cvo::CvoPointCloud> pc;
    if (dtype == cvo::KittiHandler::DataType::STEREO) {
      cv::Mat left, right;
      //vector<float> semantics_target;
      //if (kitti->read_next_stereo(left, right, 19, semantics_target) != 0) {
      if (kitti->read_next_stereo(left, right) != 0) {
        break;
      }
      pc = cvo::stereo_downsampling(left, right, calib, cvo_align.get_params().multiframe_downsample_voxel_size);
    } else {
      pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in(new pcl::PointCloud<pcl::PointXYZI>);
      if (kitti->read_next_lidar(pc_in) != 0) {
        break;
      }
      pc = cvo::downsample_lidar_points(false, pc_in, cvo_align.get_params().multiframe_downsample_voxel_size);
    }
    pcs.push_back(pc);
    // pcs_full.push_back(pc_full);

    double * poses_data = nullptr;
    //if (BA_poses.size())

    Eigen::Matrix4d id_mat = Eigen::Matrix4d::Identity();
    if (BA_poses.size() == frame_inds.size())
      poses_data = BA_poses[i].data();
    else 
      poses_data = id_mat.data();

    
    cvo::CvoFrame::Ptr new_frame(new cvo::CvoFrameGPU(pc.get(), poses_data));
    frames.push_back(new_frame);
    id_to_index[curr_frame_id] = i;
    std::cout<<"Finish constructing "<<curr_frame_id<<"\n";
  }
  std::string f_name("before_BA.pcd");
  write_transformed_pc(frames, f_name);
  std::string tracking_subset_poses_fname("cvo_track_poses.txt");
  write_traj_file(tracking_subset_poses_fname,
                  frames );

  
  std::cout<<"build edges\n"<<std::flush;
  std::list<std::pair<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr>> edges;
  std::list<cvo::BinaryState::Ptr> edge_states;
  //std::list<cvo::BinaryState::Ptr> edge_states_cpu;
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

  double time = 0;
  std::vector<bool> const_flags(frames.size(), false);
  const_flags[0] = true;
  
  auto start = std::chrono::system_clock::now();
  cvo::CvoBatchIRLS batch_irls_problem(frames, const_flags,
                                       edge_states, &cvo_align.get_params());
  std::string err_file("err_wrt_iters.txt");
  ASSERT(gt_poses.size() == frames.size(), "gt pose size is "+std::to_string(gt_poses.size()));
  batch_irls_problem.solve(gt_poses, err_file);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> t_all = end - start;  
  std::cout<<"GPU Align ends. Total time is "<<double(t_all.count()) / 1000<<" seconds."<<std::endl;

  f_name="after_BA.pcd";
  //std::string f_name_full="after_BA_full.pcd";
  //for (int i = 0; i < frames.size(); i++) {
  //  memcpy(frames_full[i]->pose_vec, frames[i]->pose_vec, sizeof(double)*12);
  // }
  //write_transformed_pc(frames_full, f_name_full);
  write_transformed_pc(frames, f_name);
  

  std::string traj_out("traj_out.txt");
  write_traj_file(traj_out,
                  frames );
 
  

  return 0;
}
