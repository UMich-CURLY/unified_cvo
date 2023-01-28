#include <Eigen/Geometry>
#include <Eigen/src/Core/util/Constants.h>
#include <iostream>
#include <list>
#include <cmath>
#include <fstream>
#include <filesystem>
#include "utils/def_assert.hpp"
#include "utils/GassianMixture.hpp"
#include "utils/eigen_utils.hpp"
#include <pcl/common/io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/impl/point_types.hpp>
#include <vector>
#include <utility>
#include <random>
#include <string>
#include <fstream>
#include <sstream>
#include <set>
#include <Eigen/Dense>
#include "cvo/CvoGPU.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoFrame.hpp"
#include "cvo/CvoFrameGPU.hpp"
#include "cvo/IRLS_State.hpp"
#include "cvo/IRLS_State_GPU.hpp"
#include "utils/VoxelMap.hpp"
#include "utils/VoxelMap_impl.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/io/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <filesystem>
namespace fs = std::filesystem;

void write_transformed_pc(std::vector<cvo::CvoFrame::Ptr> & frames, std::string & fname,
                          std::vector<std::tuple<uint8_t, uint8_t, uint8_t> > & colors) {
  pcl::PointCloud<pcl::PointXYZRGB> pc_all;
  for (int i = 0; i <  frames.size(); i++) {
    auto ptr = frames[i];
    cvo::CvoPointCloud new_pc(0,0);
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3,4>(0,0) = Eigen::Map<cvo::Mat34d_row>(ptr->pose_vec);
    
    Eigen::Matrix4f pose_f = pose.cast<float>();
    cvo::CvoPointCloud::transform(pose_f, *ptr->points, new_pc);

    pcl::PointCloud<pcl::PointXYZ> pc_curr_xyz;
    new_pc.export_to_pcd(pc_curr_xyz);
    pcl::PointCloud<pcl::PointXYZRGB> pc_curr;
    for (int j = 0; j <  pc_curr_xyz.size(); j++) {
      pcl::PointXYZRGB p;
      p.getVector3fMap() = pc_curr_xyz[j].getVector3fMap();
      p.r = std::get<0>(colors[i]);
      p.g = std::get<1>(colors[i]);
      p.b = std::get<2>(colors[i]);
      pc_curr.push_back(p);
    }

    pc_all += pc_curr;
  }
  pcl::io::savePCDFileASCII(fname, pc_all);
}



template <typename PointT>
void add_gaussian_mixture_noise(pcl::PointCloud<pcl::PointNormal> & input,
                                pcl::PointCloud<PointT> & output,
                                float ratio,
                                float sigma,
                                float uniform_range,
                                bool is_using_viewpoint) {

  cvo::GaussianMixtureDepthGenerator gaussion_mixture(ratio, sigma, uniform_range);

  output.resize(input.size());
  for (int i = 0; i < input.size(); i++) {
    if (is_using_viewpoint) {
      // using a far away view point
    } else {
      // using normal direction
      auto pt = input[i];
      Eigen::Vector3f normal_dir;
      normal_dir << pt.normal_x, pt.normal_y, pt.normal_z;
      Eigen::Vector3f center_pt = pt.getVector3fMap();
      Eigen::Vector3f result = gaussion_mixture.sample(center_pt, normal_dir);
      pcl::PointXYZ new_pt;
      output[i].getVector3fMap() = result;
      if (i == 0) {
        std::cout<<"transform "<<pt.getVector3fMap().transpose()<<" to "<<result.transpose()<<std::endl;
      }
    }
  }

  
}

void gen_random_poses(std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> & poses, int num_poses,
                      float max_angle_axis // In degrees
                      ) {
  poses.resize(num_poses);
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<> dist(0, max_angle_axis);
  std::uniform_real_distribution<> dist_trans(0,0.3);
  for (int i = 0; i < num_poses; i++) {
    if (i != 0) {
      Eigen::Matrix3f rot;
      Eigen::Vector3f axis;
      axis << (float)dist_trans(e2), (float)dist_trans(e2), (float)dist_trans(e2);
      axis = axis.normalized().eval();
      
      //rot = Eigen::AngleAxisf(dist(e2)*M_PI, Eigen::Vector3f::UnitX())
      //  * Eigen::AngleAxisf(dist(e2)*M_PI,  Eigen::Vector3f::UnitY())
      //  * Eigen::AngleAxisf(dist(e2)*M_PI, Eigen::Vector3f::UnitZ());
      rot = Eigen::AngleAxisf(max_angle_axis * 0.017453 , axis);
      poses[i] = Eigen::Matrix4f::Identity();
      poses[i].block<3,3>(0,0) = rot;
      poses[i](0,3) = dist_trans(e2);
      poses[i](1,3) = dist_trans(e2);
      poses[i](2,3) = dist_trans(e2);
    } else {
      poses[i] = Eigen::Matrix4f::Identity();
    }
    //rot = Eigen::AngleAxisf(dist(e2)*M_PI, axis );


    std::cout<<"random pose "<<i<<" is \n"<<poses[i]<<"\n";
  }
}
template <typename T>
void pcd_rescale(typename pcl::PointCloud<T>::Ptr pcd ){
  Eigen::MatrixXf pcd_eigen(3, pcd->size());
  for (int j = 0; j < pcd->size(); j++) {
    pcd_eigen.col(j) = pcd->at(j).getVector3fMap();
  }

  float scale = (pcd_eigen.rowwise().maxCoeff() - pcd_eigen.rowwise().minCoeff()).norm();
  std::cout << "scale = " << scale << std::endl;
  pcd_eigen /= scale;

  for (int j = 0; j < pcd->size(); j++)
    pcd->at(j).getVector3fMap() = pcd_eigen.col(j);
}


void gen_rand_colors(  std::vector<std::tuple<uint8_t, uint8_t, uint8_t> > & colors) {
  for (auto && color_curr_frame : colors) {
    std::get<0>(color_curr_frame) = (uint8_t)(rand() % 255);
    std::get<1>(color_curr_frame) = (uint8_t)(rand() % 255);
    std::get<2>(color_curr_frame) = (uint8_t)(rand() % 255);
  }
}

void eval_poses_se3_norm(std::vector<Sophus::SE3f> & estimates,
                         std::vector<Sophus::SE3f> & gt,
                         std::string & fname
                         ){
  assert(std::filesystem::exists(fname));
  std::ofstream err_f(fname,std::fstream::out |   std::ios::app);
  float total_err = 0;
  for (int i = 0; i < gt.size(); i ++) {
    float err = (estimates[i].inverse() * gt[i]).log().norm();
    total_err += err;
  }
  err_f << total_err<<"\n";
  err_f.close();
  //std::cout<<"Total: "<<counter<<" success out of "<<gt.size()<<"\n";
}


void mat_to_row(const Sophus::SE3f & mat,
                std::vector<float> & row) {
  row.resize(16);
  Eigen::Matrix4f m = mat.matrix();
  for (int i = 0; i < 16; i++) {
    row[i] = m(i / 4, i % 4);
  }
  
}

void eval_poses_SE3_frobenius_norm(std::vector<Sophus::SE3f> & estimates,
                                   std::vector<Sophus::SE3f> & gt,
                                   std::string & err_fname,
                                   std::string & err_framewise_fname,
                                   std::string & pose_fname
                                   ){
  assert(std::filesystem::exists(fname));
  std::ofstream err_f(err_fname,std::fstream::out |   std::ios::app);
  std::ofstream err_framewise(err_framewise_fname);
  std::ofstream pose_f(pose_fname);
  float total_err = 0;
  for (int i = 0; i < gt.size(); i ++) {
    /// element wise norm 
    float err = ((estimates[i].inverse() * gt[i]).matrix() - Eigen::Matrix4f::Identity()).norm();
    err_framewise << err<<"\n";

    /// log poses
    std::vector<float> mat_row_major;
    mat_to_row(estimates[i], mat_row_major);
    for (int j = 0; j < mat_row_major.size(); j++ )
      pose_f << mat_row_major[j]<<" ";
    pose_f << "\n";

    total_err += err;
  }
  err_f << total_err<<"\n";
  err_f.close();
  err_framewise.close();
  pose_f.close();
  //std::cout<<"Total: "<<counter<<" success out of "<<gt.size()<<"\n";
}


int main(int argc, char** argv) {

  omp_set_num_threads(24);
  std::string in_pcd_fname(argv[1]);
  std::string cvo_param_file(argv[2]);
  int num_frames = std::stoi(argv[3]);
  float max_angle_per_axis = std::stof(argv[4]);
  int num_runs = std::stoi(argv[5]);
  std::string exp_folder(argv[6]);
  int is_adding_outliers = std::stoi(argv[7]);
  float ratio = 0.8;
  float sigma = 0.01;
  float uniform_range = 0.5;
  cvo::CvoGPU cvo_align(cvo_param_file);
  if (is_adding_outliers) {
    ratio = std::stof(argv[8]); /// affects the outlier ratio
    sigma = std::stof(argv[9]); /// affects the noise scale value size
    uniform_range = std::stof(argv[10]); /// affets
  }
//   int is_logging_data_only = std::stoi(argv[11]);

//   std::vector<std::tuple<uint8_t, uint8_t, uint8_t> > colors(num_frames);
//   gen_rand_colors(colors);

//   cvo::CvoGPU cvo_align(cvo_param_file);

  /// load input point cloud (transfomed from prev step)
  pcl::PointCloud<pcl::PointXYZ>::Ptr raw_pcd(new pcl::PointCloud<pcl::PointXYZ>);
  // Add function to load 4 different point cloud 
  std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> raw_pcd_normal;


  /// experiment result folder
  fs::path exp_folder_dir(exp_folder);
  fs::create_directories(exp_folder_dir);
  std::string full_error_fname(exp_folder + "/cvo_err_bunny.txt");
  std::ofstream err_f(full_error_fname);
  err_f.close();
  std::string full_time_fname(exp_folder + "/cvo_time_bunny.txt");
  std::ofstream err_time_f(full_time_fname);
  err_time_f.close();

  /// start running 
  for (int k = 0; k < num_runs; k++) {


    // /// convert to Sophus format for ground truth
    // std::vector<Sophus::SE3f> poses_gt;
    // std::transform(tracking_poses.begin(), tracking_poses.end(), std::back_inserter(poses_gt),
    //                [&](const Eigen::Matrix4f & in){
    //                  Sophus::SE3f pose(in);
    //                  return pose.inverse();
    //                });


    /// generate point cloud
    std::vector<cvo::CvoFrame::Ptr> frames;
    std::vector<std::shared_ptr<cvo::CvoPointCloud>> pcs;
    std::string foldername = in_pcd_fname + '/' +std::to_string(k) + '/';
    for (int i = 0; i < num_frames; i++) {
        pcl::PointCloud<pcl::PointNormal>::Ptr dst(new pcl::PointCloud<pcl::PointNormal>);
        std::string fullName = foldername  + std::to_string(i) + "normal.pcd";
        std::cout << "load point cloud " << fullName << std::endl;
        pcl::io::loadPCDFile (fullName, *dst);
        pcl::PointCloud<pcl::PointXYZ>::Ptr raw_pcd(new pcl::PointCloud<pcl::PointXYZ>);
        copyPointCloud(*dst, *raw_pcd);
        std::shared_ptr<cvo::CvoPointCloud> pc (new cvo::CvoPointCloud(*raw_pcd));  
        cvo::Mat34d_row pose;
        cvo::CvoFrame::Ptr new_frame(new cvo::CvoFrameGPU(pc.get(), pose.data()));
        frames.push_back(new_frame);
        pcs.push_back(pc);
    }


    // load gt
    std::string gtfilename = foldername + "gt_poses.txt";
    std::ifstream gt_file(gtfilename);
    std::cout <<"read pose file : --- " << gtfilename << std::endl;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> tracking_poses;
    tracking_poses.resize(4);
    for (int pose_index = 0; pose_index < 4; pose_index++)
    {
        
        std::vector<float> numbers; 
        float num; 
        for (int t = 0; t < 16; t++){
          gt_file >> num;
          numbers.push_back(num);
        }
        std::cout <<"ground truth pose is " << std::endl;
        int index = 0;
        for (int l = 0; l < 4; l ++){
            for (int r = 0; r < 4; r++){
                tracking_poses[pose_index](l,r) = numbers[index]; 
                // std::cout << tracking_poses[pose_index](l,r) << " ";
                index++;
            }
            std::cout << std::endl;
        }

    }
    gt_file.close();

    /// convert to Sophus format for ground truth
    std::vector<Sophus::SE3f> poses_gt;
    std::transform(tracking_poses.begin(), tracking_poses.end(), std::back_inserter(poses_gt),
                   [&](const Eigen::Matrix4f & in){
                     Sophus::SE3f pose(in);
                     return pose.inverse();
                   });
    // running cvo
    
    std::cout<<"Start constructing cvo edges\n";
    std::list<cvo::BinaryState::Ptr> edge_states;            
    for (int i = 0; i < num_frames; i++) {
      for (int j = i+1; j < num_frames; j++ ) {
        //std::pair<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr> p(frames[i], frames[j]);
        //edges.push_back(p);
        const cvo::CvoParams & params = cvo_align.get_params();
        cvo::BinaryStateGPU::Ptr edge_state(new cvo::BinaryStateGPU(std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[i]),
                                                                    std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[j]),
                                                                    &params,
                                                                    cvo_align.get_params_gpu(),
                                                                    params.multiframe_num_neighbors,
                                                                    params.multiframe_ell_init
                                                                    //dist / 3
                                                                    ));
        edge_states.push_back((edge_state));
      
      }
    }
    std::string exp_curr_dir_str = in_pcd_fname + '/' +std::to_string(k);
    std::cout<<"start align\n";
    std::vector<bool> const_flags(frames.size(), false);
    const_flags[0] = true;
    double time = 0;
    cvo_align.align(frames, const_flags,
                    edge_states,  &time, nullptr);

    /// log time
    std::ofstream err_time_f(full_time_fname, std::ofstream::out | std::ofstream::app);
    err_time_f << time <<"\n";
    err_time_f.close();
    
    /// log resulting stacked point cloud
    std::vector<std::tuple<uint8_t, uint8_t, uint8_t> > colors(num_frames);
    gen_rand_colors(colors);
    std::string f_name;
    f_name=exp_curr_dir_str + "/after_BA_bunny_" + std::to_string(k)+".pcd";
    write_transformed_pc(frames, f_name, colors);

    /// create sophus 
    std::vector<Sophus::SE3f> estimates;
    std::transform(frames.begin(), frames.end(), std::back_inserter(estimates),
                   [&](auto & frame){
                     Eigen::Matrix<double, 3, 4, Eigen::RowMajor> pose_row = Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(frame->pose_vec);
                     Eigen::Matrix<double , 3,4> pose_col = pose_row;
                     Eigen::Matrix<float, 4, 4> pose_eigen = Eigen::Matrix4f::Identity();
                     pose_eigen.block<3,4>(0,0) = pose_col.cast<float>();
                     Sophus::SE3f pose(pose_eigen);
                     return pose;
                   });


    f_name =  exp_curr_dir_str + "/gt_BA_bunny_"+std::to_string(k)+".pcd" ;
    std::string pose_fname = exp_curr_dir_str + "/rkhs_results.txt";
    std::string err_curr_fname = exp_curr_dir_str + "/error_rksh_results.txt";
    eval_poses_SE3_frobenius_norm(estimates,
                                  poses_gt,
                                  full_error_fname,
                                  err_curr_fname,
                                  pose_fname
                                  );


  }
  return 0;
}
