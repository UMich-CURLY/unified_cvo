#include <Eigen/Geometry>

#include <Eigen/src/Core/util/Constants.h>
#include <iostream>
#include <list>
#include <cmath>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <boost/filesystem.hpp>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/random_sample.h>

//#include <experimental/filesystem>
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

namespace fs = boost::filesystem;

  template <typename PointT>
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr calculate_fpfh_pcl(typename pcl::PointCloud< PointT>::Ptr cloud,
                                                                float normal_radius,
                                                                float fpfh_radius){

    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal> ());

    // Create the normal estimation class, and pass the input dataset to it
    typename pcl::NormalEstimation<PointT, pcl::Normal> ne;
    ne.setInputCloud (cloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    typename pcl::search::KdTree<PointT>::Ptr tree_normal (new typename pcl::search::KdTree<PointT> ());
    ne.setSearchMethod (tree_normal);

    // Output datasets
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

    // Use all neighbors in a sphere of radius 3cm
    ne.setRadiusSearch (normal_radius);
    
    // Compute the features
    ne.compute (*cloud_normals);


    // Create the FPFH estimation class, and pass the input dataset+normals to it

    typename  pcl::FPFHEstimation<PointT, pcl::Normal, pcl::FPFHSignature33> fpfh;

    fpfh.setInputCloud (cloud);
    fpfh.setInputNormals (cloud_normals);

    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    typename  pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    fpfh.setSearchMethod (tree);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs (new pcl::PointCloud<pcl::FPFHSignature33> ());
    // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!

    fpfh.setRadiusSearch (fpfh_radius);

    // Compute the features
    fpfh.compute (*fpfhs);

    return fpfhs;
  }




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
                      float max_angle_axis=1.0 // max: [0, 1)
                      ) {
  poses.resize(num_poses);
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<> dist(0, max_angle_axis);
  std::uniform_real_distribution<> dist_trans(0,1);
  for (int i = 0; i < num_poses; i++) {
    if (i != 0) {
      Eigen::Matrix3f rot;
      Eigen::Vector3f axis;
      axis << (float)dist_trans(e2), (float)dist_trans(e2), (float)dist_trans(e2);
      axis = axis.normalized().eval();
      
      //rot = Eigen::AngleAxisf(dist(e2)*M_PI, Eigen::Vector3f::UnitX())
      //  * Eigen::AngleAxisf(dist(e2)*M_PI,  Eigen::Vector3f::UnitY())
      //  * Eigen::AngleAxisf(dist(e2)*M_PI, Eigen::Vector3f::UnitZ());
      rot = Eigen::AngleAxisf(max_angle_axis * M_PI, axis);
      poses[i] = Eigen::Matrix4f::Identity();
      poses[i].block<3,3>(0,0) = rot;
      poses[i](0,3) = dist_trans(e2);
      poses[i](1,3) = dist_trans(e2);
      poses[i](2,3) = dist_trans(e2);
      if (i > 0)
        poses[i] = (poses[i-1] * poses[i]).eval();
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

void eval_poses(std::vector<Sophus::SE3f> & estimates,
                std::vector<Sophus::SE3f> & gt,
                std::string & fname
                ){
  assert(fs::exists(fname));
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

int main(int argc, char** argv) {

  omp_set_num_threads(24);
  std::string in_pcd_fname(argv[1]);
  std::string cvo_param_file(argv[2]);
  int num_frames = std::stoi(argv[3]);
  float max_angle_per_axis = std::stof(argv[4]);
  int num_runs = std::stoi(argv[5]);
  std::string exp_folder(argv[6]);

  // float normal_radius = std::stof(argv[11]);
  float fpfh_radius = std::stof(argv[7]);
  
  int is_adding_outliers = std::stoi(argv[8]);
  
  float ratio = 0.8;
  float sigma = 0.01;
  float uniform_range = 0.5;
  if (is_adding_outliers) {
    ratio = std::stof(argv[9]);
    sigma = std::stof(argv[10]);
    uniform_range = std::stof(argv[11]);
  }

  std::vector<std::tuple<uint8_t, uint8_t, uint8_t> > colors(num_frames);
  gen_rand_colors(colors);


  cvo::CvoGPU cvo_align(cvo_param_file);

  pcl::PointCloud<pcl::PointXYZ>::Ptr raw_pcd(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointNormal>::Ptr raw_pcd_normal(new pcl::PointCloud<pcl::PointNormal>);
  pcl::io::loadPCDFile<pcl::PointNormal> (in_pcd_fname, *raw_pcd_normal);

  fs::path exp_folder_dir(exp_folder);
  fs::create_directories(exp_folder_dir);
  std::string fname(exp_folder + "/cvo_err_bunny.txt");
  std::ofstream err_f(fname);
  err_f.close();
  for (int k = 0; k < num_runs; k++) {
    fs::path exp_curr_dir = exp_folder_dir / fs::path(std::to_string(k));
    std::string exp_curr_dir_str = exp_curr_dir.string();
    std::cout<<"Current exp folder is "<<exp_curr_dir_str<<"\n";
    fs::create_directories(exp_curr_dir);

    
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> tracking_poses;
    gen_random_poses(tracking_poses, num_frames, max_angle_per_axis );
    std::ofstream pose_f(exp_curr_dir_str + "/gt_poses.txt");
    for (int l = 0; l < tracking_poses.size(); l++ ) {
      std::string pose_curr = cvo::mat_to_line<float, Eigen::ColMajor>(tracking_poses[l]);
      pose_f << pose_curr<<std::endl;
    }
    pose_f.close();
       


    std::vector<Sophus::SE3f> poses_gt;
    std::transform(tracking_poses.begin(), tracking_poses.end(), std::back_inserter(poses_gt),
                   [&](const Eigen::Matrix4f & in){
                     Sophus::SE3f pose(in);
                     return pose.inverse();
                   });


    // read point cloud
    std::vector<cvo::CvoFrame::Ptr> frames;
    std::vector<std::shared_ptr<cvo::CvoPointCloud>> pcs;
    for (int i = 0; i<num_frames; i++) {
      // Create the filtering object

      pcl::PointCloud<pcl::PointNormal>::Ptr raw_pcd_curr;//(new pcl::PointCloud<pcl::PointXYZ>);

      
      if (is_adding_outliers) {
        raw_pcd_curr.reset(new pcl::PointCloud<pcl::PointNormal>);
        add_gaussian_mixture_noise(*raw_pcd_normal,
                                   *raw_pcd_curr,
                                   ratio,
                                   sigma,
                                   uniform_range,
                                   false
                                   );
      } else {
        raw_pcd_curr = raw_pcd_normal;
      }
      

      // for visualzation
      copyPointCloud(*raw_pcd_curr, *raw_pcd);
      pcl::io::savePCDFileASCII(exp_curr_dir_str + "/" + std::to_string(i)+"normal.pcd", *raw_pcd);      
      pcd_rescale<pcl::PointXYZ>(raw_pcd);
      
    
      
      pcl::PointCloud<pcl::PointXYZ>::Ptr raw_pcd_transformed(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::transformPointCloud (*raw_pcd, *raw_pcd_transformed, tracking_poses[i]);
      pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh = calculate_fpfh_pcl<pcl::PointXYZ>(raw_pcd_transformed, 0.03, 0.05);

      /*
      pcl::VoxelGrid<pcl::PointXYZ> sor;
      sor.setInputCloud (raw_pcd_transformed);
      sor.setLeafSize (0.025f, 0.025f, 0.025f);
      sor.filter (*raw_pcd_transformed);
      */
      pcl::PointCloud<pcl::PointXYZ>::Ptr raw_pcd_downsampled(new pcl::PointCloud<pcl::PointXYZ>);      
      pcl::RandomSample <pcl::PointXYZ> random;
      random.setInputCloud(raw_pcd_transformed);
      random.setSeed (std::rand ());
      random.setSample((unsigned int)(1024));
      std::vector<int> ind_downsampled(1024);
      random.filter(ind_downsampled);
      for (auto m : ind_downsampled)
        raw_pcd_downsampled->push_back((*raw_pcd_transformed)[m]);
      std::cout<<"downsample to "<<raw_pcd_downsampled->size()<<"points\n";
      /*
      cvo::VoxelMap<pcl::PointXYZ> voxel_map(cvo_align.get_params().multiframe_downsample_voxel_size);
      for (int l = 0; l < raw_pcd_transformed->size(); l++) {
        voxel_map.insert_point(&raw_pcd_transformed->at(l));
      }
      std::vector<pcl::PointXYZ*> sampled =  voxel_map.sample_points();
      for (auto p : sampled)
        raw_pcd_downsampled->push_back(*p);
      */
      std::cout<<"after downsampling, num of points is "<<raw_pcd_downsampled->size()<<std::endl;

      // std::shared_ptr<cvo::CvoPointCloud> pc (new cvo::CvoPointCloud(raw_pcd_downsampled));


      std::shared_ptr<cvo::CvoPointCloud> pc (new cvo::CvoPointCloud(*raw_pcd_downsampled));
      pc->add_semantics(33);
      for (int k = 0 ; k < pc->size(); k++)  {
        memcpy((*pc)[k].label_distribution, (*fpfh)[ind_downsampled[k]].histogram, sizeof(float)*33  );
        //memcpy((*pc)[k].label_distribution, (*fpfh)[k].histogram, sizeof(float)*33  );
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>> label_map((*pc)[k].label_distribution, 33);
        label_map.normalize();
        /*
        if (k == 0) {
        for (int l = 0; l < 33; l++)
          std::cout<<(*fpfh)[k].histogram[l]<<",";
        std::cout<<"\n";
        }*/
      }
      
      
      //std::shared_ptr<cvo::CvoPointCloud> pc(new cvo::CvoPointCloud(0,0));
    
      // cvo::CvoPointCloud::transform(tracking_poses[i],
      //                               *raw,
      //                               *pc);
    
      cvo::Mat34d_row pose;
      Eigen::Matrix<double, 4,4, Eigen::RowMajor> pose44 = Eigen::Matrix<double, 4,4, Eigen::RowMajor>::Identity();
      pose = pose44.block<3,4>(0,0);
    

      cvo::CvoFrame::Ptr new_frame(new cvo::CvoFrameGPU(pc.get(), pose.data()));
      frames.push_back(new_frame);
      pcs.push_back(pc);
    }

    //std::vector<cvo::Mat34d_row, Eigen::aligned_allocator<cvo::Mat34d_row>> tracking_poses;
    std::cout<<"write to before_BA.pcd\n";
    std::string f_name(exp_curr_dir_str + "/before_BA_bunny_");
    f_name += std::to_string(k)+".pcd";
    write_transformed_pc(frames, f_name, colors);

    std::cout<<"write to gt\n";    
    for (int i = 0; i < num_frames; i++ ) {
      cvo::Mat34d_row pose;
      Eigen::Matrix<double, 4,4, Eigen::RowMajor> pose44 = tracking_poses[i].cast<double>().inverse();
      memcpy(frames[i]->pose_vec, pose44.data(), sizeof(double) * 12);
    }

    f_name =  exp_curr_dir_str + "/gt_BA_bunny_"+std::to_string(k)+".pcd" ;
    write_transformed_pc(frames, f_name, colors);

    for (int i = 0; i < num_frames; i++ ) {
      cvo::Mat34d_row pose;
      Eigen::Matrix<double, 4,4, Eigen::RowMajor> pose44 = Eigen::Matrix<double, 4,4, Eigen::RowMajor>::Identity();
      memcpy(frames[i]->pose_vec, pose44.data(), sizeof(double) * 12);
    }
    

    // std::list<std::pair<std::shared_ptr<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr>> edges;
    std::cout<<"Start constructing cvo edges\n";
    std::list<cvo::BinaryState::Ptr> edge_states;            
    for (int i = 0; i < num_frames-1; i++) {
      for (int j = i+1; j < num_frames; j++ ) {
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


        break;
      
      }
    }
    const cvo::CvoParams & params = cvo_align.get_params();
    cvo::BinaryStateGPU::Ptr edge_state(new cvo::BinaryStateGPU(std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[num_frames-1]),
                                                                std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[0]),
                                                                &params,
                                                                cvo_align.get_params_gpu(),
                                                                params.multiframe_num_neighbors,
                                                                params.multiframe_ell_init
                                                                //dist / 3
                                                                ));
    edge_states.push_back((edge_state));
    

    std::vector<bool> const_flags(frames.size(), false);
    const_flags[0] = true;

    std::cout<<"start align\n";
    cvo_align.align(frames, const_flags,
                    edge_states,  nullptr);

    //std::cout<<"Align ends. Total time is "<<time<<std::endl;
    f_name=exp_curr_dir_str + "/after_BA_bunny_" + std::to_string(k)+".pcd";
    write_transformed_pc(frames, f_name, colors);
    
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


    eval_poses(estimates,
               poses_gt,
               fname
               );


  }
  return 0;
}
