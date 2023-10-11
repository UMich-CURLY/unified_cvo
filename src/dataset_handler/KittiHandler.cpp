#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cassert>
#include <sstream>
#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include "dataset_handler/KittiHandler.hpp"
#include "utils/debug_visualization.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <map>
using namespace std;
using namespace boost::filesystem;

namespace cvo
{
  KittiHandler::KittiHandler(std::string kitti_folder, DataType data_type,
                             LidarCamCalibType calib_type)
  {
    curr_index = 0;
    folder_name = kitti_folder;
    debug_plot = true;
    string data_folder;
    if (data_type == DataType::STEREO)
    {
      data_folder = folder_name + "/image_2/";
    }
    else if (data_type == DataType::LIDAR)
    {
      data_folder = folder_name + "/velodyne/";
      if (calib_type == LidarCamCalibType::CAM2 &&
          boost::filesystem::exists( folder_name + "/calib_velo_to_cam.txt" )) {
        this->read_lidar_calib(folder_name + "/calib_velo_to_cam.txt", calib_type);
      } else if (calib_type == LidarCamCalibType::CAM0 &&
                 boost::filesystem::exists( folder_name + "/calib.txt" )) {
        this->read_lidar_calib(folder_name + "/calib_velo_to_cam.txt", calib_type);
      } else {
        Eigen::Affine3f rx = Eigen::Affine3f::Identity();
        Eigen::Affine3f ry = Eigen::Affine3f(Eigen::AngleAxisf(-M_PI / 2.0, Eigen::Vector3f::UnitY()));
        Eigen::Affine3f rz = Eigen::Affine3f(Eigen::AngleAxisf(M_PI / 2.0, Eigen::Vector3f::UnitZ()));
        Eigen::Affine3f tf_change_basis = rz * ry * rx;
        Eigen::Matrix<float, 3,4> lidar_to_cam = Eigen::Matrix<float, 3,4>::Zero();
        lidar_to_cam.block<3,3>(0,0) = tf_change_basis.matrix().block<3,3>(0,0);
        this->set_lidar_calib(lidar_to_cam);
      }
    }

    path kitti(data_folder.c_str());
    for (auto &p : directory_iterator(kitti))
    {
      if (is_regular_file(p.path()))
      {
        string curr_file = p.path().filename().string();
        size_t last_ind = curr_file.find_last_of(".");
        string raw_name = curr_file.substr(0, last_ind);
        names.push_back(raw_name);
      }
    }
    sort(names.begin(), names.end());
    cout << "Kitti contains " << names.size() << " files\n";

  }

  KittiHandler::~KittiHandler() {}

  int KittiHandler::read_next_stereo(cv::Mat &left,
                                     cv::Mat &right)
  {
    if (curr_index >= names.size())
      return -1;

    string left_name = folder_name + "/image_2/" + names[curr_index] + ".png";
    string right_name = folder_name + "/image_3/" + names[curr_index] + ".png";
    left = cv::imread(left_name, cv::ImreadModes::IMREAD_COLOR);
    right = cv::imread(right_name, cv::ImreadModes::IMREAD_COLOR);

    if (left.data == nullptr || right.data == nullptr)
    {
      cerr << "Image doesn't read successfully: " << left_name << ", " << right_name << "\n";
      return -1;
    }
    return 0;
  }

  int KittiHandler::read_next_stereo(cv::Mat &left, cv::Mat &right,
                                     int num_semantic_class,
                                     vector<float> &semantics)
  {

    if (read_next_stereo(left, right))
      return -1;

    string semantic_name = folder_name + "/image_semantic/" + names[curr_index] + ".bin";
    int num_bytes = sizeof(float) * num_semantic_class * left.total();
    semantics.resize(left.total() * num_semantic_class);
    infile.open(semantic_name.c_str(), std::ios::in | std::ifstream::binary);
    if (infile.is_open())
    {
      infile.read(reinterpret_cast<char *>(semantics.data()), num_bytes);
      infile.close();
    }
    else
    {
      cerr << "Semantic Image doesn't read successfully: " << semantic_name << "\n"
           << std::flush;
      return -1;
    }

    if (debug_plot)
      visualize_semantic_image("last_semantic.png", semantics.data(), num_semantic_class, left.cols, left.rows);

    return 0;
  }

  int KittiHandler::read_next_lidar_mono(cv::Mat &image,
                                         pcl::PointCloud<pcl::PointXYZ>::Ptr pc)
  {
    std::cerr << "lidar-mono not implemented in KittiHandler\n";
    assert(0);
    return -1;
  }

  int KittiHandler::read_next_lidar(pcl::PointCloud<pcl::PointXYZI>::Ptr pc)
  {

    if (curr_index >= names.size())
      return -1;

    string lidar_bin_path = folder_name + "/velodyne/" + names[curr_index] + ".bin";

    std::ifstream fLidar(lidar_bin_path.c_str(), std::ios::in | std::ios::binary);

    fLidar.seekg(0, fLidar.end);
    int fLidar_length = fLidar.tellg();
    fLidar.seekg(0, fLidar.beg);
    int num_lidar_points = fLidar_length / (4 * sizeof(float));

    std::vector<float> lidar_points;

    if (fLidar.is_open())
    {
      int num_bytes = sizeof(float) * 4 * num_lidar_points;

      lidar_points.resize(4 * num_lidar_points);
      infile.open(lidar_bin_path.c_str(), std::ios::in | std::ifstream::binary);

      infile.read(reinterpret_cast<char *>(lidar_points.data()), num_bytes);
      infile.close();

      // Eigen::Affine3f tf_change_basis = Eigen::Affine3f::Identity();
      for (int r = 0; r < num_lidar_points; ++r)
      {
        pcl::PointCloud<pcl::PointXYZI>::PointType temp_pcl;
        pcl::PointCloud<pcl::PointXYZI>::PointType pcl_after_tf;
        temp_pcl.x = lidar_points[r * 4 + 0];
        temp_pcl.y = lidar_points[r * 4 + 1];
        temp_pcl.z = lidar_points[r * 4 + 2];

        Eigen::Vector3f temp_pt(temp_pcl.x, temp_pcl.y, temp_pcl.z);
        Eigen::Vector3f after_tf_pt = this->T_velo_to_cam.block<3,3>(0,0) * temp_pt
          + this->T_velo_to_cam.block<3,1>(0,3);
        pcl_after_tf.x = after_tf_pt(0);
        pcl_after_tf.y = after_tf_pt(1);
        pcl_after_tf.z = after_tf_pt(2);

        pcl_after_tf.intensity = lidar_points[r * 4 + 3];
        pc->push_back(pcl_after_tf);
      }
    }
    else
    {
      cerr << "Lidar doesn't read successfully: " << lidar_bin_path << "\n"
           << std::flush;
      return -1;
    }
    return 0;
  }

  int KittiHandler::read_next_lidar(pcl::PointCloud<pcl::PointXYZI>::Ptr pc,
                                    vector<int> &semantics_lidar)
  {

    if (read_next_lidar(pc))
      return -1;

    string semantic_bin_path = folder_name + "/labels/" + names[curr_index] + ".label";
    int num_semantic_class = 19;
    std::ifstream fLidar_sem(semantic_bin_path.c_str(), std::ios::in | std::ios::binary);

    if (fLidar_sem.is_open())
    {
      fLidar_sem.seekg(0, fLidar_sem.end);
      int fLidar_sem_length = fLidar_sem.tellg();
      fLidar_sem.seekg(0, fLidar_sem.beg);
      int num_lidar_points = fLidar_sem_length / sizeof(uint32_t);

      int num_bytes = sizeof(uint32_t) * num_lidar_points;
      std::vector<uint32_t> temp_semantics;
      temp_semantics.resize(num_lidar_points);
      infile.open(semantic_bin_path.c_str(), std::ios::in | std::ifstream::binary);
      infile.read(reinterpret_cast<char *>(temp_semantics.data()), num_bytes);
      infile.close();

      std::map<int, int> label_map = create_label_map();

      for (int i = 0; i < num_lidar_points; i++)
      {
        pcl::PointXYZRGB temp_pt_sem;
        uint16_t semantic_label = (uint16_t)(temp_semantics.data()[i] & 0x0000FFFFuL);
        uint16_t instance_label = (uint16_t)(temp_semantics.data()[i] >> 16);

        semantics_lidar.push_back(label_map[semantic_label] - 1);
      }

      return 0;
    }
    else
    {
      cerr << "Semantic Image doesn't read successfully: " << semantic_bin_path << "\n"
           << std::flush;
      return -1;
    }
  }

  std::map<int, int> KittiHandler::create_label_map()
  {
    std::map<int, int> label_map;
    label_map[0] = 0;
    label_map[1] = 0;
    label_map[10] = 1;
    label_map[11] = 2;
    label_map[13] = 5;
    label_map[15] = 3;
    label_map[16] = 5;
    label_map[18] = 4;
    label_map[20] = 5;
    label_map[30] = 6;
    label_map[31] = 7;
    label_map[32] = 8;
    label_map[40] = 9;
    label_map[44] = 10;
    label_map[48] = 11;
    label_map[49] = 12;
    label_map[50] = 13;
    label_map[51] = 14;
    label_map[52] = 0;
    label_map[60] = 9;
    label_map[70] = 15;
    label_map[71] = 16;
    label_map[72] = 17;
    label_map[80] = 18;
    label_map[81] = 19;
    label_map[99] = 0;
    label_map[252] = 1;
    label_map[253] = 7;
    label_map[254] = 6;
    label_map[255] = 8;
    label_map[256] = 5;
    label_map[257] = 5;
    label_map[258] = 4;
    label_map[259] = 5;
    return label_map;
  }

  void KittiHandler::next_frame_index()
  {
    curr_index++;
  }

  void KittiHandler::set_start_index(int start)
  {
    curr_index = start;
  }

  int KittiHandler::get_current_index()
  {
    return curr_index;
  }

  int KittiHandler::get_total_number()
  {
    return names.size();
  }

  void KittiHandler::set_lidar_calib(const Eigen::Matrix<float, 3, 4> & lidar_to_cam) {
    this->T_velo_to_cam = lidar_to_cam;
  }

  void KittiHandler::read_lidar_calib(const std::string & calib_file,
                                      LidarCamCalibType calib_type) {

    std::ifstream infile(calib_file);

    int line_counter = 0;

    if (calib_type == LidarCamCalibType::CAM2) {
      std::string line;
      std::getline(infile, line);

      /// R
      std::getline(infile, line);
      std::stringstream ss(line);
      std::string first_str;
      ss >> first_str;
      //float r[9];
      Eigen::Matrix3f R_col;
      for (int i = 0; i < 9; i++)
        ss >> R_col(i/3, i%3);
      //Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R_row = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(r);
      //Eigen::Matrix3f R_col = R_row;

      /// t
      std::getline(infile, line);
      std::stringstream sl(line);
      sl >> first_str;
      Eigen::Vector3f t;
      for (int i = 0; i < 3; i++)
        sl >> t(i);

      T_velo_to_cam.block<3,3>(0,0) = R_col;
      T_velo_to_cam.block<3,1>(0,3) = t;
      std::cout<<"Read T_velo_to_cam to be\n"<<T_velo_to_cam<<"\n";
    } else if (calib_type == LidarCamCalibType::CAM0){
      std::string line;

      // P0, P1, P2, P3
      for (int i = 0; i <4; i++)
        std::getline(infile, line);

      /// Tr:
      std::getline(infile, line);
      std::stringstream ss(line);
      std::string first_str;
      ss >> first_str;
      
      //float r[9];
      for (int i = 0; i < 12; i++) {
        ss >> this->T_velo_to_cam(i/4, i%4);
      }
      //Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R_row = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(r);
      //Eigen::Matrix3f R_col = R_row;

      /// t
      std::cout<<"Read T_velo_to_cam to be\n"<<T_velo_to_cam<<"\n";
      
    }
  }

}
