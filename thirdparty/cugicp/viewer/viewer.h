#pragma once

#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>

using Mat34d_row = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;

namespace perl_registration {

class Viewer {
 public:
  typedef pcl::PointXYZ PointT;
  
  Viewer(bool screenshotIn, std::string saveDir) : stopped(false), visualizationTread(&Viewer::runVisualizer, this), screenshot(screenshotIn), screenshotSaveDir(saveDir) {};

  Viewer() : Viewer(false, "") {}; 

  ~Viewer() { visualizationTread.join(); };

  void addPointCloudSingleColor(const typename pcl::PointCloud<PointT>& Cloud,
                                int r, int g, int b, const std::string& id);

  void addPointCloudIntensity(const pcl::PointCloud<pcl::PointXYZI>& Cloud,
                              float min, float max, const std::string& id);

  void addColorPointCloud(const pcl::PointCloud<pcl::PointXYZRGB> & cloud,
                          const std::string &id);

  void updateColorPointCloud(const pcl::PointCloud<pcl::PointXYZRGB> & cloud,
                             const std::string &id);
  void addOrUpdateText(const std::string & text, int x, int y, const std::string &id );

  bool wasStopped();
     
  void drawTrajectory(const Mat34d_row& current_pose);

 private:
  void runVisualizer();

  bool stopped;
  std::thread visualizationTread;
  std::mutex stoppedGuard;
  std::mutex cloudsGuard;
  std::vector<std::string> idsCurrent;
  std::vector<std::string> singleIdsToAdd;
  std::vector<std::string> colorIdsToAdd;
  std::map<std::string,
           std::tuple<typename pcl::PointCloud<PointT>::Ptr, int, int, int>>
      singleCloudsToAdd;
  std::map<std::string,
           pcl::PointCloud<pcl::PointXYZRGB>::Ptr
           > colorCloudsToAdd;
  std::map<std::string,
           pcl::PointCloud<pcl::PointXYZRGB>::Ptr
           > colorCloudsToUpdate;

  std::map<std::string, std::tuple<int, int, std::string>> textsAll;

  std::vector<std::string> intensityIdsToAdd;
  std::map<std::string,
           std::tuple<pcl::PointCloud<pcl::PointXYZI>::Ptr, float, float>>
      intensityCloudsToAdd;
  bool screenshot;
  std::string screenshotSaveDir;

  Eigen::Matrix3f viewerCamInstrinsics;
  std::vector<Mat34d_row, Eigen::aligned_allocator<Mat34d_row>> trajectoryPtsToDraw;
  int trajId;
};

}  // namespace perl_registration
