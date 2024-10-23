#pragma once

#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <unordered_map>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>

using Mat34d_row = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;

namespace cvo {

  template <typename PointT>
class Viewer {
 public:
  
  Viewer(bool screenshotIn, std::string saveDir, bool isFollowingCamera=false) : stopped(false), visualizationTread(&Viewer::runVisualizer, this), screenshot(screenshotIn), screenshotSaveDir(saveDir), isFollowingCamera(isFollowingCamera) {};

  Viewer() : Viewer(false, "") {}; 

  ~Viewer() { visualizationTread.join(); };

    //void addPointCloudSingleColor(const typename pcl::PointCloud<PointT>& Cloud,
    //                            int r, int g, int b, const std::string& id);

  void updatePointCloud(const pcl::PointCloud<PointT>& Cloud,
                        const std::string& id);

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
  std::unordered_map<std::string,
                     typename pcl::PointCloud<PointT>::Ptr> cloudsToAdd;
  std::unordered_map<std::string,
                     typename pcl::PointCloud<PointT>::Ptr> addedCloud;
  std::unordered_map<std::string,
                     typename pcl::PointCloud<PointT>::Ptr> cloudsToUpdate;

  std::map<std::string, std::tuple<int, int, std::string>> textsAll;

  bool screenshot;
  std::string screenshotSaveDir;

  std::mutex trajGuard;  
  Eigen::Matrix3f viewerCamInstrinsics;
  std::vector<Mat34d_row, Eigen::aligned_allocator<Mat34d_row>> trajectoryPtsToDraw;
  int trajId;
  std::unordered_set<int> addedLineIds;
  bool isFollowingCamera;
};

}  // namespace perl_registration
