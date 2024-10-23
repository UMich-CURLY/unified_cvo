
#include <chrono>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <viewer/color_handler.h>
#include "utils/viewer.hpp" //<viewer/viewer.h>

#include <iostream>

namespace cvo {

  //template <>
  void addPointCloudToViewer(pcl::visualization::PCLVisualizer & viewer,
                                               pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloudPtr,
                                               std::string & id){
    viewer.addPointCloud<pcl::PointXYZRGB>(cloudPtr, id);
  }
  
  //template <>
  void addPointCloudToViewer(pcl::visualization::PCLVisualizer & viewer,
                             pcl::PointCloud<pcl::PointXYZI>::ConstPtr cloudPtr,
                             std::string & id){
    perl_registration::PointCloudColorHandlerIntensityMap colormap(cloudPtr, 0, 255);    
    viewer.addPointCloud<pcl::PointXYZI>(cloudPtr, colormap, id);
  }
  

  template <typename PointT>
  void Viewer<PointT>::updatePointCloud(const pcl::PointCloud<PointT> & cloud,
                                        const std::string &id) {
    typename pcl::PointCloud<PointT>::Ptr cloudPtr(
                                                   new pcl::PointCloud<PointT>(cloud));
    std::lock_guard<std::mutex> lockClouds(cloudsGuard);
    if (addedCloud.find(id) != addedCloud.end()) {
      cloudsToUpdate[id] = cloudPtr;
    } else
      cloudsToAdd[id] = cloudPtr;    
  }

  template <typename PointT>  
  void Viewer<PointT>::addOrUpdateText(const std::string & text, int x, int y, const std::string &id ) {
    std::string text_to_add = text;
    textsAll[id] = std::make_tuple(x, y, text_to_add);
  }

  template <typename PointT>  
  void Viewer<PointT>::runVisualizer() {
    pcl::visualization::PCLVisualizer viewer("Viewer");

    trajId = 0;

    viewerCamInstrinsics << 640, 0, 320, 0, 480, 240, 0, 0, 1;

    viewer.setBackgroundColor(255, 255, 255);
    viewer.setCameraPosition(0, 15, 0, 0, -1, 0, 1, 0, 0);
    viewer.addCoordinateSystem(0.25);
    viewer.initCameraParameters();
    //viewer.setBackgroundColor (0, 0, 0);

    while (!viewer.wasStopped()) {
      viewer.spinOnce(100);
      //std::this_thread::sleep_for(std::chrono::microseconds(100000));

      std::lock_guard<std::mutex> lockClouds(cloudsGuard);

      /// add new pc
      for (auto & pp : cloudsToAdd) {
        auto id = pp.first;
        auto cloudPtr = pp.second;
        addPointCloudToViewer(viewer, cloudPtr, id);
        viewer.setPointCloudRenderingProperties(
                                                pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, id);
        addedCloud.insert(std::make_pair(id, cloudPtr));
      }
      cloudsToAdd.clear();

      /// update existing pc
      for (auto & p : cloudsToUpdate) {
        auto cloudPtr = p.second;
        auto id  = p.first;
        viewer.removePointCloud( id);
        addPointCloudToViewer(viewer, cloudPtr, id);
      }
      cloudsToUpdate.clear();

      //// text
      for (auto & p : textsAll) {
        std::string id = p.first;
        int x = std::get<0>(p.second);
        int y = std::get<1>(p.second);
        std::string text = std::get<2>(p.second);
        if (viewer.updateText(text, x, y, id) == false) {
          viewer.addText(text, x, y, 30, 0,0,0, id);
        }
      }
      
      //// pose trajectory
      {
        std::lock_guard<std::mutex> lockStopped(trajGuard);            
        if (trajectoryPtsToDraw.empty() == false && trajectoryPtsToDraw.size() > 1) {
        
          for (int i = 1; i < trajectoryPtsToDraw.size(); i++) {
            pcl::PointXYZ prevPt(trajectoryPtsToDraw[i - 1](0, 3), trajectoryPtsToDraw[i - 1](1, 3), trajectoryPtsToDraw[i - 1](2, 3));
            pcl::PointXYZ currPt(trajectoryPtsToDraw[i](0, 3), trajectoryPtsToDraw[i](1, 3), trajectoryPtsToDraw[i](2, 3));
            std::cout<<__func__<<": draw pose last is "<<prevPt.getVector3fMap().transpose()<<", curr is "<<currPt.getVector3fMap().transpose()<<"\n";
            viewer.addLine(prevPt, currPt, 1, 0, 0, std::to_string(trajId) + "_line");
            addedLineIds.insert(trajId);

          }
          Mat34d_row prevPose = trajectoryPtsToDraw.back();
          trajectoryPtsToDraw.clear();
          trajectoryPtsToDraw.push_back(prevPose);
          //std::cout<<__func__<<": push back "<<prevPose<<"\n";
      
          // update camera to follow the trajectory
          if (isFollowingCamera) {
            Eigen::Matrix4f Twc = Eigen::Matrix4f::Identity();
            Twc.block<3, 4>(0, 0) = prevPose.cast<float>();
            Eigen::Matrix4f Tcv;
            Tcv << -1,  0, 0, -1,
              0, -1, 0, -1,
              0,  0, 1, -5,
              0,  0, 0, 1;
            Eigen::Matrix4f Twv = Twc * Tcv;
            viewer.setCameraParameters(viewerCamInstrinsics, Twv);
          }
        }
      }
      trajId++;
      if (screenshot) {
        std::string filename = screenshotSaveDir + "/" + std::to_string(trajId) + ".png";
        viewer.saveScreenshot(filename);

      }
      
    }
    

    std::lock_guard<std::mutex> lockStopped(stoppedGuard);

    stopped = true;
  };

  template <typename PointT>  
  bool Viewer<PointT>::wasStopped() {
    std::lock_guard<std::mutex> lockStopped(stoppedGuard);
    bool temp = stopped;
    return temp;
  };

  template <typename PointT>
  void Viewer<PointT>::drawTrajectory(const Mat34d_row& current_pose) {
    std::lock_guard<std::mutex> lockStopped(trajGuard);
    
    trajectoryPtsToDraw.push_back(current_pose);
  }

  template class cvo::Viewer<pcl::PointXYZI>;
  template class cvo::Viewer<pcl::PointXYZRGB>;

}  // namespace perl_registration
