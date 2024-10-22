
#include <chrono>

#include <pcl/visualization/pcl_visualizer.h>

#include <viewer/color_handler.h>
#include <viewer/viewer.h>

#include <iostream>

namespace perl_registration {

  void Viewer::addPointCloudSingleColor(
                                        const typename pcl::PointCloud<PointT>& Cloud, int r, int g, int b,
                                        const std::string& id) {
    typename pcl::PointCloud<PointT>::Ptr cloudPtr(
                                                   new pcl::PointCloud<PointT>(Cloud));

    std::lock_guard<std::mutex> lockClouds(cloudsGuard);
    singleIdsToAdd.push_back(id);
    singleCloudsToAdd[id] = std::make_tuple(cloudPtr, r, g, b);
  };


  /*
  void Viewer::addColorPointCloud(const pcl::PointCloud<pcl::PointXYZRGB> & cloud,
                             const std::string &id
                             ) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudPtr(
                                                    new pcl::PointCloud<pcl::PointXYZRGB>(cloud));
    std::lock_guard<std::mutex> lockClouds(cloudsGuard);
    colorIdsToAdd.push_back(id);
    colorCloudsToAdd[id] = cloudPtr;    
    }*/

  void Viewer::updateColorPointCloud(const pcl::PointCloud<pcl::PointXYZRGB> & cloud,
                                     const std::string &id) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudPtr(
                                                    new pcl::PointCloud<pcl::PointXYZRGB>(cloud));
    std::lock_guard<std::mutex> lockClouds(cloudsGuard);

    if (addedColorCloud.find(id) != addedColorCloud.end()) {

      colorCloudsToUpdate[id] = cloudPtr;
    } else
      colorCloudsToAdd[id] = cloudPtr;    
  }

  void Viewer::updateColorPointCloud(const pcl::PointCloud<pcl::PointXYZI> & cloud,
                                     const std::string &id) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr(
                                                  new pcl::PointCloud<pcl::PointXYI>(cloud));
    std::lock_guard<std::mutex> lockClouds(cloudsGuard);

    if (addedColorCloud.find(id) != addedColorCloud.end()) {

      colorCloudsToUpdate[id] = cloudPtr;
    } else
      colorCloudsToAdd[id] = cloudPtr;    
  }
  
  
  
    
  void Viewer::addPointCloudIntensity(
                                      const pcl::PointCloud<pcl::PointXYZI>& cloud, float min, float max,
                                      const std::string& id) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr(
                                                  new pcl::PointCloud<pcl::PointXYZI>(cloud));

    std::lock_guard<std::mutex> lockClouds(cloudsGuard);
    intensityIdsToAdd.push_back(id);
    intensityCloudsToAdd[id] = std::make_tuple(cloudPtr, min, max);
  };

  void Viewer::addOrUpdateText(const std::string & text, int x, int y, const std::string &id ) {
    std::string text_to_add = text;
    textsAll[id] = std::make_tuple(x, y, text_to_add);

  }

  void Viewer::runVisualizer() {
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
      for (auto id : singleIdsToAdd) {
        auto pair = singleCloudsToAdd[id];
        pcl::visualization::PointCloudColorHandlerCustom<PointT> single_color(
                                                                              std::get<0>(pair), std::get<1>(pair), std::get<2>(pair),
                                                                              std::get<3>(pair));
        viewer.addPointCloud<PointT>(std::get<0>(pair), single_color, id);
        viewer.setPointCloudRenderingProperties(
                                                pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, id);
        //singleCloudsToAdd.erase(id);
      }
      singleIdsToAdd.clear();


      //// intensity
      for (auto id : intensityIdsToAdd) {
        auto tuple = intensityCloudsToAdd[id];
        PointCloudColorHandlerIntensityMap intensity_colormap(
                                                              std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple));
        viewer.addPointCloud<pcl::PointXYZI>(std::get<0>(tuple),
                                             intensity_colormap, id);
        viewer.setPointCloudRenderingProperties(
                                                pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, id);
        //intensityCloudsToAdd.erase(id);
      }
      intensityIdsToAdd.clear();


      /// color pc
      for (auto & pp : colorCloudsToAdd) {
        auto id = pp.first;
        auto cloudPtr = pp.second;
        // pcl::visualization::PointCloudColorHandler<pcl::PointXYZRGB> colorHandler()
        viewer.addPointCloud<pcl::PointXYZRGB>(cloudPtr, id);
        viewer.setPointCloudRenderingProperties(
                                                pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, id);
        addedColorCloud.insert(id);
      }
      colorCloudsToAdd.clear();
      for (auto & p : colorCloudsToUpdate) {
        auto cloudPtr = p.second;
        auto id  = p.first;
        std::cout<<__func__<<"before transform: "<<static_cast<int>((*cloudPtr)[0].r)<<", "<<static_cast<int>((*cloudPtr)[0].g)<<", "<<static_cast<int>((*cloudPtr)[0].b)<<"\n";
        viewer.removePointCloud( id);
        viewer.addPointCloud<pcl::PointXYZRGB>(cloudPtr, id);
      }
      colorCloudsToUpdate.clear();

      for (auto & p : textsAll) {
        std::string id = p.first;
        int x = std::get<0>(p.second);
        int y = std::get<1>(p.second);
        std::string text = std::get<2>(p.second);
        if (viewer.updateText(text, x, y, id) == false) {
          viewer.addText(text, x, y, 30, 0,0,0, id);
        }
      }
      
      //colorCloudsToUpdate.clear();
      //colorIdsToAdd.clear();
      
      {
        std::lock_guard<std::mutex> lockStopped(trajGuard);            
        if (trajectoryPtsToDraw.empty() || trajectoryPtsToDraw.size() == 1) continue;
        
        for (int i = 1; i < trajectoryPtsToDraw.size(); i++) {
          pcl::PointXYZ prevPt(trajectoryPtsToDraw[i - 1](0, 3), trajectoryPtsToDraw[i - 1](1, 3), trajectoryPtsToDraw[i - 1](2, 3));
          pcl::PointXYZ currPt(trajectoryPtsToDraw[i](0, 3), trajectoryPtsToDraw[i](1, 3), trajectoryPtsToDraw[i](2, 3));
          std::cout<<__func__<<": draw pose last is "<<prevPt.getVector3fMap().transpose()<<", curr is "<<currPt.getVector3fMap().transpose()<<"\n";
          viewer.addLine(prevPt, currPt, 1, 0, 0, std::to_string(trajId) + "_line");
          addedLineIds.insert(trajId);
          trajId++;
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
      if (screenshot) {
        std::string filename = screenshotSaveDir + "/" + std::to_string(trajId) + ".png";
        viewer.saveScreenshot(filename);

      }
      
    }

    std::lock_guard<std::mutex> lockStopped(stoppedGuard);

    stopped = true;
  };

  bool Viewer::wasStopped() {
    std::lock_guard<std::mutex> lockStopped(stoppedGuard);
    bool temp = stopped;
    return temp;
  };

  void Viewer::drawTrajectory(const Mat34d_row& current_pose) {
    std::lock_guard<std::mutex> lockStopped(trajGuard);
    
    trajectoryPtsToDraw.push_back(current_pose);
  }

}  // namespace perl_registration
