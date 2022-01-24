
#include <chrono>

#include <pcl/visualization/pcl_visualizer.h>

#include <viewer/color_handler.h>
#include <viewer/viewer.h>

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


  void Viewer::addColorPointCloud(const pcl::PointCloud<pcl::PointXYZRGB> & cloud,
                             const std::string &id
                             ) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudPtr(
                                                    new pcl::PointCloud<pcl::PointXYZRGB>(cloud));
    std::lock_guard<std::mutex> lockClouds(cloudsGuard);
    colorIdsToAdd.push_back(id);
    colorCloudsToAdd[id] = cloudPtr;    
  }

  void Viewer::updateColorPointCloud(const pcl::PointCloud<pcl::PointXYZRGB> & cloud,
                                     const std::string &id) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudPtr(
                                                    new pcl::PointCloud<pcl::PointXYZRGB>(cloud));
    std::lock_guard<std::mutex> lockClouds(cloudsGuard);

    colorCloudsToUpdate[id] = cloudPtr;    
    
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

    viewer.setBackgroundColor(255, 255, 255);
    viewer.setCameraPosition(0, 15, 0, 0, -1, 0, 1, 0, 0);
    viewer.updateCamera();
    viewer.addCoordinateSystem(0.25);
    viewer.initCameraParameters();

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
      for (auto id : intensityIdsToAdd) {
        auto tuple = intensityCloudsToAdd[id];
        PointCloudColorHandlerIntensityMap intensity_colormap(
                                                              std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple));
        viewer.addPointCloud<pcl::PointXYZI>(std::get<0>(tuple),
                                             intensity_colormap, id);
        viewer.setPointCloudRenderingProperties(
                                                pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, id);
        //intensityCloudsToAdd.erase(id);
      }
      intensityIdsToAdd.clear();
      for (auto id : colorIdsToAdd) {
        auto cloudPtr = colorCloudsToAdd[id];
        // pcl::visualization::PointCloudColorHandler<pcl::PointXYZRGB> colorHandler()
        viewer.addPointCloud<pcl::PointXYZRGB>(cloudPtr, id);
        viewer.setPointCloudRenderingProperties(
                                                pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, id);
        
      }
      for (auto & p : colorCloudsToUpdate) {
        auto cloudPtr = p.second;
        auto id  = p.first;
        // pcl::visualization::PointCloudColorHandler<pcl::PointXYZRGB> colorHandler()
        viewer.updatePointCloud<pcl::PointXYZRGB>(cloudPtr, id);
        //viewer.setPointCloudRenderingProperties(
        //                                        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, id);
        
      }

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
      colorIdsToAdd.clear();
      
    }

    std::lock_guard<std::mutex> lockStopped(stoppedGuard);
    stopped = true;
  };

  bool Viewer::wasStopped() {
    std::lock_guard<std::mutex> lockStopped(stoppedGuard);
    bool temp = stopped;
    return temp;
  };

}  // namespace perl_registration
