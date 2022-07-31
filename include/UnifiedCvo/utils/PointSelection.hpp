#pragma once
#include <string>
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include "utils/CvoPoint.hpp"
#include "utils/data_type.hpp"


namespace cvo {
    template <typename DepthType> class ImageRGBD;
    class ImageStereo;
    class RawImage;
    class Calibration;

  enum PointCloudType {
      STEREO,
      RGBD,
      LIDAR
  };

  enum PointSelectionMethod {
      CV_FAST,
      RANDOM,
      DSO_EDGES,
      DSO_EDGES_WITH_RANDOM,
      LIDAR_EDGES,
      CANNY_EDGES,
      EDGES_ONLY,
      LOAM,
      FULL
  };

    // functions to support point selection and output pcl pointcloud
    void pointSelection(const ImageStereo& left_raw_image,
                        const Calibration& calib,
                        pcl::PointCloud<CvoPoint>& out_pc,
                        PointSelectionMethod pt_selection_method=CV_FAST);

    template <typename DepthType>
    void pointSelection(const ImageRGBD<DepthType>& raw_image,
                        const Calibration& calib,
                        pcl::PointCloud<CvoPoint>& out_pc,
                        PointSelectionMethod pt_selection_method=CV_FAST);

    void pointSelection(pcl::PointCloud<pcl::PointXYZI>::Ptr pc,
                        int target_num_points,
                        int beam_num,
                        pcl::PointCloud<CvoPoint>& out_pc,
                        PointSelectionMethod pt_selection_method=LOAM);

    void pointSelection(pcl::PointCloud<pcl::PointXYZI>::Ptr pc,
                        const std::vector<int> & semantic,
                        int num_classes,
                        int target_num_points,
                        int beam_num,
                        pcl::PointCloud<CvoPoint>& out_pc,
                        PointSelectionMethod pt_selection_method=LOAM);
}