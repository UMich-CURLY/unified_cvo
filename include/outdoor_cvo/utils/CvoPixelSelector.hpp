/**
 * This file is part of DSO. used by CVO SLAM. 
 * 
 * Copyright 2016 Technical University of Munich and Intel.
 * Developed by Jakob Engel <engelj at in dot tum dot de>,
 * for more information see <http://vision.in.tum.de/dso>.
 * If you use this code, please cite the respective publications as
 * listed on the above website.
 *
 * DSO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DSO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DSO. If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "utils/data_type.hpp"
#include "utils/RawImage.hpp"

#include <pcl/features/normal_3d_omp.h>
#include <pcl/search/impl/search.hpp>

namespace cvo
{
  void select_pixels(const RawImage & raw_image,
                     int num_want,
                     // output
                     std::vector<Vec2i, Eigen::aligned_allocator<Vec2i>> & output_uv );

  pcl::PointCloud<pcl::Normal>::Ptr
  compute_pcd_normals(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in, float radius);
  
  void edge_detection(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in,
                      int num_want,
                      double intensity_bound, 
                      double depth_bound,
                      double distance_bound,
                      int num_beams,
                      // output
                      pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out,
                      std::vector <double> & output_depth_grad,
                      std::vector <double> & output_intenstity_grad);
  void edge_detection(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in,
                     int num_want,
                     double intensity_bound, 
                     double depth_bound,
                     double distance_bound,
                      int num_beams,
                     // output
                     pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out,
                     std::vector <double> & output_depth_grad,
                     std::vector <double> & output_intenstity_grad,
                     pcl::PointCloud<pcl::Normal>::Ptr normals_out);
  void edge_detection(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in,
                     const std::vector<int> & semantic_in,
                     int num_want,
                     double intensity_bound, 
                     double depth_bound,
                     double distance_bound,
                      int num_beams,
                     // output
                     pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out,
                     std::vector <double> & output_depth_grad,
                     std::vector <double> & output_intenstity_grad,
                     std::vector<int> & semantic_out); 
  void edge_detection(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in,
                     const std::vector<int> & semantic_in,
                     int num_want,
                     double intensity_bound, 
                     double depth_bound,
                     double distance_bound,
                      int num_beams,
                     // output
                     pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out,
                     std::vector <double> & output_depth_grad,
                     std::vector <double> & output_intenstity_grad,
                     pcl::PointCloud<pcl::Normal>::Ptr normals_out,
                     std::vector<int> & semantic_out); 
  void laserCloudHandler(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in,
                        int num_want,
                        double intensity_bound, 
                        double depth_bound,
                        // output
                        pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out,
                        std::vector <double> & output_depth_grad,
                        std::vector <double> & output_intenstity_grad);
  int get_quadrant(pcl::PointXYZI point);


  class PixelSelector
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    enum PixelSelectorStatus {PIXSEL_VOID=0, PIXSEL_1, PIXSEL_2, PIXSEL_3};

    int makeHeatMaps(const RawImage & raw_image,
                     float density, 
                     // output
                     float* map_out,
                     std::vector<Vec2i, Eigen::aligned_allocator<Vec2i>> & output_uv,
                     // default inputs
                     int customized_potential=-1,
                     int recursionsLeft=1, bool plot=false, float thFactor=1
                     );

    PixelSelector(int w, int h);
    ~PixelSelector();
    int currentPotential; // ????

    bool allowFastCornerSelector; //
    void makeHists(const RawImage & raw_image );
    
  private:

    Eigen::Vector3i select(const RawImage & raw_image,
                           int pot, float thFactor,
                           // outputs
                           float* map_out,
                           std::vector<Vec2i, Eigen::aligned_allocator<Vec2i>> & output_uv
                           );

    std::vector<unsigned char> randomPattern;

    int w, h;
    std::vector<int> gradHist;
    std::vector<float> ths;
    std::vector<float> thsSmoothed;
    int thsStep;
    const cvo::RawImage * gradHistFrame;

  };
}
