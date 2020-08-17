/*
Copyright 2013, Ji Zhang, Carnegie Mellon University
Further contributions copyright (c) 2016, Southwest Research Institute
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

This is an implementation of the algorithm described in the following paper:
  J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
    Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 
*/


#include <iostream>
#include <cstring>
#include <cmath>

#include <vector>
#include <string>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// #include <pcl/features/normal_3d_omp.h>
// #include <pcl/visualization/cloud_viewer.h>
// #include <pcl/search/impl/search.hpp>

#include "utils/LoamScanRegistration.hpp"

namespace cvo
{

  LoamScanRegistration::LoamScanRegistration(const float& lowerBound,
                                              const float& upperBound, 
                                              const uint16_t& nScanRings) :
    _lowerBound(lowerBound),
    _upperBound(upperBound),
    _nScanRings(nScanRings)
  {
    _factor = (nScanRings - 1) / (upperBound - lowerBound);
  }

  LoamScanRegistration::~LoamScanRegistration() {
  }

  // these functions are from loam_velodyne

  int LoamScanRegistration::getRingForAngle(const float& angle) {
    return int(((angle * 180 / M_PI) - _lowerBound) * _factor + 0.5);
  }

  void LoamScanRegistration::process(const pcl::PointCloud<pcl::PointXYZI>& laserCloudIn, 
                                      pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out, 
                                      std::vector <float> & edge_or_surface,
                                      std::vector <int> & selected_indexes)
  {
    reset();
    size_t cloudSize = laserCloudIn.size();

    // determine scan start and end orientations
    float startOri = -std::atan2(laserCloudIn[0].y, laserCloudIn[0].x);
    float endOri = -std::atan2(laserCloudIn[cloudSize - 1].y,
                              laserCloudIn[cloudSize - 1].x) + 2 * float(M_PI);
    if (endOri - startOri > 3 * M_PI) {
      endOri -= 2 * M_PI;
    } else if (endOri - startOri < M_PI) {
      endOri += 2 * M_PI;
    }

    bool halfPassed = false;
    pcl::PointXYZI point;
    _laserCloudScans.resize(_nScanRings);
    // clear all scanline points
    std::for_each(_laserCloudScans.begin(), _laserCloudScans.end(), [](auto&&v) {v.clear(); }); 
    int previous_quadrant = get_quadrant(laserCloudIn.points[0]);
    int quadrant = get_quadrant(laserCloudIn.points[0]);
    int scanID = 0;

    // extract valid points from input cloud
    for (int i = 0; i < cloudSize; i++) {
      point.x = laserCloudIn[i].x;
      point.y = laserCloudIn[i].y;
      point.z = laserCloudIn[i].z;
      point.intensity = i;
      // point.intensity = laserCloudIn[i].intensity;
      // point.x = laserCloudIn[i].y;
      // point.y = laserCloudIn[i].z;
      // point.z = laserCloudIn[i].x;

      // skip NaN and INF valued points
      if (!pcl_isfinite(point.x) ||
          !pcl_isfinite(point.y) ||
          !pcl_isfinite(point.z)) {
        continue;
      }

      // skip zero valued points
      if (point.x * point.x + point.y * point.y + point.z * point.z < 0.0001) {
        continue;
      }

      // calculate vertical point angle and scan ID
      float angle = std::atan(point.y / std::sqrt(point.x * point.x + point.z * point.z));

      // change scanID to be determined by quadrant
      previous_quadrant = quadrant;
      quadrant = get_quadrant(laserCloudIn.points[i]);
      if(quadrant == 1 && previous_quadrant == 4){
        scanID += 1;
      }

      // this scanID is wrong
      // int scanID = getRingForAngle(angle);      
      if (scanID >= _nScanRings || scanID < 0 ){
        continue;
      }

      // // calculate horizontal point angle
      // float ori = -std::atan2(point.x, point.z);
      // if (!halfPassed) {
      //   if (ori < startOri - M_PI / 2) {
      //     ori += 2 * M_PI;
      //   } else if (ori > startOri + M_PI * 3 / 2) {
      //     ori -= 2 * M_PI;
      //   }

      //   if (ori - startOri > M_PI) {
      //     halfPassed = true;
      //   }
      // } else {
      //   ori += 2 * M_PI;

      //   if (ori < endOri - M_PI * 3 / 2) {
      //     ori += 2 * M_PI;
      //   } else if (ori > endOri + M_PI / 2) {
      //     ori -= 2 * M_PI;
      //   }
      // }

      // calculate relative scan time based on point orientation
      // float relTime = _scanPeriod * (ori - startOri) / (endOri - startOri);
      // TODO: this intensity is not correct!!
      // point.intensity = scanID + relTime;

      _laserCloudScans[scanID].push_back(point);
      

    }

    processScanlines(laserCloudIn, _laserCloudScans, pc_out, edge_or_surface, selected_indexes);
  }

  void LoamScanRegistration::processScanlines(const pcl::PointCloud<pcl::PointXYZI>& laserCloudIn,
                                              std::vector<pcl::PointCloud<pcl::PointXYZI>> const& laserCloudScans, 
                                              pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out, 
                                              std::vector <float> & edge_or_surface,
                                              std::vector <int> & selected_indexes)
  {
    // reset internal buffers and set IMU start state based on current scan time
    reset();

    // construct sorted full resolution cloud
    size_t cloudSize = 0;
    for (int i = 0; i < laserCloudScans.size(); i++) {
      _laserCloud += laserCloudScans[i];

      IndexRange range(cloudSize, 0);
      cloudSize += laserCloudScans[i].size();
      range.second = cloudSize > 0 ? cloudSize - 1 : 0;
      _scanIndices.push_back(range);
    }

    extractFeatures(laserCloudIn, pc_out, edge_or_surface, selected_indexes);
  }

  void LoamScanRegistration::reset()
  {
      // clear cloud buffers
      _laserCloud.clear();
      _cornerPointsSharp.clear();
      _cornerPointsLessSharp.clear();
      _surfacePointsFlat.clear();
      _surfacePointsLessFlat.clear();

      // clear scan indices vector
      _scanIndices.clear();
  }


  void LoamScanRegistration::extractFeatures(const pcl::PointCloud<pcl::PointXYZI>& laserCloudIn,
                                              pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out, 
                                              std::vector <float> & edge_or_surface,
                                              std::vector <int> & selected_indexes)
  {
    // extract features from individual scans
    size_t nScans = _scanIndices.size();
    for (size_t i = 0; i < nScans; i++) {
      pcl::PointCloud<pcl::PointXYZI>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<pcl::PointXYZI>);
      size_t scanStartIdx = _scanIndices[i].first;
      size_t scanEndIdx = _scanIndices[i].second;

      // skip empty scans
      if (scanEndIdx <= scanStartIdx + 2 * _curvatureRegion) {
        continue;
      }

      // // Quick&Dirty fix for relative point time calculation without IMU data
      // float scanSize = scanEndIdx - scanStartIdx + 1;
      // for (int j = scanStartIdx; j <= scanEndIdx; j++) {
      //   _laserCloud[j].intensity = i + _scanPeriod * (j - scanStartIdx) / scanSize;
      // }

      // reset scan buffers
      setScanBuffersFor(scanStartIdx, scanEndIdx);

      // extract features from equally sized scan regions
      for (int j = 0; j < _nFeatureRegions; j++) {
        size_t sp = ((scanStartIdx + _curvatureRegion) * (_nFeatureRegions - j)
                    + (scanEndIdx - _curvatureRegion) * j) / _nFeatureRegions;
        size_t ep = ((scanStartIdx + _curvatureRegion) * (_nFeatureRegions - 1 - j)
                    + (scanEndIdx - _curvatureRegion) * (j + 1)) / _nFeatureRegions - 1;

        // skip empty regions
        if (ep <= sp) {
          continue;
        }

        size_t regionSize = ep - sp + 1;

        // reset region buffers
        setRegionBuffersFor(sp, ep);


        // extract corner features
        int largestPickedNum = 0;
        for (size_t k = regionSize; k > 0 && largestPickedNum < _maxCornerLessSharp;) {
          size_t idx = _regionSortIndices[--k];
          size_t scanIdx = idx - scanStartIdx;
          size_t regionIdx = idx - sp;

          if (_scanNeighborPicked[scanIdx] == 0 &&
              _regionCurvature[regionIdx] > _surfaceCurvatureThreshold) {

            largestPickedNum++;
            if (largestPickedNum <= _maxCornerSharp) {
              _regionLabel[regionIdx] = CORNER_SHARP;
              _cornerPointsSharp.push_back(_laserCloud[idx]);
            } else {
              _regionLabel[regionIdx] = CORNER_LESS_SHARP;
            }
            _cornerPointsLessSharp.push_back(_laserCloud[idx]);
            pcl::PointXYZI point;
            point.x = _laserCloud[idx].x;
            point.y = _laserCloud[idx].y;
            point.z = _laserCloud[idx].z;
            point.intensity = laserCloudIn[_laserCloud[idx].intensity].intensity;
            pc_out->push_back(point);
            edge_or_surface.push_back(0);
            selected_indexes.push_back(_laserCloud[idx].intensity);

            markAsPicked(idx, scanIdx);
          }
        }

        // extract flat surface features
        int smallestPickedNum = 0;
        for (int k = 0; k < regionSize && smallestPickedNum < _maxSurfaceFlat; k++) {
          size_t idx = _regionSortIndices[k];
          size_t scanIdx = idx - scanStartIdx;
          size_t regionIdx = idx - sp;

          if (_scanNeighborPicked[scanIdx] == 0 &&
              _regionCurvature[regionIdx] < _surfaceCurvatureThreshold) {

            smallestPickedNum++;
            _regionLabel[regionIdx] = SURFACE_FLAT;
            _surfacePointsFlat.push_back(_laserCloud[idx]);
            pcl::PointXYZI point;
            point.x = _laserCloud[idx].x;
            point.y = _laserCloud[idx].y;
            point.z = _laserCloud[idx].z;
            point.intensity = laserCloudIn[_laserCloud[idx].intensity].intensity;
            pc_out->push_back(point);
            edge_or_surface.push_back(1);
            selected_indexes.push_back(_laserCloud[idx].intensity);

            markAsPicked(idx, scanIdx);
          }
        }

        // extract less flat surface features
        for (int k = 0; k < regionSize; k++) {
          if (_regionLabel[k] <= SURFACE_LESS_FLAT) {
            surfPointsLessFlatScan->push_back(_laserCloud[sp + k]);
          }
        }
      }

      // down size less flat surface point cloud of current scan
      pcl::PointCloud<pcl::PointXYZI> surfPointsLessFlatScanDS;
      pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter;
      downSizeFilter.setInputCloud(surfPointsLessFlatScan);
      downSizeFilter.setLeafSize(_lessFlatFilterSize, _lessFlatFilterSize, _lessFlatFilterSize);
      downSizeFilter.filter(surfPointsLessFlatScanDS);

      _surfacePointsLessFlat += surfPointsLessFlatScanDS;
    }
    
    // std::cout << "point cloud: " << _cornerPointsSharp.size() << ", " << _cornerPointsLessSharp.size() << ", " << _surfacePointsFlat.size() << ", " << _surfacePointsLessFlat.size() << std::endl;
    // if (_cornerPointsSharp.size() > 0)
    //   pcl::io::savePCDFile("cornerPointsSharp.pcd", _cornerPointsSharp);
    // if (_cornerPointsLessSharp.size() > 0)
    //   pcl::io::savePCDFile("cornerPointsLessSharp.pcd", _cornerPointsLessSharp);
    // if (_surfacePointsFlat.size() > 0)
    //   pcl::io::savePCDFile("surfacePointsFlat.pcd", _surfacePointsFlat);
    // if (_surfacePointsLessFlat.size() > 0)
    //   pcl::io::savePCDFile("surfacePointsLessFlat.pcd", _surfacePointsLessFlat);

    compute_normal_and_remove_ground(pc_out, edge_or_surface);
  }

  void LoamScanRegistration::compute_normal_and_remove_ground(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out, 
                                                              std::vector <float> & edge_or_surface)
  {
    // // compute normal of the surface 
    // pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    // pcl::NormalEstimationOMP<pcl::PointXYZI, pcl::Normal> ne;
    // pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_surface;
    // // *pc_out_surface = pcl::createPointCloud(_surfacePointsFlat);
    // pc_out_surface = _surfacePointsFlat.makeShared();
    // ne.setInputCloud(pc_out_surface);
    // // Create an empty kdtree representation, and pass it to the normal estimation object.
    // // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    // pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI> ());
    // ne.setSearchMethod(tree);
    // // Use all neighbors in a sphere of radius 50cm
    // ne.setRadiusSearch(1);
    // // ne.setKSearch(30);
    // ne.compute(*normals);

    // // push back if normal is not ground
    // float threshold = 0.0875;  // 5 degree
    // pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_without_ground (new pcl::PointCloud<pcl::PointXYZI>);
    // std::vector <float> edge_or_surface_without_ground;
    // int index_surface = 0;
    // for (int i=0; i<pc_out->size(); i++){
    //   if (edge_or_surface[i] == 1){
    //     // surface, examinate the normal
    //     if (!isnan(normals->points[index_surface].normal_x)){
    //       float angle_xy = abs(normals->points[index_surface].normal_x / normals->points[index_surface].normal_y);
    //       float angle_zy = abs(normals->points[index_surface].normal_z / normals->points[index_surface].normal_y);
    //       if (angle_xy > threshold || angle_zy > threshold){
    //         // this is not the ground, push back
    //         pc_out_without_ground->push_back(pc_out->points[i]);
    //         edge_or_surface_without_ground.push_back(1);
    //       }
    //     }
    //     index_surface += 1;        
    //   }
    //   else if (edge_or_surface[i] == 0){
    //     // edge, just push back the point
    //     pc_out_without_ground->push_back(pc_out->points[i]);
    //     edge_or_surface_without_ground.push_back(0);
    //   }
    // }
    
    // // replace pointer (?)
    // pc_out = pc_out_without_ground;
    // edge_or_surface = edge_or_surface_without_ground;

  }

  void LoamScanRegistration::setRegionBuffersFor(const size_t& startIdx, const size_t& endIdx)
  {
    // resize buffers
    size_t regionSize = endIdx - startIdx + 1;
    _regionCurvature.resize(regionSize);
    _regionSortIndices.resize(regionSize);
    _regionLabel.assign(regionSize, SURFACE_LESS_FLAT);

    // calculate point curvatures and reset sort indices
    float pointWeight = -2 * _curvatureRegion;

    for (size_t i = startIdx, regionIdx = 0; i <= endIdx; i++, regionIdx++) {
      float diffX = pointWeight * _laserCloud[i].x;
      float diffY = pointWeight * _laserCloud[i].y;
      float diffZ = pointWeight * _laserCloud[i].z;

      for (int j = 1; j <= _curvatureRegion; j++) {
        diffX += _laserCloud[i + j].x + _laserCloud[i - j].x;
        diffY += _laserCloud[i + j].y + _laserCloud[i - j].y;
        diffZ += _laserCloud[i + j].z + _laserCloud[i - j].z;
      }

      _regionCurvature[regionIdx] = diffX * diffX + diffY * diffY + diffZ * diffZ;
      _regionSortIndices[regionIdx] = i;
    }

    // sort point curvatures
    for (size_t i = 1; i < regionSize; i++) {
      for (size_t j = i; j >= 1; j--) {
        if (_regionCurvature[_regionSortIndices[j] - startIdx] < _regionCurvature[_regionSortIndices[j - 1] - startIdx]) {
          std::swap(_regionSortIndices[j], _regionSortIndices[j - 1]);
        }
      }
    }
  }


  void LoamScanRegistration::setScanBuffersFor(const size_t& startIdx, const size_t& endIdx)
  {
    // resize buffers
    size_t scanSize = endIdx - startIdx + 1;
    _scanNeighborPicked.assign(scanSize, 0);

    // mark unreliable points as picked
    for (size_t i = startIdx + _curvatureRegion; i < endIdx - _curvatureRegion; i++) {
      const pcl::PointXYZI& previousPoint = (_laserCloud[i - 1]);
      const pcl::PointXYZI& point = (_laserCloud[i]);
      const pcl::PointXYZI& nextPoint = (_laserCloud[i + 1]);

      float diffNext = calcSquaredDiff(nextPoint, point);

      if (diffNext > 0.1) {
        float depth1 = calcPointDistance(point);
        float depth2 = calcPointDistance(nextPoint);

        if (depth1 > depth2) {
          float weighted_distance = std::sqrt(calcSquaredDiff(nextPoint, point, depth2 / depth1)) / depth2;

          if (weighted_distance < 0.1) {
            std::fill_n(&_scanNeighborPicked[i - startIdx - _curvatureRegion], _curvatureRegion + 1, 1);

            continue;
          }
        } else {
          float weighted_distance = std::sqrt(calcSquaredDiff(point, nextPoint, depth1 / depth2)) / depth1;

          if (weighted_distance < 0.1) {
            std::fill_n(&_scanNeighborPicked[i - startIdx + 1], _curvatureRegion + 1, 1);
          }
        }
      }

      float diffPrevious = calcSquaredDiff(point, previousPoint);
      float dis = calcSquaredPointDistance(point);

      if (diffNext > 0.0002 * dis && diffPrevious > 0.0002 * dis) {
        _scanNeighborPicked[i - startIdx] = 1;
      }
    }
  }



  void LoamScanRegistration::markAsPicked(const size_t& cloudIdx, const size_t& scanIdx)
  {
    _scanNeighborPicked[scanIdx] = 1;

    for (int i = 1; i <= _curvatureRegion; i++) {
      if (calcSquaredDiff(_laserCloud[cloudIdx + i], _laserCloud[cloudIdx + i - 1]) > 0.05) {
        break;
      }

      _scanNeighborPicked[scanIdx + i] = 1;
    }

    for (int i = 1; i <= _curvatureRegion; i++) {
      if (calcSquaredDiff(_laserCloud[cloudIdx - i], _laserCloud[cloudIdx - i + 1]) > 0.05) {
        break;
      }

      _scanNeighborPicked[scanIdx - i] = 1;
    }
  }

  int LoamScanRegistration::get_quadrant(pcl::PointXYZI point){
    int res = 0;
    /* because for kitti dataset lidar, we changed the coordinate...
    now.x = -raw.y;
    now.y = -raw.z;
    now.z = raw.x;
    */
    float x = point.z;
    float y = -point.x;

    if(x > 0 && y >= 0){res = 1;}
    else if(x <= 0 && y > 0){res = 2;}
    else if(x < 0 && y <= 0){res = 3;}
    else if(x >= 0 && y < 0){res = 4;}   

    return res;
  }



} // namespcae cvo