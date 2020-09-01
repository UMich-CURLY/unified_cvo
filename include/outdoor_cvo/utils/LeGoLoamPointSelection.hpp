// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following papers:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). October 2018.


#ifndef LEGOLOAMSCANREGISTRATION_HPP
#define LEGOLOAMSCANREGISTRATION_HPP

#pragma once
#include "LidarPointType.hpp"

namespace cvo
{


class LeGoLoamPointSelection{

public:
    LeGoLoamPointSelection();

    ~LeGoLoamPointSelection();
    
    void cloudHandler(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr pc_in, 
                      pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out, 
                      std::vector <float> & edge_or_surface,
                      std::vector <int> & selected_indexes);

private:
           
    // ImageProjection from LeGO-LOAM

    pcl::PointCloud<PointType>::Ptr laserCloudInPtr;
    pcl::PointCloud<PointType> laserCloudIn;
    pcl::PointCloud<pcl::PointXYZIR>::Ptr laserCloudInRing;

    pcl::PointCloud<PointType>::Ptr fullCloud; // projected velodyne raw cloud, but saved in the form of 1-D matrix
    pcl::PointCloud<PointType>::Ptr fullInfoCloud; // same as fullCloud, but with intensity - range

    pcl::PointCloud<PointType>::Ptr groundCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloudPure;
    pcl::PointCloud<PointType>::Ptr outlierCloud;

    PointType nanPoint; // fill in fullCloud at each iteration

    cv::Mat rangeMat; // range matrix for range image
    cv::Mat labelMat; // label matrix for segmentaiton marking
    cv::Mat groundMat; // ground matrix for ground cloud marking
    int labelCount;

    // cloud_msgs::cloud_info segMsg; // info of segmented cloud
    // std_msgs::Header cloudHeader;

    // cloud_info.msg
    std::vector<int> startRingIndex;
    std::vector<int> endRingIndex;

    float startOrientation;
    float endOrientation;
    float orientationDiff;

    std::vector<bool> segmentedCloudGroundFlag; // true - ground point, false - other points
    std::vector<int> segmentedCloudColInd; // point column index in range image
    std::vector<float> segmentedCloudRange; // point range 

    // continued in ImageProjection

    std::vector<std::pair<int8_t, int8_t> > neighborIterator; // neighbor iterator for segmentaiton process

    uint16_t *allPushedIndX; // array for tracking points of a segmented object
    uint16_t *allPushedIndY;

    uint16_t *queueIndX; // array for breadth-first search process of segmentation, for speed
    uint16_t *queueIndY;

    // moved from public to private
    void allocateMemory();
    void resetParameters();
    void copyPointCloud(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr pc_in);

    void findStartEndAngle();

    void projectPointCloud();
    void groundRemoval();
    void cloudSegmentation(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_segmented);
    void labelComponents(int row, int col);

    // FeatureAssociation from LeGO-LOAM

    // pcl::PointCloud<PointType>::Ptr segmentedCloud;
    // pcl::PointCloud<PointType>::Ptr outlierCloud;

    pcl::PointCloud<PointType>::Ptr cornerPointsSharp;
    pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp;
    pcl::PointCloud<PointType>::Ptr surfPointsFlat;
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlat;

    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan;
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScanDS;

    pcl::VoxelGrid<PointType> downSizeFilter;

    double timeScanCur;
    double timeNewSegmentedCloud;
    double timeNewSegmentedCloudInfo;
    double timeNewOutlierCloud;

    bool newSegmentedCloud;
    bool newSegmentedCloudInfo;
    bool newOutlierCloud;

    // cloud_msgs::cloud_info segInfo;
    // std_msgs::Header cloudHeader;

    int systemInitCount;
    bool systemInited;

    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    int *cloudNeighborPicked;
    int *cloudLabel;

    // int imuPointerFront;
    // int imuPointerLast;
    // int imuPointerLastIteration;

    // float imuRollStart, imuPitchStart, imuYawStart;
    // float cosImuRollStart, cosImuPitchStart, cosImuYawStart, sinImuRollStart, sinImuPitchStart, sinImuYawStart;
    // float imuRollCur, imuPitchCur, imuYawCur;

    // float imuVeloXStart, imuVeloYStart, imuVeloZStart;
    // float imuShiftXStart, imuShiftYStart, imuShiftZStart;

    // float imuVeloXCur, imuVeloYCur, imuVeloZCur;
    // float imuShiftXCur, imuShiftYCur, imuShiftZCur;

    // float imuShiftFromStartXCur, imuShiftFromStartYCur, imuShiftFromStartZCur;
    // float imuVeloFromStartXCur, imuVeloFromStartYCur, imuVeloFromStartZCur;

    // float imuAngularRotationXCur, imuAngularRotationYCur, imuAngularRotationZCur;
    // float imuAngularRotationXLast, imuAngularRotationYLast, imuAngularRotationZLast;
    // float imuAngularFromStartX, imuAngularFromStartY, imuAngularFromStartZ;

    // double imuTime[imuQueLength];
    // float imuRoll[imuQueLength];
    // float imuPitch[imuQueLength];
    // float imuYaw[imuQueLength];

    // float imuAccX[imuQueLength];
    // float imuAccY[imuQueLength];
    // float imuAccZ[imuQueLength];

    // float imuVeloX[imuQueLength];
    // float imuVeloY[imuQueLength];
    // float imuVeloZ[imuQueLength];

    // float imuShiftX[imuQueLength];
    // float imuShiftY[imuQueLength];
    // float imuShiftZ[imuQueLength];

    // float imuAngularVeloX[imuQueLength];
    // float imuAngularVeloY[imuQueLength];
    // float imuAngularVeloZ[imuQueLength];

    // float imuAngularRotationX[imuQueLength];
    // float imuAngularRotationY[imuQueLength];
    // float imuAngularRotationZ[imuQueLength];



    // ros::Publisher pubLaserCloudCornerLast;
    // ros::Publisher pubLaserCloudSurfLast;
    // ros::Publisher pubLaserOdometry;
    // ros::Publisher pubOutlierCloudLast;

    int skipFrameNum;
    bool systemInitedLM;

    int laserCloudCornerLastNum;
    int laserCloudSurfLastNum;

    int *pointSelCornerInd;
    float *pointSearchCornerInd1;
    float *pointSearchCornerInd2;

    int *pointSelSurfInd;
    float *pointSearchSurfInd1;
    float *pointSearchSurfInd2;
    float *pointSearchSurfInd3;

    float transformCur[6];
    float transformSum[6];

    float imuRollLast, imuPitchLast, imuYawLast;
    float imuShiftFromStartX, imuShiftFromStartY, imuShiftFromStartZ;
    float imuVeloFromStartX, imuVeloFromStartY, imuVeloFromStartZ;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;
    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerLast;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfLast;

    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    PointType pointOri, pointSel, tripod1, tripod2, tripod3, pointProj, coeff;

    // nav_msgs::Odometry laserOdometry;

    // tf::TransformBroadcaster tfBroadcaster;
    // tf::StampedTransform laserOdometryTrans;

    bool isDegenerate;
    cv::Mat matP;

    int frameCount;

    void initializationValue();
    void runFeatureAssociation(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out, std::vector <float> & edge_or_surface, std::vector <int> & selected_indexes);
    void adjustDistortion();
    void calculateSmoothness();
    void markOccludedPoints();
    void extractFeatures(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out, std::vector <float> & edge_or_surface, std::vector <int> & selected_indexes);


    // From utility.h

  const std::string pointCloudTopic = "/kitti/velo/pointcloud";
  const std::string imuTopic = "/imu/data";

    // Save pcd
  const std::string fileDirectory = "/tmp/";

    // Using velodyne cloud "ring" channel for image projection (other lidar may have different name for this channel, change "PointXYZIR" below)
    const bool useCloudRing = true; // if true, ang_res_y and ang_bottom are not used

    // VLP-16
    // extern const int N_SCAN = 16;
    // extern const int Horizon_SCAN = 1800;
    // extern const float ang_res_x = 0.2;
    // extern const float ang_res_y = 2.0;
    // extern const float ang_bottom = 15.0+0.1;
    // extern const int groundScanInd = 7;

    // HDL-32E
    // extern const int N_SCAN = 32;
    // extern const int Horizon_SCAN = 1800;
    // extern const float ang_res_x = 360.0/float(Horizon_SCAN);
    // extern const float ang_res_y = 41.33/float(N_SCAN-1);
    // extern const float ang_bottom = 30.67;
    // extern const int groundScanInd = 20;

    // VLS-128
    // extern const int N_SCAN = 128;
    // extern const int Horizon_SCAN = 1800;
    // extern const float ang_res_x = 0.2;
    // extern const float ang_res_y = 0.3;
    // extern const float ang_bottom = 25.0;
    // extern const int groundScanInd = 10;

    // HDL-64E
    const int N_SCAN = 64;
    const int Horizon_SCAN = 1800;
    const float ang_res_x = 0.2;
    const float ang_res_y = 0.427;
    const float ang_bottom = 24.9;
    const int groundScanInd = 50;

    // Ouster users may need to uncomment line 159 in imageProjection.cpp
    // Usage of Ouster imu data is not fully supported yet (LeGO-LOAM needs 9-DOF IMU), please just publish point cloud data
    // Ouster OS1-16
    // extern const int N_SCAN = 16;
    // extern const int Horizon_SCAN = 1024;
    // extern const float ang_res_x = 360.0/float(Horizon_SCAN);
    // extern const float ang_res_y = 33.2/float(N_SCAN-1);
    // extern const float ang_bottom = 16.6+0.1;
    // extern const int groundScanInd = 7;

    // Ouster OS1-64
    // extern const int N_SCAN = 64;
    // extern const int Horizon_SCAN = 1024;
    // extern const float ang_res_x = 360.0/float(Horizon_SCAN);
    // extern const float ang_res_y = 33.2/float(N_SCAN-1);
    // extern const float ang_bottom = 16.6+0.1;
    // extern const int groundScanInd = 15;

    const bool loopClosureEnableFlag = false;
    const double mappingProcessInterval = 0.3;

    const float scanPeriod = 0.1;
    const int systemDelay = 0;
    const int imuQueLength = 200;

    const float sensorMinimumRange = 1.0;
    const float sensorMountAngle = 0.0;
    const float segmentTheta = 60.0/180.0*M_PI; // decrese this value may improve accuracy
    const int segmentValidPointNum = 5;
    const int segmentValidLineNum = 3;
    const float segmentAlphaX = ang_res_x / 180.0 * M_PI;
    const float segmentAlphaY = ang_res_y / 180.0 * M_PI;


    const int edgeFeatureNum = 2;
    const int surfFeatureNum = 4;
    const int sectionsTotal = 6;
    const float edgeThreshold = 0.1;
    const float surfThreshold = 0.1;
    const float nearestFeatureSearchSqDist = 25;


    // Mapping Params
    const float surroundingKeyframeSearchRadius = 50.0; // key frame that is within n meters from current pose will be considerd for scan-to-map optimization (when loop closure disabled)
    const int   surroundingKeyframeSearchNum = 50; // submap size (when loop closure enabled)
    // history key frames (history submap for loop closure)
    const float historyKeyframeSearchRadius = 7.0; // key frame that is within n meters from current pose will be considerd for loop closure
    const int   historyKeyframeSearchNum = 25; // 2n+1 number of hostory key frames will be fused into a submap for loop closure
    const float historyKeyframeFitnessScore = 0.3; // the smaller the better alignment

    const float globalMapVisualizationSearchRadius = 500.0; // key frames with in n meters will be visualized

    // My own fuction
    int get_quadrant(pcl::PointXYZI point);


};


}


#endif
