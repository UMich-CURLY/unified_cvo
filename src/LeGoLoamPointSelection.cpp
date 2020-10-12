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

#include "utils/LeGoLoamPointSelection.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <cstdlib>

namespace cvo{

    LeGoLoamPointSelection::LeGoLoamPointSelection()
    {
        // imageProjection
        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;

        allocateMemory();
        resetParameters();

        //  Feature Association
        initializationValue();
    }    

    LeGoLoamPointSelection::~LeGoLoamPointSelection(){}
    
    void LeGoLoamPointSelection::cloudHandler(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr pc_in, 
                                              pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out, 
                                              std::vector <float> & edge_or_surface,
                                              std::vector <int> & selected_indexes){

        pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_segmented (new pcl::PointCloud<pcl::PointXYZI>);

        // Running imageProjection

        // 1. Convert ros message to pcl point cloud
        copyPointCloud(pc_in);
        // 2. Start and end angle of a scan
        // findStartEndAngle();
        // 3. Range image projection
        projectPointCloud();
        // 4. Mark ground points
        groundRemoval();
        // 5. Point cloud segmentation
        cloudSegmentation(pc_out_segmented);
        // 6. Reset parameters for next iteration
        // resetParameters();

        // Running Feature Association
        runFeatureAssociation(pc_out, edge_or_surface, selected_indexes);
    }


  // ImageProjection from LeGO-LOAM

    // moved from public to private
    void LeGoLoamPointSelection::allocateMemory(){

        //laserCloudIn.reset(new pcl::PointCloud<PointType>());
        laserCloudInRing.reset(new pcl::PointCloud<pcl::PointXYZIR>());

        fullCloud.reset(new pcl::PointCloud<PointType>());
        fullInfoCloud.reset(new pcl::PointCloud<PointType>());

        groundCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloudPure.reset(new pcl::PointCloud<PointType>());
        outlierCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);
        fullInfoCloud->points.resize(N_SCAN*Horizon_SCAN);

        startRingIndex.assign(N_SCAN, 0);
        endRingIndex.assign(N_SCAN, 0);

        segmentedCloudGroundFlag.assign(N_SCAN*Horizon_SCAN, false);
        segmentedCloudColInd.assign(N_SCAN*Horizon_SCAN, 0);
        segmentedCloudRange.assign(N_SCAN*Horizon_SCAN, 0);

        std::pair<int8_t, int8_t> neighbor;
        neighbor.first = -1; neighbor.second =  0; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second =  1; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second = -1; neighborIterator.push_back(neighbor);
        neighbor.first =  1; neighbor.second =  0; neighborIterator.push_back(neighbor);

        allPushedIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        allPushedIndY = new uint16_t[N_SCAN*Horizon_SCAN];

        queueIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        queueIndY = new uint16_t[N_SCAN*Horizon_SCAN];
    }

    void LeGoLoamPointSelection::resetParameters(){
        //laserCloudIn->clear();
        laserCloudIn.clear();
        groundCloud->clear();
        segmentedCloud->clear();
        segmentedCloudPure->clear();
        outlierCloud->clear();

        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
        groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
        labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
        labelCount = 1;

        std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
        std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);
    }

    void LeGoLoamPointSelection::copyPointCloud(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr pc_in){

        // Remove Nan points
        //laserCloudIn = pc_in;
	laserCloudIn = *pc_in;
        //std::cout<<"input size: "<<laserCloudIn->size()<<std::endl;
        //std::vector<int> indices;
        //pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
        
	//std::cout<<"output size: "<<laserCloudIn->size()<<std::endl;
        //for(int a=0; a<indices.size(); ++a)
        //    std::cout << indices[a] << ' ';
        //std::cout<<std::endl;

        // if (laserCloudIn->size() > 0)
        //     pcl::io::savePCDFile("legoloam_laserCloudIn.pcd", *laserCloudIn);


        // have "ring" channel in the cloud
        if (useCloudRing == true){
            // initialize for XYZIR point cloud
            size_t cloudSize = laserCloudIn.size();
            pcl::PointXYZIR point;
            int previous_quadrant = get_quadrant(laserCloudIn.points[0]);
            int quadrant = get_quadrant(laserCloudIn.points[0]);
            int scanID = 0;
            for (int i = 0; i < cloudSize; i++) {
                //skip is a NaN point
		if(laserCloudIn.points[i].x != laserCloudIn.points[i].x){
		    std::cout<<"Found a NaN point"<<std::endl;
		    continue;
		}
		// change scanID to be determined by quadrant
                previous_quadrant = quadrant;
                quadrant = get_quadrant(laserCloudIn.points[i]);
                if(quadrant == 1 && previous_quadrant == 4){
                    scanID += 1;
                }

                point.x = laserCloudIn.points[i].x;
                point.y = laserCloudIn.points[i].y;
                point.z = laserCloudIn.points[i].z;
                // instead of keeping the intensity information, we would like to keep track of the indexes of the point, so we can add back the information after point selection.
                // point.intensity = laserCloudIn->points[i].intensity;
                point.intensity = i;
                //std::cout<<"index for point "<<i<<" is "<<point.intensity<<std::endl;
                
                point.ring = scanID;
                laserCloudInRing->push_back(point);
            }
            if (laserCloudInRing->is_dense == false) {
              std::cout << "Point cloud is not in dense format, please remove NaN points first!" << std::endl;
              return;
            }  
        }
    }

    void LeGoLoamPointSelection::findStartEndAngle(){
        // start and end orientation of this cloud
        startOrientation = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
        endOrientation   = -atan2(laserCloudIn.points[laserCloudIn.points.size() - 1].y,
                                                     laserCloudIn.points[laserCloudIn.points.size() - 1].x) + 2 * M_PI;
        if (endOrientation - startOrientation > 3 * M_PI) {
            endOrientation -= 2 * M_PI;
        } else if (endOrientation - startOrientation < M_PI)
            endOrientation += 2 * M_PI;
        orientationDiff = endOrientation - startOrientation;
    }

    void LeGoLoamPointSelection::projectPointCloud(){
        // range image projection
        float verticalAngle, horizonAngle, range;
        size_t rowIdn, columnIdn, index, cloudSize; 
        PointType thisPoint;

        cloudSize = laserCloudIn.points.size();

        for (size_t i = 0; i < cloudSize; ++i){

            if(laserCloudIn.points[i].x != laserCloudIn.points[i].x){
                std::cout<<"Found a NaN point"<<std::endl;
                continue;
            }

            thisPoint.x = laserCloudIn.points[i].x;
            thisPoint.y = laserCloudIn.points[i].y;
            thisPoint.z = laserCloudIn.points[i].z;
            // find the row and column index in the iamge for this point
            if (useCloudRing == true){
                rowIdn = laserCloudInRing->points[i].ring;
            }
            else{
                // verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
                // change to our coordinate
                verticalAngle = atan2(-thisPoint.y, sqrt(thisPoint.x * thisPoint.x + thisPoint.z * thisPoint.z)) * 180 / M_PI;
                rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
            }
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            // horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
            // change to our coordinate
            horizonAngle = atan2(thisPoint.z, -thisPoint.x) * 180 / M_PI;

            columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
            if (range < sensorMinimumRange)
                continue;
            
            rangeMat.at<float>(rowIdn, columnIdn) = range;

            // instead of keeping the intensity information, we would like to keep track of the indexes of the point, so we can add back the information after point selection.
            thisPoint.intensity = i;
            // thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;
            // thisPoint.intensity = laserCloudIn->points[i].intensity;

            index = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
            fullInfoCloud->points[index] = thisPoint;
            
            //std::cout<<"index for point "<<i<<" is "<<thisPoint.intensity<<std::endl;

            // fullInfoCloud->points[index].intensity = range; // the corresponding range of a point is saved as "intensity"
            // fullInfoCloud->points[index].intensity = laserCloudIn->points[i].intensity;
        }
    }


    void LeGoLoamPointSelection::groundRemoval(){
        size_t lowerInd, upperInd;
        float diffX, diffY, diffZ, angle, thispoint_angle;
        // groundMat
        // -1, no valid info to check if ground of not
        //  0, initial value, after validation, means not ground
        //  1, ground
        for (size_t j = 0; j < Horizon_SCAN; ++j){
            for (size_t i = 0; i < groundScanInd; ++i){

                lowerInd = j + ( i )*Horizon_SCAN;
                upperInd = j + (i+1)*Horizon_SCAN;

                if (fullCloud->points[lowerInd].intensity == -1 ||
                    fullCloud->points[upperInd].intensity == -1){
                    // no info to check, invalid points
                    groundMat.at<int8_t>(i,j) = -1;
                    continue;
                }
                    
                diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
                diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
                diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;

                // angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;
                // change to our coordinate
                angle = atan2(diffY, sqrt(diffX*diffX + diffZ*diffZ) ) * 180 / M_PI;
                thispoint_angle = atan2(fullCloud->points[lowerInd].y, sqrt(fullCloud->points[lowerInd].x*fullCloud->points[lowerInd].x + fullCloud->points[lowerInd].z*fullCloud->points[lowerInd].z) ) * 180 / M_PI;


                if ((abs(angle - sensorMountAngle) <= 10) && (abs(thispoint_angle - sensorMountAngle) > 3)){  // TODO: this angle should be tuned, original 10
                    groundMat.at<int8_t>(i,j) = 1;
                    groundMat.at<int8_t>(i+1,j) = 1;
                }
            }
        }
        // extract ground cloud (groundMat == 1)
        // mark entry that doesn't need to label (ground and invalid point) for segmentation
        // note that ground remove is from 0~N_SCAN-1, need rangeMat for mark label matrix for the 16th scan
        for (size_t i = 0; i < N_SCAN; ++i){
            for (size_t j = 0; j < Horizon_SCAN; ++j){
                if (groundMat.at<int8_t>(i,j) == 1 || rangeMat.at<float>(i,j) == FLT_MAX){
                    labelMat.at<int>(i,j) = -1;
                }
            }
        }
        for (size_t i = 0; i <= groundScanInd; ++i){
            for (size_t j = 0; j < Horizon_SCAN; ++j){
                if (groundMat.at<int8_t>(i,j) == 1)
                    groundCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
            }
        }
        // if (groundCloud->size() > 0)
        //     pcl::io::savePCDFile("legoloam_groundCloud_final.pcd", *groundCloud);
    }

    void LeGoLoamPointSelection::cloudSegmentation(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_segmented){
        // segmentation process
        for (size_t i = 0; i < N_SCAN; ++i)
            for (size_t j = 0; j < Horizon_SCAN; ++j)
                if (labelMat.at<int>(i,j) == 0)
                    labelComponents(i, j);


        int sizeOfSegCloud = 0;
        // extract segmented cloud for lidar odometry
        for (size_t i = 0; i < N_SCAN; ++i) {

            startRingIndex[i] = sizeOfSegCloud-1 + 5;

            for (size_t j = 0; j < Horizon_SCAN; ++j) {
                if (labelMat.at<int>(i,j) > 0 || groundMat.at<int8_t>(i,j) == 1){
                    // outliers that will not be used for optimization (always continue)
                    if (labelMat.at<int>(i,j) == 999999){
                        if (i > groundScanInd && j % 5 == 0){
                            outlierCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                            continue;
                        }else{
                            continue;
                        }
                    }
                    // majority of ground points are skipped
                    if (groundMat.at<int8_t>(i,j) == 1){
                         //if (  std::rand() % 100 < 1 )  // TODO: this should be tuned, skip more! original num = 5 (j%5!=0 && j>5 && j<Horizon_SCAN-5)
                           continue;
                    }
                    // mark ground points so they will not be considered as edge features later
                    segmentedCloudGroundFlag[sizeOfSegCloud] = (groundMat.at<int8_t>(i,j) == 1);
                    // mark the points' column index for marking occlusion later
                    segmentedCloudColInd[sizeOfSegCloud] = j;
                    // save range info
                    segmentedCloudRange[sizeOfSegCloud]  = rangeMat.at<float>(i,j);
                    // save seg cloud
                    segmentedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // output
                    pc_out_segmented->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of seg cloud
                    ++sizeOfSegCloud;
                }
            }

            endRingIndex[i] = sizeOfSegCloud-1 - 5;
        }
        
        
        // extract segmented cloud for visualization
        //for (size_t i = 0; i < N_SCAN; ++i){
        //    for (size_t j = 0; j < Horizon_SCAN; ++j){
        //        if (labelMat.at<int>(i,j) > 0 && labelMat.at<int>(i,j) != 999999){
        //            segmentedCloudPure->push_back(fullCloud->points[j + i*Horizon_SCAN]);
        //            segmentedCloudPure->points.back().intensity = labelMat.at<int>(i,j);
        //        }
        //    }
        //}

        // if (segmentedCloud->size() > 0)
        //     pcl::io::savePCDFile("legoloam_segmentedCloud_final.pcd", *segmentedCloud);
        // if (outlierCloud->size() > 0)
        //     pcl::io::savePCDFile("legoloam_outlierCloud_final.pcd", *outlierCloud);
    }

    void LeGoLoamPointSelection::labelComponents(int row, int col){
        // use std::queue std::vector std::deque will slow the program down greatly
        float d1, d2, alpha, angle;
        int fromIndX, fromIndY, thisIndX, thisIndY; 
        bool lineCountFlag[N_SCAN] = {false};

        queueIndX[0] = row;
        queueIndY[0] = col;
        int queueSize = 1;
        int queueStartInd = 0;
        int queueEndInd = 1;

        allPushedIndX[0] = row;
        allPushedIndY[0] = col;
        int allPushedIndSize = 1;
        
        while(queueSize > 0){
            // Pop point
            fromIndX = queueIndX[queueStartInd];
            fromIndY = queueIndY[queueStartInd];
            --queueSize;
            ++queueStartInd;
            // Mark popped point
            labelMat.at<int>(fromIndX, fromIndY) = labelCount;
            // Loop through all the neighboring grids of popped grid
            for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter){
                // new index
                thisIndX = fromIndX + (*iter).first;
                thisIndY = fromIndY + (*iter).second;
                // index should be within the boundary
                if (thisIndX < 0 || thisIndX >= N_SCAN)
                    continue;
                // at range image margin (left or right side)
                if (thisIndY < 0)
                    thisIndY = Horizon_SCAN - 1;
                if (thisIndY >= Horizon_SCAN)
                    thisIndY = 0;
                // prevent infinite loop (caused by put already examined point back)
                if (labelMat.at<int>(thisIndX, thisIndY) != 0)
                    continue;

                d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY), 
                              rangeMat.at<float>(thisIndX, thisIndY));
                d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY), 
                              rangeMat.at<float>(thisIndX, thisIndY));

                if ((*iter).first == 0)
                    alpha = segmentAlphaX;
                else
                    alpha = segmentAlphaY;

                angle = atan2(d2*sin(alpha), (d1 -d2*cos(alpha)));

                if (angle > segmentTheta){

                    queueIndX[queueEndInd] = thisIndX;
                    queueIndY[queueEndInd] = thisIndY;
                    ++queueSize;
                    ++queueEndInd;

                    labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                    lineCountFlag[thisIndX] = true;

                    allPushedIndX[allPushedIndSize] = thisIndX;
                    allPushedIndY[allPushedIndSize] = thisIndY;
                    ++allPushedIndSize;
                }
            }
        }

        // check if this segment is valid
        bool feasibleSegment = false;
        if (allPushedIndSize >= 30)
            feasibleSegment = true;
        else if (allPushedIndSize >= segmentValidPointNum){
            int lineCount = 0;
            for (size_t i = 0; i < N_SCAN; ++i)
                if (lineCountFlag[i] == true)
                    ++lineCount;
            if (lineCount >= segmentValidLineNum)
                feasibleSegment = true;            
        }
        // segment is valid, mark these points
        if (feasibleSegment == true){
            ++labelCount;
        }else{ // segment is invalid, mark these points
            for (size_t i = 0; i < allPushedIndSize; ++i){
                labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
            }
        }
    }



    // FeatureAssociation from LeGO-LOAM

    void LeGoLoamPointSelection::initializationValue()
    {
        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];

        pointSelCornerInd = new int[N_SCAN*Horizon_SCAN];
        pointSearchCornerInd1 = new float[N_SCAN*Horizon_SCAN];
        pointSearchCornerInd2 = new float[N_SCAN*Horizon_SCAN];

        pointSelSurfInd = new int[N_SCAN*Horizon_SCAN];
        pointSearchSurfInd1 = new float[N_SCAN*Horizon_SCAN];
        pointSearchSurfInd2 = new float[N_SCAN*Horizon_SCAN];
        pointSearchSurfInd3 = new float[N_SCAN*Horizon_SCAN];

        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);

        downSizeFilter.setLeafSize(0.7, 0.7, 0.7);  // TODO: this should be tuned, original (0.2, 0.2, 0.2)

        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        outlierCloud.reset(new pcl::PointCloud<PointType>());

        cornerPointsSharp.reset(new pcl::PointCloud<PointType>());
        cornerPointsLessSharp.reset(new pcl::PointCloud<PointType>());
        surfPointsFlat.reset(new pcl::PointCloud<PointType>());
        surfPointsLessFlat.reset(new pcl::PointCloud<PointType>());

        surfPointsLessFlatScan.reset(new pcl::PointCloud<PointType>());
        surfPointsLessFlatScanDS.reset(new pcl::PointCloud<PointType>());

        timeScanCur = 0;
        timeNewSegmentedCloud = 0;
        timeNewSegmentedCloudInfo = 0;
        timeNewOutlierCloud = 0;

        // newSegmentedCloud = false;
        // newSegmentedCloudInfo = false;
        // newOutlierCloud = false;

        systemInitCount = 0;
        systemInited = false;

        // imuPointerFront = 0;
        // imuPointerLast = -1;
        // imuPointerLastIteration = 0;

        // imuRollStart = 0; imuPitchStart = 0; imuYawStart = 0;
        // cosImuRollStart = 0; cosImuPitchStart = 0; cosImuYawStart = 0;
        // sinImuRollStart = 0; sinImuPitchStart = 0; sinImuYawStart = 0;
        // imuRollCur = 0; imuPitchCur = 0; imuYawCur = 0;

        // imuVeloXStart = 0; imuVeloYStart = 0; imuVeloZStart = 0;
        // imuShiftXStart = 0; imuShiftYStart = 0; imuShiftZStart = 0;

        // imuVeloXCur = 0; imuVeloYCur = 0; imuVeloZCur = 0;
        // imuShiftXCur = 0; imuShiftYCur = 0; imuShiftZCur = 0;

        // imuShiftFromStartXCur = 0; imuShiftFromStartYCur = 0; imuShiftFromStartZCur = 0;
        // imuVeloFromStartXCur = 0; imuVeloFromStartYCur = 0; imuVeloFromStartZCur = 0;

        // imuAngularRotationXCur = 0; imuAngularRotationYCur = 0; imuAngularRotationZCur = 0;
        // imuAngularRotationXLast = 0; imuAngularRotationYLast = 0; imuAngularRotationZLast = 0;
        // imuAngularFromStartX = 0; imuAngularFromStartY = 0; imuAngularFromStartZ = 0;

        // for (int i = 0; i < imuQueLength; ++i)
        // {
        //     imuTime[i] = 0;
        //     imuRoll[i] = 0; imuPitch[i] = 0; imuYaw[i] = 0;
        //     imuAccX[i] = 0; imuAccY[i] = 0; imuAccZ[i] = 0;
        //     imuVeloX[i] = 0; imuVeloY[i] = 0; imuVeloZ[i] = 0;
        //     imuShiftX[i] = 0; imuShiftY[i] = 0; imuShiftZ[i] = 0;
        //     imuAngularVeloX[i] = 0; imuAngularVeloY[i] = 0; imuAngularVeloZ[i] = 0;
        //     imuAngularRotationX[i] = 0; imuAngularRotationY[i] = 0; imuAngularRotationZ[i] = 0;
        // }


        skipFrameNum = 1;

        for (int i = 0; i < 6; ++i){
            transformCur[i] = 0;
            transformSum[i] = 0;
        }

        systemInitedLM = false;

        imuRollLast = 0; imuPitchLast = 0; imuYawLast = 0;
        imuShiftFromStartX = 0; imuShiftFromStartY = 0; imuShiftFromStartZ = 0;
        imuVeloFromStartX = 0; imuVeloFromStartY = 0; imuVeloFromStartZ = 0;

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());
        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerLast.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfLast.reset(new pcl::KdTreeFLANN<PointType>());

        // laserOdometry.header.frame_id = "/camera_init";
        // laserOdometry.child_frame_id = "/laser_odom";

        // laserOdometryTrans.frame_id_ = "/camera_init";
        // laserOdometryTrans.child_frame_id_ = "/laser_odom";
        
        isDegenerate = false;
        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));

        frameCount = skipFrameNum;
    }

    void LeGoLoamPointSelection::runFeatureAssociation(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out, 
                                                        std::vector <float> & edge_or_surface, 
                                                        std::vector <int> & selected_indexes)
    {

        /**
        	1. Feature Extraction
        */
        // adjustDistortion();

        calculateSmoothness();

        markOccludedPoints();

        extractFeatures(pc_out, edge_or_surface, selected_indexes);

        // publishCloud(); // cloud for visualization
	
        // /**
		    // 2. Feature Association
        // */
        // if (!systemInitedLM) {
        //     checkSystemInitialization();
        //     return;
        // }

        // updateInitialGuess();

        // updateTransformation();

        // integrateTransformation();

        // publishOdometry();

        // publishCloudsLast(); // cloud to mapOptimization
    }

    void LeGoLoamPointSelection::calculateSmoothness()
    {
        int cloudSize = segmentedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; i++) {

            float diffRange = segmentedCloudRange[i-5] + segmentedCloudRange[i-4]
                            + segmentedCloudRange[i-3] + segmentedCloudRange[i-2]
                            + segmentedCloudRange[i-1] - segmentedCloudRange[i] * 10
                            + segmentedCloudRange[i+1] + segmentedCloudRange[i+2]
                            + segmentedCloudRange[i+3] + segmentedCloudRange[i+4]
                            + segmentedCloudRange[i+5];  

            cloudCurvature[i] = diffRange*diffRange;

            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;

            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

    void LeGoLoamPointSelection::markOccludedPoints()
    {
        int cloudSize = segmentedCloud->points.size();

        for (int i = 5; i < cloudSize - 6; ++i){

            float depth1 = segmentedCloudRange[i];
            float depth2 = segmentedCloudRange[i+1];
            int columnDiff = std::abs(int(segmentedCloudColInd[i+1] - segmentedCloudColInd[i]));

            if (columnDiff < 10){

                if (depth1 - depth2 > 0.3){
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.3){
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }

            float diff1 = std::abs(float(segmentedCloudRange[i-1] - segmentedCloudRange[i]));
            float diff2 = std::abs(float(segmentedCloudRange[i+1] - segmentedCloudRange[i]));

            if (diff1 > 0.02 * segmentedCloudRange[i] && diff2 > 0.02 * segmentedCloudRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    void LeGoLoamPointSelection::extractFeatures(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out, 
                                                 std::vector <float> & edge_or_surface,
                                                 std::vector <int> & selected_indexes)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_temp (new pcl::PointCloud<pcl::PointXYZI>);
        cornerPointsSharp->clear();
        cornerPointsLessSharp->clear();
        surfPointsFlat->clear();
        surfPointsLessFlat->clear();


        for (int i = 0; i < N_SCAN; i++) {

            surfPointsLessFlatScan->clear();

            for (int j = 0; j < 6; j++) {

                int sp = (startRingIndex[i] * (6 - j)    + endRingIndex[i] * j) / 6;
                int ep = (startRingIndex[i] * (5 - j)    + endRingIndex[i] * (j + 1)) / 6 - 1;


                if (sp >= ep)
                    continue;

                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());

                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--) {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 &&
                        cloudCurvature[ind] > edgeThreshold &&
                        segmentedCloudGroundFlag[ind] == false) {
                    
                        largestPickedNum++;
                        if (largestPickedNum <= 2) {
                            cloudLabel[ind] = 2;
                            cornerPointsSharp->push_back(segmentedCloud->points[ind]);
                            cornerPointsLessSharp->push_back(segmentedCloud->points[ind]);
                            // output edges to CvoPointCloud
                            pc_out_temp->push_back(segmentedCloud->points[ind]);
                            //std::cout<<"index for point("<<i<<","<<j<<") is "<<segmentedCloud->points[ind].intensity<<std::endl;
                            // edge_or_surface.push_back(0);
                        } else if (largestPickedNum <= 20) {
                            cloudLabel[ind] = 1;
                            cornerPointsLessSharp->push_back(segmentedCloud->points[ind]);
                            // output edges to CvoPointCloud
                            pc_out_temp->push_back(segmentedCloud->points[ind]);
                            //std::cout<<"index for point("<<i<<","<<j<<") is "<<segmentedCloud->points[ind].intensity<<std::endl;
                            // edge_or_surface.push_back(0);
                        } else {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++) {
                            int columnDiff = std::abs(int(segmentedCloudColInd[ind + l] - segmentedCloudColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {
                            int columnDiff = std::abs(int(segmentedCloudColInd[ind + l] - segmentedCloudColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                int smallestPickedNum = 0;
                for (int k = sp; k <= ep; k++) {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 &&
                        cloudCurvature[ind] < surfThreshold &&
                        segmentedCloudGroundFlag[ind] == true) {

                        cloudLabel[ind] = -1;
                        surfPointsFlat->push_back(segmentedCloud->points[ind]);
                        //std::cout<<"index for point["<<ind<<"] is "<<segmentedCloud->points[ind].intensity<<std::endl;
                        // output surface points to CvoPointCloud
                        // pc_out_temp->push_back(segmentedCloud->points[ind]);
                        // edge_or_surface.push_back(1);

                        smallestPickedNum++;
                        if (smallestPickedNum >= 4) {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++) {

                            int columnDiff = std::abs(int(segmentedCloudColInd[ind + l] - segmentedCloudColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {

                            int columnDiff = std::abs(int(segmentedCloudColInd[ind + l] - segmentedCloudColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++) {
                    if (cloudLabel[k] <= 0) {
                          if (std::rand() % 4 == 0){
                              pc_out_temp->push_back(segmentedCloud->points[k]);
                          }
//                        surfPointsLessFlatScan->push_back(segmentedCloud->points[k]);
                        //std::cout<<"index for point["<<k<<"] is "<<segmentedCloud->points[k].intensity<<std::endl;

                    }
                }
            }

            surfPointsLessFlatScanDS->clear();
            //downSizeFilter.setInputCloud(surfPointsLessFlatScan);
            //downSizeFilter.filter(*surfPointsLessFlatScanDS);
            // std::cout << "before = " << surfPointsLessFlatScan->size() << ", after = " << surfPointsLessFlatScanDS->size() << std::endl;

            // instead of using voxel grid filter, we randomly choose points
//            for(int p=0; p<surfPointsLessFlatScan->size(); p++){
//                if (std::rand() % 4 == 0){
//                    surfPointsLessFlatScanDS->push_back(surfPointsLessFlatScan->points[p]);
//                }
//            }

            // set downSizeFilter depend on the number of surface points
            // if (surfPointsLessFlatScan->size() <= 300){
            //     downSizeFilter.setLeafSize(0.7, 0.2, 0.7);
            //     downSizeFilter.setInputCloud(surfPointsLessFlatScan);
            //     downSizeFilter.filter(*surfPointsLessFlatScanDS);
            // }
            // else{
            //     downSizeFilter.setLeafSize(1.2, 0.5, 1.2);
            //     downSizeFilter.setInputCloud(surfPointsLessFlatScan);
            //     downSizeFilter.filter(*surfPointsLessFlatScanDS);
            // }

//            *surfPointsLessFlat += *surfPointsLessFlatScanDS;
            // output to CvoPointCloud
//            *pc_out_temp += *surfPointsLessFlatScanDS;
            //*pc_out_temp += *surfPointsFlat;

            // if (cornerPointsLessSharp->size() > 0)
            //     pcl::io::savePCDFile("legoloam_cornerPointsLessSharp_final.pcd", *cornerPointsLessSharp);
            // if (cornerPointsSharp->size() > 0)
            //     pcl::io::savePCDFile("legoloam_cornerPointsSharp_final.pcd", *cornerPointsSharp);
            // if (surfPointsFlat->size() > 0)
            //     pcl::io::savePCDFile("legoloam_surfPointsFlat_final.pcd", *surfPointsFlat);
            // if (surfPointsLessFlat->size() > 0)
            //     pcl::io::savePCDFile("legoloam_surfPointsLessFlat_final.pcd", *surfPointsLessFlat);
        }

        // add back the intensity information and selected_indexes
        size_t output_cloud_size = pc_out_temp->size();
        for (int n = 0; n < output_cloud_size; n++) {
            pcl::PointXYZI temp_point;
            temp_point.x = pc_out_temp->points[n].x;
            temp_point.y = pc_out_temp->points[n].y;
            temp_point.z = pc_out_temp->points[n].z;
            // the intensity of pc_out_temp is the index of that point in the input cloud
            int index = pc_out_temp->points[n].intensity;
            temp_point.intensity = laserCloudIn.points[index].intensity;
	    pc_out->push_back(temp_point);
            // std::cout<<"index for point "<<n<<" is "<<round(pc_out_temp->points[n].intensity)<<std::endl;
            selected_indexes.push_back(index);
   
            // check if it is the same
//            std::cout<<"pc_out (x,y,z,i) = "<<temp_point.x<<","<<temp_point.y<<","<<temp_point.z<<","<<temp_point.intensity<<"; selected_indexes (x,y,z,i) = "<<laserCloudIn.points[index].x<<","<<laserCloudIn.points[index].y<<","<<laserCloudIn.points[index].z<<","<<laserCloudIn.points[index].intensity<<")"<<std::endl; 
        }
    }


    // My own fuction
    int LeGoLoamPointSelection::get_quadrant(pcl::PointXYZI point){
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

    

};
