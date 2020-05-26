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
#ifndef LOAMSCANREGISTRATION_HPP
#define LOAMSCANREGISTRATION_HPP


#include "Angle.h"
#include "Vector3.h"
#include "CircularBuffer.h"
#include "time_utils.h"
#include "math_utils.h"


namespace cvo
{

// From BasicScanRegistration.h

/** \brief A pair describing the start end end index of a range. */
typedef std::pair<size_t, size_t> IndexRange;

/** Point label options. */
enum PointLabel
{
  CORNER_SHARP = 2,       ///< sharp corner point
  CORNER_LESS_SHARP = 1,  ///< less sharp corner point
  SURFACE_LESS_FLAT = 0,  ///< less flat surface point
  SURFACE_FLAT = -1       ///< flat surface point
};






/** IMU state data. */
typedef struct IMUState
{
  /** The time of the measurement leading to this state (in seconds). */
  Time stamp;

  /** The current roll angle. */
  Angle roll;

  /** The current pitch angle. */
  Angle pitch;

  /** The current yaw angle. */
  Angle yaw;

  /** The accumulated global IMU position in 3D space. */
  Vector3 position;

  /** The accumulated global IMU velocity in 3D space. */
  Vector3 velocity;

  /** The current (local) IMU acceleration in 3D space. */
  Vector3 acceleration;

  /** \brief Interpolate between two IMU states.
  *
  * @param start the first IMUState
  * @param end the second IMUState
  * @param ratio the interpolation ratio
  * @param result the target IMUState for storing the interpolation result
  */
  static void interpolate(const IMUState& start,
    const IMUState& end,
    const float& ratio,
    IMUState& result)
  {
    float invRatio = 1 - ratio;

    result.roll = start.roll.rad() * invRatio + end.roll.rad() * ratio;
    result.pitch = start.pitch.rad() * invRatio + end.pitch.rad() * ratio;
    if (start.yaw.rad() - end.yaw.rad() > M_PI)
    {
      result.yaw = start.yaw.rad() * invRatio + (end.yaw.rad() + 2 * M_PI) * ratio;
    }
    else if (start.yaw.rad() - end.yaw.rad() < -M_PI)
    {
      result.yaw = start.yaw.rad() * invRatio + (end.yaw.rad() - 2 * M_PI) * ratio;
    }
    else
    {
      result.yaw = start.yaw.rad() * invRatio + end.yaw.rad() * ratio;
    }

    result.velocity = start.velocity * invRatio + end.velocity * ratio;
    result.position = start.position * invRatio + end.position * ratio;
  };
} IMUState;



// For a new class!

class LoamScanRegistration{
  public:
    LoamScanRegistration(const float& lowerBound,
                        const float& upperBound, 
                        const uint16_t& nScanRings);
    ~LoamScanRegistration();


    // From BasicScanRegistration
    auto const& imuTransform          () { return _imuTrans             ; }
    auto const& sweepStart            () { return _sweepStart           ; }
    auto const& laserCloud            () { return _laserCloud           ; }
    auto const& cornerPointsSharp     () { return _cornerPointsSharp    ; }
    auto const& cornerPointsLessSharp () { return _cornerPointsLessSharp; }
    auto const& surfacePointsFlat     () { return _surfacePointsFlat    ; }
    auto const& surfacePointsLessFlat () { return _surfacePointsLessFlat; }

    // From MultiScanMapper

    const float& getLowerBound() { return _lowerBound; }
    const float& getUpperBound() { return _upperBound; }
    const uint16_t& getNumberOfScanRings() { return _nScanRings; }

    /** \brief Set mapping parameters.
    *
    * @param lowerBound - the lower vertical bound (degrees)
    * @param upperBound - the upper vertical bound (degrees)
    * @param nScanRings - the number of scan rings
    */
    void set(const float& lowerBound,
            const float& upperBound,
            const uint16_t& nScanRings);

    /** \brief Map the specified vertical point angle to its ring ID.
    *
    * @param angle the vertical point angle (in rad)
    * @return the ring ID
    */
    int getRingForAngle(const float& angle);


    /** \brief Process a new input cloud.
    *
    * @param laserCloudIn the new input cloud to process
    * @param scanTime the scan (message) timestamp
    */
    void process(const pcl::PointCloud<pcl::PointXYZI>& laserCloudIn, pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out, std::vector <float> & edge_or_surface);

  private:
    // from loam_velodyne

    // From BasicScanRegistration
    /** \brief Process a new cloud as a set of scanlines.
    *
    * @param relTime the time relative to the scan time
    */
    void processScanlines(std::vector<pcl::PointCloud<pcl::PointXYZI>> const& laserCloudScans, pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out, std::vector <float> & edge_or_surface);

    /** \brief Update new IMU state. NOTE: MUTATES ARGS! */
    void updateIMUData(Vector3& acc, IMUState& newState);

    /** \brief Project a point to the start of the sweep using corresponding IMU data
    *
    * @param point The point to modify
    * @param relTime The time to project by
    */
    void projectPointToStartOfSweep(pcl::PointXYZI& point, float relTime);

    /** \brief Check is IMU data is available. */
    inline bool hasIMUData() { return _imuHistory.size() > 0; };

    /** \brief Set up the current IMU transformation for the specified relative time.
     *
     * @param relTime the time relative to the scan time
     */
    void setIMUTransformFor(const float& relTime);

    /** \brief Project the given point to the start of the sweep, using the current IMU state and position shift.
     *
     * @param point the point to project
     */
    void transformToStartIMU(pcl::PointXYZI& point);

    /** \brief Prepare for next scan / sweep.
     *
     * @param scanTime the current scan time
     * @param newSweep indicator if a new sweep has started
     */
    void reset();

    /** \brief Extract features from current laser cloud.
     *
     * @param beginIdx the index of the first scan to extract features from
     */
    void extractFeatures(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out, std::vector <float> & edge_or_surface);

    /** \brief Set up region buffers for the specified point range.
     *
     * @param startIdx the region start index
     * @param endIdx the region end index
     */
    void setRegionBuffersFor(const size_t& startIdx,
      const size_t& endIdx);

    /** \brief Set up scan buffers for the specified point range.
     *
     * @param startIdx the scan start index
     * @param endIdx the scan start index
     */
    void setScanBuffersFor(const size_t& startIdx,
      const size_t& endIdx);

    /** \brief Mark a point and its neighbors as picked.
     *
     * This method will mark neighboring points within the curvature region as picked,
     * as long as they remain within close distance to each other.
     *
     * @param cloudIdx the index of the picked point in the full resolution cloud
     * @param scanIdx the index of the picked point relative to the current scan
     */
    void markAsPicked(const size_t& cloudIdx,
      const size_t& scanIdx);

    /** \brief Try to interpolate the IMU state for the given time.
     *
     * @param relTime the time relative to the scan time
     * @param outputState the output state instance
     */
    void interpolateIMUStateFor(const float& relTime, IMUState& outputState);

    void updateIMUTransform();

    // RegistrationParams _config;  ///< registration parameter
    /** Scan Registration configuration parameters. */
    /** The time per scan. */
    float _scanPeriod = 0.1;

    /** The size of the IMU history state buffer. */
    int _imuHistorySize = 200;

    /** The number of (equally sized) regions used to distribute the feature extraction within a scan. */
    int _nFeatureRegions = 6;

    /** The number of surrounding points (+/- region around a point) used to calculate a point curvature. */
    int _curvatureRegion = 5;

    /** The maximum number of sharp corner points per feature region. */
    int _maxCornerSharp = 2;

    /** The maximum number of less sharp corner points per feature region. */
    int _maxCornerLessSharp = 20;

    /** The maximum number of flat surface points per feature region. */
    int _maxSurfaceFlat = 4;

    /** The voxel size used for down sizing the remaining less flat surface points. */
    float _lessFlatFilterSize = 0.2;

    /** The curvature threshold below / above a point is considered a flat / corner point. */
    float _surfaceCurvatureThreshold = 0.1;

    pcl::PointCloud<pcl::PointXYZI> _laserCloud;   ///< full resolution input cloud
    std::vector<IndexRange> _scanIndices;          ///< start and end indices of the individual scans withing the full resolution cloud

    pcl::PointCloud<pcl::PointXYZI> _cornerPointsSharp;      ///< sharp corner points cloud
    pcl::PointCloud<pcl::PointXYZI> _cornerPointsLessSharp;  ///< less sharp corner points cloud
    pcl::PointCloud<pcl::PointXYZI> _surfacePointsFlat;      ///< flat surface points cloud
    pcl::PointCloud<pcl::PointXYZI> _surfacePointsLessFlat;  ///< less flat surface points cloud

    Time _sweepStart;            ///< time stamp of beginning of current sweep
    Time _scanTime;              ///< time stamp of most recent scan
    IMUState _imuStart;                     ///< the interpolated IMU state corresponding to the start time of the currently processed laser scan
    IMUState _imuCur;                       ///< the interpolated IMU state corresponding to the time of the currently processed laser scan point
    Vector3 _imuPositionShift;              ///< position shift between accumulated IMU position and interpolated IMU position
    size_t _imuIdx = 0;                         ///< the current index in the IMU history
    CircularBuffer<IMUState> _imuHistory;   ///< history of IMU states for cloud registration

    pcl::PointCloud<pcl::PointXYZ> _imuTrans = { 4,1 };  ///< IMU transformation information

    std::vector<float> _regionCurvature;      ///< point curvature buffer
    std::vector<PointLabel> _regionLabel;     ///< point label buffer
    std::vector<size_t> _regionSortIndices;   ///< sorted region indices based on point curvature
    std::vector<int> _scanNeighborPicked;     ///< flag if neighboring point was already picked


    // From MultiScanMapper
    float _lowerBound;      ///< the vertical angle of the first scan ring
    float _upperBound;      ///< the vertical angle of the last scan ring
    uint16_t _nScanRings;   ///< number of scan rings
    float _factor;          ///< linear interpolation factor


    // From MultiScanRegistration

    int _systemDelay = 20;             ///< system startup delay counter
    std::vector<pcl::PointCloud<pcl::PointXYZI> > _laserCloudScans;

    int get_quadrant(pcl::PointXYZI point);
    void compute_normal_and_remove_ground(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out, std::vector <float> & edge_or_surface);
};


}

#endif