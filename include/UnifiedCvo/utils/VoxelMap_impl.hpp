#pragma once

#include "utils/VoxelMap.hpp"
#include <random>

namespace cvo {
  template <typename PointType>
  VoxelMap<PointType>::VoxelMap(float voxelSize) : voxelSize_(voxelSize) {
    // std::cout << "Assigned voxelSize_" << voxelSize << std::endl;
    std::srand(3141592);	// want to be deterministic.

  }

  template <typename PointType>
  bool VoxelMap<PointType>::insert_point(PointType* pt) {
    // 1. find coresponding voxel coordinates
    VoxelCoord intCoord = point_to_voxel_center(pt);
    //std::cout<<"insert the point to "<<intCoord.xc<<", "<<intCoord.yc<<", "<<intCoord.zc<<"\n";    
    // 2. insert point to map
    if (vmap_.count(intCoord)) {
      // voxel already exists
      // std::cout << "Existing voxel" << std::endl;
      std::vector<PointType*>& voxPts = vmap_[intCoord].voxPoints;
      // Check if the point already exists
      for (auto it = voxPts.begin(); it != voxPts.end(); it++) {
        // TODO: make sure this equal is correct
        if (*it == pt) 
          return false; 
      }
      // add only if point didn't exist
      vmap_[intCoord].voxPoints.push_back(pt);
    } else {
      // voxel didn't exist, create voxel and add the point
      vmap_[intCoord] = Voxel<PointType>(intCoord.xc, intCoord.yc, intCoord.zc);
      vmap_[intCoord].voxPoints.push_back(pt);
    }
    // std::cout << "=============================================\n";
    return true;
  }

  template <typename PointType>
  bool VoxelMap<PointType>::delete_point(PointType* pt) {
    // 1. convert to integer coord to look up its voxel
    VoxelCoord intCoord = point_to_voxel_center(pt);
    if (!vmap_.count(intCoord))
      return false;
    // 2. remove this point from the voxel
    std::vector<PointType*>& curVoxPts = vmap_[intCoord].voxPoints;
    // iterate through to find the point to remove
    for (auto it = curVoxPts.begin(); it != curVoxPts.end(); it++) {
      if (*it == pt) {
        curVoxPts.erase(it);
        break;
      }
    }
    // if the voxel contains no point after removal, erase the voxel too
    if (curVoxPts.empty()) {
      vmap_.erase(intCoord);
    }
    return true;
  }

  template <typename PointType>
  bool VoxelMap<PointType>::delete_point_BA(PointType* pt, const Voxel<PointType>* voxel) {
    VoxelCoord intCoord{voxel->xc, voxel->yc, voxel->zc};
    if (!vmap_.count(intCoord))
      return false;
    // 2. remove this point from the voxel
    std::vector<PointType*>& curVoxPts = vmap_[intCoord].voxPoints;
    // iterate through to find the point to remove
    for (auto it = curVoxPts.begin(); it != curVoxPts.end(); it++) {
      if (*it == pt) {
        curVoxPts.erase(it);
        break;
      }
    }
    // if the voxel contains no point after removal, erase the voxel too
    if (curVoxPts.empty()) {
      vmap_.erase(intCoord);
    }
    return true;
  }

  template <typename PointType>
  VoxelMap<PointType>::~VoxelMap() {
    //std::cout<<"Voxel map destructed\n";
  }

  template <typename PointType>
  const Voxel<PointType>* VoxelMap<PointType>::query_point(const PointType* pt) const {
    // 1. convert to integer coord to look up its voxel
    VoxelCoord intCoord = point_to_voxel_center(pt);
    if (!vmap_.count(intCoord))
      return nullptr;
    // std::cout << vmap_[intCoord].xc << ", " << vmap_[intCoord].yc << ", " << vmap_[intCoord].zc << std::endl;
    return &vmap_.at(intCoord);
  }

  template <typename PointType>
  const Voxel<PointType>* VoxelMap<PointType>::query_point(float globalX, float globalY, float globalZ) const {
    VoxelCoord intCoord = point_to_voxel_center(globalX, globalY, globalZ);
    //std::cout << "query_point intCoord is "<<intCoord.xc << ", " << intCoord.yc << ", " << intCoord.zc << std::endl;    
    if (!vmap_.count(intCoord))
      return nullptr;

    return &vmap_.at(intCoord);
    
  }  
  /*
  template <typename PointType>
  std::unordered_set<int> VoxelMap<PointType>::voxel_seen_frames(PointType* pt) const {
    std::unordered_set<int> resSet;
    const Voxel<PointType>* curVoxel = query_point(pt);
    if (curVoxel == nullptr)
      return resSet;
    for (const PointType* p : curVoxel->voxPoints) {
      resSet.insert(p->currentID());
    }
    return resSet;
  }
  */
  template <typename PointType>
  size_t VoxelMap<PointType>::size() {
    return vmap_.size();
  }

  template <typename PointType>
  const std::vector<PointType*> VoxelMap<PointType>::sample_points() const {
    std::vector<PointType*> res;
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()

    int total_voxels = vmap_.size();
    
    for (const auto& voxelPair : vmap_) {
      const std::vector<PointType*> voxPts = voxelPair.second.voxPoints;
      
      if (voxPts.empty()) {
        std::cout << "Shouldn't be empty, skipping\n";
        //continue;
      }

      else if (voxPts.size() == 1)
        res.push_back(voxPts[0]);
      else {
        std::uniform_int_distribution<int> uniform_dist(0, voxPts.size()-1);
        res.push_back(voxPts[uniform_dist(gen)]);
      }
    }
    return res;
  }

  template <typename PointType>
  VoxelCoord VoxelMap<PointType>::point_to_voxel_center(const PointType* pt) const {
    //std::cout<<"point world pos is "<<p_wld.transpose();
    // 2. find its corresponding voxel
    std::vector<float> res(3);
    std::vector<float> orig = {pt->x, pt->y, pt->z};
    for (int i = 0; i < 3; i++) {
      // find remainder
      //float rem = fmod(orig[i], voxelSize_);
      //float rem = std::remainder(orig[i], voxelSize_);      
      // float rem = std::remander(orig[i], voxelSize_);
      //if (std::abs(std::abs(rem) - voxelSize_) < 1e-6 ) rem = 0;      
      //int addOne = 0;
      //if (rem >= 0.0)
      //  addOne = rem > (voxelSize_ / 2.0f);
      //else 
      //   addOne = - (std::abs(rem) > (voxelSize_/2.0)); // -(rem < (voxelSize_ / 2.0f));
      //std::cout<<", rem is "<<rem<<", ";
      //res[i] = (int(orig[i] / voxelSize_) + addOne) * voxelSize_;
      res[i] = float(std::lrint(orig[i] / voxelSize_) ) * voxelSize_;
    }
    VoxelCoord resCoord{res[0], res[1], res[2]};
    return resCoord;
  }


  template <typename PointType>
  VoxelCoord VoxelMap<PointType>::point_to_voxel_center(float globalX, float globalY, float globalZ) const {
    // 1. get pt coord in world frame
    Eigen::Vector4f p_wld;
    p_wld << globalX, globalY, globalZ, 1.0;
    //std::cout<<"point world pos is "<<p_wld.transpose();    
    // 2. find its corresponding voxel
    std::vector<float> res(3);
    std::vector<float> orig = {p_wld(0), p_wld(1), p_wld(2)};
    for (int i = 0; i < 3; i++) {
      // find remainder
      //float rem = fmod(orig[i], voxelSize_);
      //float rem = std::remainder(orig[i], voxelSize_);
      //if (std::abs(std::abs(rem) - voxelSize_) < 1e-6 ) rem = 0;
      //float rem = std::remainder()
      //int addOne = 0;
      //if (rem >= 0.0)
      //  addOne = rem > (voxelSize_ / 2.0f);
      //else 
      //  addOne = -(std::fabs(rem) > (voxelSize_ / 2.0f));
      //std::cout<<", rem is "<<rem<<", ";      
      //res[i] = (int(orig[i] / voxelSize_) + addOne) * voxelSize_;
      res[i] = (float)(std::lrint(orig[i] / voxelSize_)) * voxelSize_;
    }
    VoxelCoord resCoord{res[0], res[1], res[2]};
    return resCoord;
  }
  
  //updateCovis debug use
  template <typename PointType>
  void VoxelMap<PointType>::save_voxels_pcd(std::string filename) const {
      pcl::PointCloud<pcl::PointXYZ> pc;
      for (const auto& voxelPair : vmap_) {
          const VoxelCoord& vc = voxelPair.first;
          pcl::PointXYZ p;
          p.x = vc.xc;
          p.y = vc.yc;
          p.z = vc.zc;
          pc.push_back(p);
      }
      pcl::io::savePCDFile(filename, pc);
      std::cout << "Wrote voxel centers to " << filename << std::endl;
  }

}
