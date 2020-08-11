#pragma once

#include <thrust/device_vector.h>
#include "cupointcloud/cupointcloud.h"

#include <chrono>

#define KDTREE_K_SIZE 20

namespace perl_registration {

struct TreeNode {
  int point;
  int axis;
  int leftChild;
  int rightChild;
};

struct Range_t {
  int start;
  int stop;
};

struct Results_t {
  int parrent_id;
  Range_t range;
  bool left;
};

struct isActive {
  __host__ __device__ bool operator()(const Results_t r) {
    return r.range.start > -1;
  }
};

class cuPQueue {
 private:
  float _m_max;
  int _m_size;
  int _m_top;

  float _m_priorities[KDTREE_K_SIZE];
  int *_m_values;

 public:
  __host__ __device__ cuPQueue() {
    _m_max = -1;
    _m_size = -1;
    _m_top = -1;
  }

  __host__ __device__ cuPQueue(const int &size) {
    _m_max = 2e16;
    _m_size = size;
    _m_top = 0;

    _m_priorities[0] = _m_max;
  }

  __host__ __device__ ~cuPQueue() {}

  __host__ __device__ void push(const float &p, const int &v) {
    int insert = -1;
    for (int it = _m_top; it >= 0; it--) {
      if (p < _m_priorities[it]) {
        if (it + 1 < _m_size) {
          _m_priorities[it + 1] = _m_priorities[it];
          _m_values[it + 1] = _m_values[it];
        }
        insert = it;
      }
    }
    if (insert != -1) {
      _m_priorities[insert] = p;
      _m_values[insert] = v;
      _m_top += 1;
    }
    if (_m_top == _m_size) {
      _m_top--;
      _m_max = _m_priorities[_m_top];
    }
  }

  __host__ __device__ void set_value_ptr(int *val) { _m_values = val; }

  __host__ __device__ float get_max() { return _m_max; }
  __host__ __device__ int *get_values() { return _m_values; }
  __host__ __device__ float *get_priorities() { return _m_priorities; }
  __host__ __device__ int get_top() { return _m_top; };
};

template <typename PointT>
class cuKdTree {
 private:
  typename cuPointCloud<PointT>::SharedPtr d_point_cloud_;
  thrust::device_vector<TreeNode> d_tree_;

  bool profile_;
  bool input_cloud_set_;
  float epsilon_;
  size_t n_tree_points_;

  static constexpr int stage_pivot = 13;
  static constexpr int thread_num = 32;

  std::chrono::time_point<std::chrono::high_resolution_clock> t1_;
  std::chrono::time_point<std::chrono::high_resolution_clock> t2_;

 public:
  typedef PointT PointType;
  typedef typename std::shared_ptr<cuKdTree<PointT>> SharedPtr;
  typedef typename std::shared_ptr<const cuKdTree<PointT>> SharedConstPtr;

  typedef typename cuPointCloud<PointT>::SharedPtr cuPointCloudSharedPtr;

  cuKdTree()
      : epsilon_(std::numeric_limits<float>::epsilon()),
        input_cloud_set_(false),
        profile_(true) {}

  ~cuKdTree() {}

  void SetEpsilon(const float epsilon) { epsilon_ = epsilon; }

  void SetInputCloud(cuPointCloudSharedPtr &d_cloud);
  int NearestKSearch(const cuPointCloudSharedPtr &d_query_points, int k,
                     thrust::device_vector<int> &indices);
  bool IsInputCloudSet() {return input_cloud_set_;}
};

}  // namespace perl_registration

#include "cukdtree.cuh"
