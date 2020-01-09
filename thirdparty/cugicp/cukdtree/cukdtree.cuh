#pragma once

#define CUDA_STACK 50

#include <curand.h>
#include <curand_kernel.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <algorithm>
#include <chrono>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/swap.h>
#include <thrust/system/system_error.h>

#include "common/gpuutil.h"
#include "cukdtree.h"

namespace perl_registration {

template <typename PointT>
__host__ __device__ __forceinline__ int biggestAxis(const PointT &p) {
  if (p.data[0] > p.data[1] && p.data[0] > p.data[2]) {
    return 0;
  } else if (p.data[1] > p.data[2]) {
    return 1;
  } else {
    return 2;
  }
};

template <typename PointT>
__host__ __device__ PointT PointDifference(const PointT &a, const PointT &b) {
  PointT out(a.x - b.x, a.y - b.y, a.z - b.z);
  return out;
}

template <typename PointT>
__host__ __device__ PointT PointMax(const PointT &a, const PointT &b) {
  PointT out(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
  return out;
}

template <typename PointT>
__host__ __device__ PointT PointMin(const PointT &a, const PointT &b) {
  PointT out(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
  return out;
}

template <typename PointT>
struct MinMaxVec {
  PointT min;
  PointT max;
};

template <typename PointT>
struct MinMaxUnary {
  __host__ __device__ MinMaxVec<PointT> operator()(const PointT &in) {
    MinMaxVec<PointT> v;
    v.min = in;
    v.max = in;
    return v;
  }
};

template <typename PointT>
struct MinMaxBinary {
  __host__ __device__ MinMaxVec<PointT> operator()(const MinMaxVec<PointT> &a,
                                                   const MinMaxVec<PointT> &b) {
    MinMaxVec<PointT> result;
    result.min = PointMin<PointT>(a.min, b.min);
    result.max = PointMax<PointT>(a.max, b.max);
    return result;
  }
};

template <typename PointT>
struct isLess {
  float value;
  int axis;
  __host__ __device__ isLess(const float &f, const int &a) {
    value = f;
    axis = a;
  }

  __host__ __device__ bool operator()(const PointT &a) {
    return a.data[axis] < value;
  }
};

template <typename PointT>
struct isGreater {
  float value;
  int axis;
  __host__ __device__ isGreater(const float &f, const int &a) {
    value = f;
    axis = a;
  }
  __host__ __device__ bool operator()(const PointT &a) {
    return a.data[axis] > value;
  }
};

template <typename PointT>
struct sortByAxis {
  int axis;
  __host__ __device__ sortByAxis(const int &a) { axis = a; }

  __host__ __device__ bool operator()(const PointT &a, const PointT &b) {
    return a.data[axis] < b.data[axis];
  }
};

template <typename PointT>
__device__ __forceinline__ int computeVar(KernelArray<PointT> d_points,
                                          const int &start, const int &stop) {
  float sum[3] = {0};
  float sqSum[3] = {0};
  float var[3] = {0};
  for (int i = start; i <= stop; i++) {
    for (int j = 0; j < 3; j++) {
      sum[j] += d_points.data[i].data[j];
      sqSum[j] += (d_points.data[i].data[j] * d_points.data[i].data[j]);
    }
  }
  int n = stop + 1 - start;
  int maxIdx = -1;
  float max = 0;
  for (int j = 0; j < 3; j++) {
    var[j] = (sqSum[j] - (sum[j] * sum[j]) / n) / (n - 1);
    if (var[j] > max) {
      max = var[j];
      maxIdx = j;
    }
  }
  return maxIdx;
}

template <typename PointT>
__device__ __forceinline__ int partition(KernelArray<PointT> d_points,
                                         const int &start, const int &stop,
                                         const int &pivot, const int &axis) {
  float pivotVal = d_points.data[pivot].data[axis];
  int storeIdx = 0;
  for (int i = start; i <= stop; i++) {
    if (d_points.data[i].data[axis] < pivotVal) {
      thrust::swap(d_points.data[storeIdx + start], d_points.data[i]);
      storeIdx++;
    }
  }
  return start + storeIdx;
}

template <typename PointT>
__device__ __forceinline__ void qselect(KernelArray<PointT> d_points, int start,
                                        int stop, int n, const int &axis) {
  curandState_t state;
  curand_init(0, 0, 0, &state);
  while (true) {
    if (start == stop) return;
    int max = stop - start;
    int pivot = curand(&state) % max;
    thrust::swap(d_points.data[pivot + start], d_points.data[stop]);
    pivot = partition<PointT>(d_points, start, stop - 1, stop, axis);
    thrust::swap(d_points.data[pivot], d_points.data[stop]);

    if (n == pivot) {
      return;
    } else if (n < pivot) {
      stop = pivot - 1;
    } else {
      start = pivot + 1;
    }
  }
}

template <typename PointT>
__global__ void smallNodeStage(KernelArray<PointT> d_points,
                               KernelArray<Results_t> d_results,
                               KernelArray<Results_t> d_results_out,
                               KernelArray<TreeNode> d_tree, int current_index,
                               int n) {
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= n) return;
  int id = current_index + pos;

  if (d_results.data[pos].left) {
    d_tree.data[d_results.data[pos].parrent_id].leftChild = id;
  } else {
    d_tree.data[d_results.data[pos].parrent_id].rightChild = id;
  }

  int start = d_results.data[pos].range.start;
  int stop = d_results.data[pos].range.stop;
  int axis = computeVar<PointT>(d_points, start, stop);
  int median = ((stop + 1.0 - start) / 2.0) + start;

  qselect<PointT>(d_points, start, stop, median, axis);
  // nth_element(thrust::device,
  //            d_points + start,
  //            d_points + median,
  //            d_points + stop + 1,
  //            sortByAxis(axis));

  d_tree.data[id].point = median;
  d_tree.data[id].axis = axis;

  d_results_out.data[2 * pos].parrent_id = id;
  d_results_out.data[2 * pos].left = true;
  if (median != stop) {
    d_results_out.data[2 * pos].range.start = start;
    d_results_out.data[2 * pos].range.stop = median - 1;
  } else {
    d_results_out.data[2 * pos].range.start = -1;
    d_results_out.data[2 * pos].range.stop = -1;
  }

  d_results_out.data[2 * pos + 1].parrent_id = id;
  d_results_out.data[2 * pos + 1].left = false;
  if (stop != median) {
    d_results_out.data[2 * pos + 1].range.start = median + 1;
    d_results_out.data[2 * pos + 1].range.stop = stop;
  } else {
    d_results_out.data[2 * pos + 1].range.start = -1;
    d_results_out.data[2 * pos + 1].range.stop = -1;
  }
}

template <typename PointT>
void largeNodeStage(thrust::device_vector<PointT> &d_points,
                    thrust::host_vector<Results_t> &results,
                    thrust::host_vector<Results_t> &results_new,
                    thrust::host_vector<TreeNode> &tree, int p, int level) {
#pragma omp parallel for
  for (int i = 0; i < p; i++) {
    int treeIdx = pow(2, level) + i - 1;
    int start = results[i].range.start;
    int stop = results[i].range.stop;
    int median = ((stop + 1 - start) / 2.0) + start;

    MinMaxVec<PointT> startCnt;
    startCnt.min.x = FLT_MAX;
    startCnt.min.y = FLT_MAX;
    startCnt.min.z = FLT_MAX;

    startCnt.max.x = FLT_MIN;
    startCnt.max.y = FLT_MIN;
    startCnt.max.z = FLT_MIN;

    MinMaxVec<PointT> result = thrust::transform_reduce(
        thrust::device, d_points.begin() + start, d_points.begin() + stop + 1,
        MinMaxUnary<PointT>(), startCnt, MinMaxBinary<PointT>());

    int axis = biggestAxis<PointT>(PointDifference(result.max, result.min));
    // int axis = computeVar<PointT>(d_points, start, stop);
    int idLeft = pow(2, level + 1) + i * 2 - 1;
    int idRight = pow(2, level + 1) + i * 2;
    tree[treeIdx].point = median;
    tree[treeIdx].axis = axis;
    tree[treeIdx].leftChild = idLeft;
    tree[treeIdx].rightChild = idRight;

    results_new[2 * i].parrent_id = treeIdx;
    results_new[2 * i].left = true;
    results_new[2 * i].range.start = start;
    results_new[2 * i].range.stop = median - 1;

    results_new[2 * i + 1].parrent_id = treeIdx;
    results_new[2 * i + 1].left = false;
    results_new[2 * i + 1].range.start = median + 1;
    results_new[2 * i + 1].range.stop = stop;

    // qselect<PointT>(d_points, start, stop, median, axis);
    thrust::sort(thrust::device, d_points.begin() + start,
                 d_points.begin() + stop + 1, sortByAxis<PointT>(axis));
  }
}

template <typename PointT>
__device__ __inline__ float pDist(const PointT &a, const PointT &b) {
  return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) +
         (a.z - b.z) * (a.z - b.z);
}

template <typename PointT>
__device__ void TreeSearch(KernelArray<PointT> d_points,
                           KernelArray<TreeNode> d_tree,
                           const PointT &query_point,
                           KernelArray<int> nearest_indices, int k) {
  // printf("inside kernel\n");
  cuPQueue pq(k);
  pq.set_value_ptr(nearest_indices.data);
  float max = pq.get_max();
  // int top = 0;

  int visit_stack[CUDA_STACK];
  int visit_pos = 0;

  int back_stack[CUDA_STACK];
  int back_pos = 0;

  visit_stack[visit_pos++] = 0;

  while (visit_pos > 0 || back_pos > 0) {
    while (visit_pos > 0) {
      int current_node = visit_stack[--visit_pos];
      if (current_node < 0) break;

      const int &node_point = d_tree.data[current_node].point;
      const int &axis = d_tree.data[current_node].axis;
      const int &left_node = d_tree.data[current_node].leftChild;
      const int &right_node = d_tree.data[current_node].rightChild;
      float current_dist =
          pDist<PointT>(*(d_points.data + node_point), query_point);

      if (current_dist < max) {
        pq.push(current_dist, d_tree.data[current_node].point);
        max = pq.get_max();
      }

      if (axis < 0) break;

      if (query_point.data[axis] < d_points.data[node_point].data[axis]) {
        visit_stack[visit_pos++] = left_node;
        back_stack[back_pos++] = current_node;
      } else {
        visit_stack[visit_pos++] = right_node;
        back_stack[back_pos++] = current_node;
      }
    }

    if (back_pos > 0) {
      int current_node = back_stack[--back_pos];

      if (current_node > 0) {
        const int &node_point = d_tree.data[current_node].point;
        const int &axis = d_tree.data[current_node].axis;
        const int &left_node = d_tree.data[current_node].leftChild;
        const int &right_node = d_tree.data[current_node].rightChild;
        float axis_dist =
            query_point.data[axis] - d_points.data[node_point].data[axis];
        axis_dist = axis_dist * axis_dist;

        if (axis_dist < max) {
          if (query_point.data[axis] < d_points.data[node_point].data[axis]) {
            visit_stack[visit_pos++] = right_node;
          } else {
            visit_stack[visit_pos++] = left_node;
          }
        }
      }
    }
  }
  return;
}

template <typename PointT>
__global__ void NKSearch(KernelArray<PointT> d_points,
                         KernelArray<TreeNode> d_tree,
                         KernelArray<PointT> queryPoints,
                         KernelArray<int> nearest_indices, int k) {
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= d_points.size) return;

  KernelArray<int> indices(nearest_indices.data + pos * k, k);
  TreeSearch<PointT>(d_points, d_tree, queryPoints.data[pos], indices, k);
  return;
}

template <typename PointT>
void cuKdTree<PointT>::SetInputCloud(cuPointCloudSharedPtr &d_cloud) {
  n_tree_points_ = d_cloud->points.size();
  int log_tree_points = std::floor(std::log2(n_tree_points_));
  int log2n = std::max(1, log_tree_points - stage_pivot);
  int large_n = std::pow(2, log2n);
  d_point_cloud_ = d_cloud;
  input_cloud_set_ = true;

  cudaStream_t *stream = (cudaStream_t *)malloc(sizeof(cudaStream_t));
  cudaStreamCreate(stream);

  // Allocate and fill tree
  d_tree_.resize(n_tree_points_);
  thrust::host_vector<TreeNode> host_tree(n_tree_points_);
  TreeNode tempNode;
  tempNode.point = -1;
  tempNode.axis = -1;
  tempNode.leftChild = -1;
  tempNode.rightChild = -1;
  thrust::fill(thrust::device, d_tree_.begin(), d_tree_.end(), tempNode);
  thrust::fill(thrust::host, host_tree.begin(), host_tree.end(), tempNode);

  // Allocate and fill first result range
  Results_t temp_result;
  temp_result.range.start = 0;
  temp_result.range.stop = n_tree_points_ - 1;
  std::cout << "N tree points: " << n_tree_points_
            << " Start: " << temp_result.range.start
            << " Stop: " << temp_result.range.stop
            << " log tree points: " << log_tree_points << " log2n: " << log2n
            << " large_n: " << large_n << std::endl;
  thrust::device_vector<Results_t> d_results, d_new_results;
  d_results.resize(n_tree_points_);
  d_new_results.resize(n_tree_points_);
  thrust::host_vector<Results_t> host_results, host_new_results;
  host_results.resize(n_tree_points_);
  host_new_results.resize(n_tree_points_);
  thrust::fill_n(thrust::host, host_results.begin(), 1, temp_result);

  // Large node stage, use many threads to sort
  for (int i = 0; i <= log2n; i++) {
    int p = pow(2, i);
    int blocks = ceil((float)p / (float)thread_num);
    if (profile_) {
      t1_ = std::chrono::high_resolution_clock::now();
      std::cout << "P: " << p << " i: " << i << std::endl;
    }
    largeNodeStage<PointT>(d_point_cloud_->points, host_results,
                           host_new_results, host_tree, p, i);
    cudaSafe(cudaDeviceSynchronize());
    thrust::copy(thrust::host, host_new_results.begin(),
                 host_new_results.begin() + 2 * p, host_results.begin());
    if (profile_) {
      t2_ = std::chrono::high_resolution_clock::now();
      std::cout << "large node stage step: "
                << "time "
                << std::chrono::duration_cast<std::chrono::microseconds>(t2_ -
                                                                         t1_)
                       .count()
                << std::endl;
    }
  }
  thrust::copy(host_new_results.begin(), host_new_results.end(),
               d_new_results.begin());
  thrust::copy(host_results.begin(), host_results.end(), d_results.begin());
  thrust::copy(host_tree.begin(), host_tree.end(), d_tree_.begin());

  int current_index = large_n * 4 - 1;
  int n_active = large_n * 2;

  // Small Node stage, each thread sorts their own region
  while (n_active > 0) {
    int blocks = ceil((float)n_active / (float)thread_num);
    if (profile_) {
      t1_ = std::chrono::high_resolution_clock::now();
      std::cout << "Active Nodes: " << n_active << " Blocks " << blocks
                << std::endl;
    }
    smallNodeStage<PointT><<<blocks, thread_num>>>(
        d_point_cloud_->points, d_results, d_new_results, d_tree_,
        current_index, n_active);
    current_index += n_active;
    cudaSafe(cudaDeviceSynchronize());
    // Clean up
    auto last_active = thrust::copy_if(thrust::device, d_new_results.begin(),
                                       d_new_results.begin() + 2 * n_active,
                                       d_results.begin(), isActive());
    n_active = thrust::distance(d_results.begin(), last_active);
    if (profile_) {
      t2_ = std::chrono::high_resolution_clock::now();
      std::cout << "Small Node step: "
                << "time "
                << std::chrono::duration_cast<std::chrono::microseconds>(t2_ -
                                                                         t1_)
                       .count()
                << std::endl;
    }
  }
  free(stream);

  return;
}

template <typename PointT>
int cuKdTree<PointT>::NearestKSearch(const cuPointCloudSharedPtr &d_query_cloud,
                                     int k,
                                     thrust::device_vector<int> &indices) {
  size_t num_indices = d_query_cloud->points.size() * k;
  indices.resize(num_indices);

  int blocks = ceil((float)d_query_cloud->points.size() / (float)thread_num);
  NKSearch<PointT><<<blocks, thread_num>>>(d_point_cloud_->points, d_tree_,
                                           d_query_cloud->points, indices, k);
  cudaSafe(cudaDeviceSynchronize());

  return k;
}

}  // namespace perl_registration
