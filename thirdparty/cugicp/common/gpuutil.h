#pragma once

#define cudaSafe(ans) \
  { gpuAssert((ans), __FILE__, __LINE__, true); }

template <typename T>
struct KernelArray {
  T* data;
  int size;

  // constructor allows for implicit conversion
  KernelArray(thrust::device_vector<T>& dVec) {
    data = thrust::raw_pointer_cast(&dVec[0]);
    size = static_cast<int>(dVec.size());
  }

  template <typename allocator>
  KernelArray(thrust::device_vector<T, allocator>& dVec) {
    data = thrust::raw_pointer_cast(&dVec[0]);
    size = static_cast<int>(dVec.size());
  }

  __host__ __device__ KernelArray(T* data_, int size_)
      : data(data_), size(size_) {}
};

inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}
