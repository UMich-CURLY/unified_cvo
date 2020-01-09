#pragma once
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
namespace cvo {
    
  // this is allocated on GPU
  struct SparseKernelMat {
    int rows;
    int cols;
    float * mat;
    int * ind_row2col;
  };

  // square<T> computes the square of a number f(x) -> x*x
  template <typename T>
  struct larger_than_threshold
  {
    __host__ __device__
    bool operator()(const T& x) const { 
      return fabs(x);
    }
  };
  

  inline  int nonzeros(SparseKernelMat * A ) {
     thrust::device_ptr<float> A_ptr = thrust::device_pointer_cast(A->mat);
     thrust::device_vector<float> v(A_ptr, A_ptr + A->rows * A->cols );
     //return (int)thrust::reduce(v.begin(),v.end(), 0,
     //                           [=] __host__ __device__ (float x, float y) { return fabs(x) > 1e-7 ? 1.0f : 0.0f; });
     thrust::plus<float> binary_add;
     return (int)thrust::transform_reduce(v.begin(), v.end(),
                                          []__host__ __device__(float x) { return fabs(x) > 1e-7 ? 1.0f : 0.0f; },
                                          0,
                                          binary_add);  
  }

  inline void clear_SparseKernelMat(SparseKernelMat * A) {
    cudaMemset(A->mat, 0, A->rows * A->cols * sizeof(float));
    cudaMemset(A->ind_row2col, 0, A->rows * A->cols * sizeof(int));
  }

  inline SparseKernelMat * init_SparseKernelMat_gpu(int row, int col) {
    SparseKernelMat * A;
    cudaMalloc((void **)&A, sizeof(SparseKernelMat));
    cudaMemcpy((void*)&A->rows, &row , sizeof(int), cudaMemcpyHostToDevice  );
    cudaMemcpy((void*)&A->cols, &col , sizeof(int), cudaMemcpyHostToDevice  );

    float * mat;
    int * ind;
    cudaMalloc((void**)&mat, sizeof(float) * row *col);
    cudaMalloc((void**)&mat, sizeof(int)*row*col);
    cudaMemcpy( &A->mat, &mat, sizeof(float*), cudaMemcpyHostToDevice   );
    cudaMemcpy( &A->ind_row2col , &ind , sizeof(int *), cudaMemcpyHostToDevice   );
    
    return A;
  }

  inline void delete_SparseKernelMat_gpu(SparseKernelMat * A ) {
    cudaFree(A->mat);
    cudaFree(A->ind_row2col );
    cudaFree(A);
  }

  
}
