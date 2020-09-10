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
    unsigned int nonzero_sum;
    float * mat;
    int * ind_row2col;
    unsigned int * nonzeros;
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
  

  inline unsigned int nonzeros(SparseKernelMat * A_host ) {
    /*
      thrust::device_ptr<float> A_ptr = thrust::device_pointer_cast(A_host->mat);
      thrust::device_vector<float> v(A_ptr, A_ptr + A_host->rows * A_host->cols );

      thrust::plus<float> binary_add;
      return (int)thrust::transform_reduce(v.begin(), v.end(),
      []__host__ __device__(float x) { return fabs(x) > 0.0 ? 1.0f : 0.0f; },
      0,
      binary_add);  
    */
    return A_host->nonzero_sum;
  }

  inline void compute_nonzeros(SparseKernelMat * A_host) {
    A_host->nonzero_sum = 0;
    thrust::device_ptr<unsigned int> A_ptr = thrust::device_pointer_cast(A_host->nonzeros);
    thrust::device_vector<unsigned int> v(A_ptr, A_ptr + A_host->rows  );
    A_host->nonzero_sum = thrust::reduce(v.begin(), v.end());
    /*
      thrust::plus<int> binary_add;
      A_host->nonzero_sum =  thrust::transform_reduce(v.begin(), v.end(),
      []__host__ __device__(int x) { return x > 0; },
      0,
      binary_add);  
    */
    
  }

  inline float A_sum(SparseKernelMat * A_host) {
    thrust::device_ptr<float> A_ptr = thrust::device_pointer_cast(A_host->mat);
    thrust::device_vector<float> v(A_ptr, A_ptr + A_host->rows * A_host->cols );
    return thrust::reduce(v.begin(), v.end());
    
    
  }

  inline void clear_SparseKernelMat(SparseKernelMat * A_host) {
    //SparseKernelMat A;
    //cudaMemcpy(&A, A_gpu, sizeof(SparseKernelMat), cudaMemcpyDeviceToHost);
    
    cudaMemset(A_host->mat, 0, A_host->rows * A_host->cols * sizeof(float));
    cudaMemset(A_host->ind_row2col, -1, A_host->rows * A_host->cols * sizeof(int));
    cudaMemset(A_host->nonzeros, 0, A_host->rows * sizeof(unsigned int));
  }

  inline SparseKernelMat * init_SparseKernelMat_gpu(int row, int col, SparseKernelMat & A_host) {
    SparseKernelMat *A_out; //= new SparseKernelMat;
    cudaMalloc((void **)&A_out, sizeof(SparseKernelMat));

    A_host.rows = row;
    A_host.cols = col;
    A_host.nonzero_sum=0;
    //cudaMemcpy((void*)&A->rows, &row , sizeof(int), cudaMemcpyHostToDevice  );
    //cudaMemcpy((void*)&A->cols, &col , sizeof(int), cudaMemcpyHostToDevice  );

    cudaMalloc((void**)&A_host.mat, sizeof(float) * row *col);
    cudaMalloc((void**)&A_host.ind_row2col, sizeof(int)*row*col);
    cudaMalloc((void**)&A_host.nonzeros, sizeof(unsigned int)*row);
    //cudaMemcpy( &A->mat, &mat, sizeof(float*), cudaMemcpyHostToDevice   );
    //cudaMemcpy( &A->ind_row2col , &ind , sizeof(int *), cudaMemcpyHostToDevice   );

    cudaMemcpy((void*)A_out, &A_host, sizeof(SparseKernelMat), cudaMemcpyHostToDevice  );
    
    return A_out;
  }

  inline void delete_SparseKernelMat_gpu(SparseKernelMat * A_gpu, SparseKernelMat * A_host  ) {
    
    cudaFree(A_host->mat);
    cudaFree(A_host->ind_row2col );
    cudaFree(A_host->nonzeros);
    cudaFree(A_gpu);
  }

  
}
