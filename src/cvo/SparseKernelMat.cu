#include "cvo/SparseKernelMat.hpp"  
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>
namespace cvo {
/*
  // square<T> computes the square of a number f(x) -> x*x
  template <typename T>
  struct larger_than_threshold
  {
    __host__ __device__
    bool operator()(const T& x) const { 
      return fabs(x);
    }
  };
  */
  
   unsigned int nonzeros(SparseKernelMat * A_host ) {
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

   void compute_nonzeros(SparseKernelMat * A_host) {
    A_host->nonzero_sum = 0;
    //thrust::device_ptr<unsigned int> A_ptr = thrust::device_pointer_cast(A_host->nonzeros);
    //thrust::device_vector<unsigned int> v(A_ptr, A_ptr + A_host->rows  );
    //A_host->nonzero_sum = thrust::reduce(v.begin(), v.end());    
    std::vector<unsigned int> v(A_host->rows);
    cudaMemcpy(v.data(), A_host->nonzeros, sizeof(unsigned int)*A_host->rows, cudaMemcpyDeviceToHost);
    A_host->nonzero_sum = std::accumulate(v.begin(), v.end(), 0);
    
  }

   unsigned int max_neighbors(SparseKernelMat *A_host) {

    std::vector<unsigned int> v(A_host->rows);
    cudaMemcpy(v.data(), A_host->nonzeros, sizeof(unsigned int)*A_host->rows, cudaMemcpyDeviceToHost);
    return *std::max_element(v.begin(), v.end());

     
     ///thrust::device_ptr<unsigned int> ind_ptr = thrust::device_pointer_cast(A_host->nonzeros);
     //return *thrust::max_element(thrust::device, ind_ptr, ind_ptr + A_host->rows);
     //  thrust::device_vector<unsigned int> v(A_ptr, A_ptr + A_host->rows  );
    
     //return thrust::reduce(v.begin(), v.end(), 0,   thrust::maximum<unsigned int>());
  }

  float A_sum(SparseKernelMat * A_host) {
     thrust::device_ptr<float> A_ptr = thrust::device_pointer_cast(A_host->mat);
     thrust::device_vector<float> v(A_ptr, A_ptr + A_host->rows * A_host->cols );
     return thrust::reduce(v.begin(), v.end());
    
    
  }


   float A_sum(SparseKernelMat * A_host, int num_neighbors) {
     thrust::device_ptr<float> A_ptr = thrust::device_pointer_cast(A_host->mat);
     thrust::device_vector<float> v(A_ptr, A_ptr + A_host->rows * num_neighbors );
     return thrust::reduce(v.begin(), v.end());
    
    
  }

  
   void clear_SparseKernelMat(SparseKernelMat * A_host) {
    //SparseKernelMat A;
    //cudaMemcpy(&A, A_gpu, sizeof(SparseKernelMat), cudaMemcpyDeviceToHost);
    A_host->nonzero_sum = 0;
    cudaMemset(A_host->mat, 0, A_host->rows * A_host->cols * sizeof(float));
    cudaMemset(A_host->ind_row2col, -1, A_host->rows * A_host->cols * sizeof(int));
    cudaMemset(A_host->nonzeros, 0, A_host->rows * sizeof(unsigned int));
    //cudaMemset(A_host->max_index, -1, A_host->rows * sizeof(int));
  }

   void clear_SparseKernelMat(SparseKernelMat * A_host, int num_neighbors) {
    //SparseKernelMat A;
    //cudaMemcpy(&A, A_gpu, sizeof(SparseKernelMat), cudaMemcpyDeviceToHost);
    A_host->nonzero_sum = 0;
    cudaMemset(A_host->mat, 0, A_host->rows * num_neighbors* sizeof(float));
    cudaMemset(A_host->ind_row2col, -1, A_host->rows * num_neighbors * sizeof(int));
    cudaMemset(A_host->nonzeros, 0, A_host->rows * sizeof(unsigned int));
    //cudaMemset(A_host->max_index, -1, A_host->rows * sizeof(int));
  }


  
   SparseKernelMat * init_SparseKernelMat_gpu(int row, int col, SparseKernelMat & A_host) {
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
    //cudaMalloc((void**)&A_host.max_index, sizeof(int)*row);
    //cudaMemcpy( &A->mat, &mat, sizeof(float*), cudaMemcpyHostToDevice   );
    //cudaMemcpy( &A->ind_row2col , &ind , sizeof(int *), cudaMemcpyHostToDevice   );

    cudaMemcpy((void*)A_out, &A_host, sizeof(SparseKernelMat), cudaMemcpyHostToDevice  );
    
    return A_out;
  }

   void delete_SparseKernelMat_gpu(SparseKernelMat * A_gpu, SparseKernelMat * A_host  ) {
    
    cudaFree(A_host->mat);
    cudaFree(A_host->ind_row2col );
    cudaFree(A_host->nonzeros);
    // cudaFree(A_host->max_index);
    cudaFree(A_gpu);
    
  }


  // allocate memory for A_cpu's internal
  // assume: the length of A_cpu is larger or equal to A_host
   int copy_internal_SparseKernelMat_gpu_to_cpu(SparseKernelMat * A_host,
                                                SparseKernelMat * A_cpu,
                                                int num_neighbors) {

    if (A_host->rows != A_cpu->rows ||
         A_host->cols != A_cpu->cols )
      return -1;

    //A_cpu->rows = A_host->rows;
    if (num_neighbors == -1)
      num_neighbors = A_cpu->cols;

    A_cpu->nonzero_sum = A_host->nonzero_sum;

    cudaMemcpy((void*)A_cpu->mat, (void*)A_host->mat, sizeof(float) * num_neighbors * A_host->rows, cudaMemcpyDeviceToHost  );
    cudaMemcpy((void*)A_cpu->ind_row2col, (void*)A_host->ind_row2col, sizeof(int) * num_neighbors * A_host->rows, cudaMemcpyDeviceToHost  );
    cudaMemcpy((void*)A_cpu->nonzeros, (void*)A_host->nonzeros, sizeof(unsigned int) * A_host->rows, cudaMemcpyDeviceToHost );

    return 0;
   
  }

   void clear_SparseKernelMat_cpu(SparseKernelMat * A_cpu, int num_neighbors) {
    //SparseKernelMat A;
    //cudaMemcpy(&A, A_gpu, sizeof(SparseKernelMat), cudaMemcpyDeviceToHost);
    A_cpu->nonzero_sum = 0;
    memset(A_cpu->mat, 0, A_cpu->rows * num_neighbors* sizeof(float));
    memset(A_cpu->ind_row2col, -1, A_cpu->rows * num_neighbors * sizeof(int));
    memset(A_cpu->nonzeros, 0, A_cpu->rows * sizeof(unsigned int));
    //cudaMemset(A_host->max_index, -1, A_host->rows * sizeof(int));
  }
  

   void init_internal_SparseKernelMat_cpu(int rows,
                                          int cols,
                                          SparseKernelMat * A_cpu 
                                          ) {
    A_cpu->rows = rows;
    A_cpu->cols = cols;
    A_cpu->nonzero_sum = 0;
    A_cpu->mat = new float [rows*cols] (); //(0);
    A_cpu->ind_row2col = new int [rows * cols]();
    memset(A_cpu->ind_row2col, -1, rows*cols*sizeof(int));
    A_cpu->nonzeros = new unsigned int[rows]();
    
    
    // cudaFree(A_host->max_index);
    
  }
  

   void delete_internal_SparseKernelMat_cpu(SparseKernelMat * A_cpu ) {
    
    delete [] A_cpu->mat;
    delete [] A_cpu->ind_row2col ;
    delete [] A_cpu->nonzeros;
    // cudaFree(A_host->max_index);
    
  }
  

}

