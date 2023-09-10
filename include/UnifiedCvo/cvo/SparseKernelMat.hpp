#pragma once
namespace cvo {
    
  // this is allocated on GPU
  struct
  /*#ifdef __CUDACC__
  __align__(16)
#else
  alignas(16)
  #endif    */
  SparseKernelMat {
    int rows;
    int cols;
    unsigned int nonzero_sum;
    float * mat;
    int * ind_row2col;
    unsigned int * nonzeros;
    //int * max_index; // for least square implementation
  };


  unsigned int nonzeros(SparseKernelMat * A_host );

  void compute_nonzeros(SparseKernelMat * A_host);

  unsigned int max_neighbors(SparseKernelMat *A_host);

  float A_sum(SparseKernelMat * A_host);
  float A_sum(SparseKernelMat * A_host, int num_neighbors);
  
  void clear_SparseKernelMat(SparseKernelMat * A_host);
  void clear_SparseKernelMat(SparseKernelMat * A_host, int num_neighbors);
  
  SparseKernelMat * init_SparseKernelMat_gpu(int row, int col, SparseKernelMat & A_host);

  void delete_SparseKernelMat_gpu(SparseKernelMat * A_gpu, SparseKernelMat * A_host  );


  // allocate memory for A_cpu's internal
  // assume: the length of A_cpu is larger or equal to A_host
  int copy_internal_SparseKernelMat_gpu_to_cpu(SparseKernelMat * A_host,
                                               SparseKernelMat * A_cpu,
                                               int num_neighbors=-1);

  void clear_SparseKernelMat_cpu(SparseKernelMat * A_cpu, int num_neighbors);


  void init_internal_SparseKernelMat_cpu(int rows,
                                         int cols,
                                         SparseKernelMat * A_cpu 
                                         );
  void delete_internal_SparseKernelMat_cpu(SparseKernelMat * A_cpu );

  
  
  
}
