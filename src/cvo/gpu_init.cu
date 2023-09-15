#include "cvo/gpu_init.hpp"
#include "cvo/CvoParams.hpp"
#include <iostream>
namespace cvo {
  
  void gpu_init(int multiframe_is_sorting_inner_product) {
    std::cout<<"========================\nLaunching gpu_init\n";
    if (multiframe_is_sorting_inner_product) {
      cudaError_t err = cudaDeviceSetLimit ( cudaLimitMallocHeapSize, multiframe_is_sorting_inner_product * 1048576 * 8  );
      if (err != cudaSuccess) { 
        fprintf(stderr, "Failed to allocate heap memory %s .\n", cudaGetErrorString(err)); 
        exit(EXIT_FAILURE); 
      } else {
        std::cout<<("gpu init heap size to be ")<<multiframe_is_sorting_inner_product * 1048576 * 8 <<"\n";
      }
      
    }
    std::cout<<"========================\n";
  }
  
}
