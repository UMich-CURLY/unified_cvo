# Unified CVO (Continuous Visual Odometry)

### Dockerfile to help resolve dependencies
[Docker file for building CVO](https://github.com/UMich-BipedLab/docker_images/tree/master/cvo_gpu)

Follow it to first get a cuda10 environment. 

### Dependencies
*  `cuda10.2` (already in docker)
*  `gcc7` (already in docker)
*  `SuiteParse` (already in docker)
* `Sophus 1.0.0 release` (already in docker)
* `Eigen 3.3.7` (already in docker)
* `TBB` (already in docker)
* `Boost 1.65` (already in docker)
* `pcl 1.9.1` (built from source)
* `OpenCV3` or `OpenCV4` (already in docker)
* `GTSAM` (`default branch`, already in docker)

Note: 'pcl-1.9.1' need to be changed and compiled to get it working with cuda. 
* `pcl/io/boost.h`: add `#include <boost/numeric/conversion/cast.hpp>` at the end of the file before `#endif`
* `pcl/point_cloud.h`: Some meet the error 
```
pcl/point_cloud.h:586100 error: template-id ‘getMapping’ used as a declarator
friend boost::shared_ptr& detail::getMapping(pcl::PointCloud &p);
```
Please see [this dock](https://github.com/autowarefoundation/autoware/issues/2094) for reference

### Compile
```
mkdir build
cd build
cmake ..
make -j
```



### How to use the library?

Compile the CvoGPU library for a customized stereo point cloud with 5 dimension color channels (r,g,b, gradient_x, gradient_y) and 19 semantic classes:

#### CMakeLists.txt

```
add_library(cvo_gpu_img_lib ${CVO_GPU_SOURCE})                                                               
target_link_libraries(cvo_gpu_img_lib PRIVATE lie_group_utils cvo_utils_lib  )                               
target_compile_definitions(cvo_gpu_img_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5)                  
set_target_properties(cvo_gpu_img_lib PROPERTIES                                                               
POSITION_INDEPENDENT_CODE ON                                                                                 
CUDA_SEPERABLE_COMPILATION ON                                                                                 
COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>") 
```

#### Example [experiment code for KITTI stereo](https://github.com/UMich-CURLY/unified_cvo/blob/release/src/experiments/main_cvo_gpu_align_raw_image.cpp) 

```
add_executable(cvo_align_gpu_img ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_align_raw_image.cpp)     
target_compile_definitions(cvo_align_gpu_img PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5)                 
target_link_libraries(cvo_align_gpu_img cvo_gpu_img_lib cvo_utils_lib kitti  boost_filesystem boost_system) 
```

#### Example calibration file (cvo_calib.txt)     fx fy cx cy  baseline  image_width image_height

`707.0912 707.0912 601.8873 183.1104 0.54 1226 370`

#### Example [parameter file for geometry registration](https://github.com/UMich-CURLY/unified_cvo/blob/release/cvo_params/cvo_geometric_params_img_gpu0.yaml): 

```%YAML:1.0                                                                                                                                
---                                                                                                                                   
ell_init_first_frame: 0.95   # The lengthscale for the first frame if initialization is unknow                                                                                                                     
ell_init: 0.25               # Initial Lengthscale                                                                                                                        
ell_min: 0.05                # Minimum Lengthscale                                                                                                                            
ell_max: 0.9                                                                                                                              
dl: 0.0                                                                                                                                 
dl_step: 0.3                                                                                                                              
sigma: 0.1                                                                                                                               
sp_thres: 0.007                                                                                                                             
c: 7.0                                                                                                                                 
d: 7.0                                                                                                                                 
c_ell: 0.05                    # lengthscale for color/intensity if used                                                                                                                          
c_sigma: 1.0                                                                                                                                
s_ell: 0.1                     # lengthscale for semantics if used                                                                                                                            
s_sigma: 1.0                                                                                                                            
MAX_ITER: 10000                # max number of iterations to run in the optimization                                                                                                                          
min_step: 0.000001             # minimum step size                                                                                                                         
max_step: 0.01                 # maximum step size                                                                                                                            
eps: 0.00005                                                                                                                              
eps_2: 0.000012                                                                                                                             
ell_decay_rate: 0.98 #0.98                                                                                                                       
ell_decay_rate_first_frame: 0.99                                                                                                                    
ell_decay_start: 60                                                                                                                           
ell_decay_start_first_frame: 600  #2000                                                                                                                 
indicator_window_size: 50                                                                                                                        
indicator_stable_threshold: 0.001 #0.002                                                                                                                
is_ell_adaptive: 0                                                                                                                           
is_dense_kernel: 0                                                                                                                           
is_full_ip_matrix: 0                                                                                                                          
is_using_geometry: 1            # if geoemtric kernel is computed k(x,z)                                                                                                                       
is_using_intensity: 0           # if color kernel is computed <l_x, l_z>. Enable it if using color info                                                                                                              
is_using_semantics: 0           # if semantic kernel is computed. Enable it if using semantics                                                                                                                        
is_using_range_ell: 0 ```
```

#### Headers 

Core Library: `include/unified_cvo/cvo/CvoGPU.hpp`

Customized PCL PointCloud: `include/unified_cvo/utils/PointSegmentedDistribution.hpp`.  This customized point definition takes number of classes and number of intensity channels as template arguments. These two are specified as target compiler definitions in the CMakeLists.txt

Point Selector and Cvo PointCloud constructor: `include/unified_cvo/utils/CvoPointCloud.hpp` 



