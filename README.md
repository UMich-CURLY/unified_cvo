# Unified CVO (Continuous Visual Odometry)

This repository is an implementation for CVO (Continuous Visual Odometry).  It can perform pure geometric point cloud registration, color-based registration, and semantic-based registration. It is tested in KITTI stereo and Tum RGB-D dataset. Details are in the [A New Framework for Registration of Semantic Point Clouds from Stereo and RGB-D Cameras](https://arxiv.org/abs/2012.03683) and [Nonparametric Continuous Sensor Registration](https://arxiv.org/abs/2001.04286). 

Specifically, this repository provides:
* GPU implentation of goemetric, color, and semantic based registration
* CPU and GPU implementation of `cos` function angle computation that measures the overlap of two point clouds
* Soft data association between any two pairs of points in the two point clouds given a guess of their relative pose

And the following modules are under-development:
* Multiframe point cloud registration

Stacked point clouds based on the resulting frame-to-frame trajectory:
![The stacked pointcloud based on CVO's trajectory](https://github.com/UMich-CURLY/unified_cvo/raw/multiframe/results/stacked_pointcloud.png "Stacked Point Cloud after registration")

[Video](https://drive.google.com/file/d/1GA-2eS9ZE28c4t0BafaiTUJT93WHbFvt/view?usp=sharing) on test results of KITTI and TUM:
[![Test results of KITTI and TUM](https://github.com/UMich-CURLY/unified_cvo/raw/multiframe/results/TUM_featureless.png)](https://drive.google.com/file/d/1GA-2eS9ZE28c4t0BafaiTUJT93WHbFvt/view?usp=sharing)

---

### Dependencies
We recommend using this [Dockerfile](https://github.com/UMich-CURLY/docker_images/tree/master/cvo_gpu) to get a prebuilt environment with all the following dependencies. 

*  `cuda 10 or 11`  (already in docker)
*  `gcc9` (already in docker)
*  `SuiteParse` (already in docker)
* `Sophus 1.0.0 release` (already in docker)
* `Eigen 3.3.9` (already in docker)
* `TBB` (already in docker)
* `Boost 1.65` (already in docker)
* `pcl 1.9.1` (already in docker)
* `OpenCV3` or `OpenCV4` (already in docker)
* `Ceres` (already in docker)
* `Openmp` (already in docker)
* `yaml-cpp 0.7.0` (already in docker)

Note: As specified in the above [Dockerfile](https://github.com/UMich-CURLY/docker_images/tree/master/cvo_gpu) , 'pcl-1.9.1' need to be changed and compiled to get it working with cuda. 
* `pcl/io/boost.h`: add `#include <boost/numeric/conversion/cast.hpp>` at the end of the file before `#endif`
* `pcl/point_cloud.h`: Some meet the error 
```
pcl/point_cloud.h:586100 error: template-id ‘getMapping’ used as a declarator
friend boost::shared_ptr& detail::getMapping(pcl::PointCloud &p);
```
Please see [this doc](https://github.com/autowarefoundation/autoware/issues/2094) for reference

### Compile
```
export CC=gcc-9
export CXX=g++-9
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${YOUR_INSTALL_DIR} 
make -j4
```

---

### Demo
#### Example of aligning two point clouds:
###### Input Colored Point Clouds: 

demo_data/source.pcd |  demo_data/target.pcd
--- | ---
![](https://github.com/UMich-CURLY/unified_cvo/raw/multiframe/demo_data/source.png "source.png")  | ![demo_data/target.pcd](https://github.com/UMich-CURLY/unified_cvo/raw/multiframe/demo_data/target.png "target.png")

###### Launch registration: 
`./build/bin/cvo_align_gpu_two_color_pcd  demo_data/source.pcd  demo_data/target.pcd  cvo_params/cvo_outdoor_params.yaml `

###### Results: Stacking two point clouds before and after alignment 

Before registration (`before_align.pcd`) |  After registration (`after_align.pcd`) 
--- | ---
![stacking source.pcd and target.pcd before registration](https://github.com/UMich-CURLY/unified_cvo/raw/multiframe/demo_data/before_align.png "Stacked Point Cloud before registration")  | ![stacking source.pcd and target.pcd after registration](https://github.com/UMich-CURLY/unified_cvo/raw/multiframe/demo_data/after_align.png "Stacked Point Cloud before registration")


#### Frame-to-Frame Registration Demo on Kitti
Make sure the folder of Kitti Stereo sequences contains the `cvo_calib.txt` and the parameter yaml file is specified. Now inside docker container:
* Geometric Registration: `bash scripts/kitti_geometric_stereo.bash`
* Color Registration:     `bash scripts/kitti_intensity_stereo.bash`
* Semantic Registration:  `bash scripts/cvo_semantic_img_oct26_gpu0.bash`


---

### Installation 
If you want to import Unified CVO in your CMAKE project
* Install this library: `make install`
* In your own repository's `CMakeLists.txt`:
 ```
 find_package(UnifiedCvo REQUIRED ) 
 target_link_libraries(${YOUR_LIBRARY_NAME}                                                                                                                                                                              
 PUBLIC                                                                                                                                     ${YOUR_OTHER_LINKED_LIBRARIES}                                                                                             
 UnifiedCvo::cvo_utils_lib
 UnifiedCvo::lie_group_utils
 UnifiedCvo::cvo_gpu_img_lib 
 UnifiedCvo::elas
 UnifiedCvo::tum
 UnifiedCvo::kitti
 ) 
 ```

---

### Tutorial: How to use the library?

The function that aligns the two input point clouds are declared in `include/UnifiedCvo/cvo/CvoGPU.hpp`:
```
int align(/// inputs
          source_pointcloud,
          target_pointcloud,
          init_pose_from_target_frame_to_source_frame,
          /// outputs
          result_pose_from_source_frame_to_target_frame,
          result_data_correspondence,
          total_running_time
        )
```

#### Definitions of the point clouds
We currently support two data structures to represent point clouds. The two data structures could support many types of information. Only information necessary to the user has to be assigned, while the remaining can be initialized as zero. 

1. PCL format: defined in `include/UnifiedCvo/utils/PointSegmentedDistribution.hpp` 
```
  template <unsigned int FEATURE_DIM, unsigned int NUM_CLASS>  
  PointSegmentedDistribution
  {
    PCL_ADD_POINT4D;                      /// x, y, z
    PCL_ADD_RGB;                          /// r, g, b
    float features[FEATURE_DIM];          /// features invariant to transformations, scaled between [0,1]. 
                                          /// It can include rescaled colors, lidar intensities, 
                                          /// image gradients, etc
    int   label;                          /// its semantic label
    float label_distribution[NUM_CLASS];  /// semantic distribution vector, whose sum is 1.0
    float geometric_type[2];              /// (optional) edge: 0; surface: 1
    float normal[3];                      /// (optional) normal vector at this point
    float covariance[9];                  /// (optional) sample covariance at this point
    float cov_eigenvalues[3];             /// (optional) eigenvalues of the covariance matrix
  };
```
`FEATURE_DIM` and `NUM_CLASS` are template arguments and have to be determined at compile time in `CMakeLists.txt`. Values of each field can be assigned like a regular Point object in PCL library. If you don't use some fields, they can be assigned as zero.

2. Our customized point cloud data structure, `include/UnifiedCvo/utils/CvoPointCloud`. It wraps around the same pointwise information like 3D coordinates, invariant features, semantic distributions, etc. Moreover, it provides constructors from stereo images, RGB-D images, lidar point clouds, and PCL format point clouds.

Examples:
```
/// Construct CvoPointCloud by inserting points 
CvoPointCloud pc(FEATURE_DIMENSIONS, NUM_CLASSES);
pc.reserve(num_points, FEATURE_DIMENSIONS, NUM_CLASSES);
for (int i = 0; i < num_points; i++) {
  /// xyz: the 3D coordinates of the points
  /// feature: the invariant features, such as color, image gradients, etc. Its dimension is 
  ///            FEATURE_DIMENSIONS. If you don't use it,
  ///            they can be assigend as zero, i.e. Eigen::VectorXf::Zero(FEATURE_DIMENSION)
  /// semantics: the semantic distribution vector, whose sum is 1. Its dimension is
  ///            NUM_CLASSES. If you don't use it, they can be assigned as zero
  /// geometric_type: A 2-dim vector, deciding whether the point is an edge or a surface. 
  ///            They can be assigned as zero if you don't need this information
  pc.add_point(i, xyz, feature, semantics, geometric_type);
}
 
```


```
/// CvoPointCloud from pcl::PointXYZ, with only geometric information
pcl::PointCloud<pcl::PointXYZ>::Ptr source_pcd(new pcl::PointCloud<pcl::PointXYZ>);
pcl::io::loadPCDFile(source_file, *source_pcd);
std::shared_ptr<cvo::CvoPointCloud> source(new cvo::CvoPointCloud(*source_pcd));
```

```
/// CvoPointCloud from pcl::PointXYZRGB, with both geometric and semantic information
pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_pcd(new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::io::loadPCDFile(source_file, *source_pcd);
std::shared_ptr<cvo::CvoPointCloud> source(new cvo::CvoPointCloud(*source_pcd));
```

```
/// CvoPointCloud from RGB-D camera with color information
  cv::Mat source_rgb, source_dep;
  tum.read_next_rgbd(source_rgb, source_dep);
  std::vector<uint16_t> source_dep_data(source_dep.begin<uint16_t>(), source_dep.end<uint16_t>());
  std::shared_ptr<cvo::ImageRGBD<uint16_t>> source_raw(new cvo::ImageRGBD(source_rgb, source_dep_data));
  std::shared_ptr<cvo::CvoPointCloud> source(new cvo::CvoPointCloud(*source_raw,
                                                                    calib
                                                                    ));
```


```
/// CvoPointCloud from Stereo camera with color and semantic information
  cv::Mat source_left, source_right;
  std::vector<float> semantics_source;
  kitti.read_next_stereo(source_left, source_right, NUM_SEMANTIC_CLASSES, semantics_source);
  std::shared_ptr<cvo::ImageStereo> source_raw(new cvo::ImageStereo(source_left, source_right, NUM_CLASSES, semantics_source));
  std::shared_ptr<cvo::CvoPointCloud> source(new cvo::CvoPointCloud(*source_raw, calib));

```


#### Edit CMakeLists.txt to build the library

Based on the dimension of intensity features and the semantic features, we will need to add compile definitions in `CMakeLists.txt`:
```
target_compile_definitions(cvo_gpu_img_lib PRIVATE -DNUM_CLASSES=${YOUR_FEATURE_DIMENSION} -DFEATURE_DIMENSIONS=${YOUR_SEMANTIC_VECTOR_DIMESNION}) 
```
For example, if you compile this library for a customized stereo point cloud with 5 dimension color channels `(r,g,b, gradient_x, gradient_y)` and 19 semantic classes (a distribution vector of dimension 19):

```
add_library(cvo_gpu_img_lib ${CVO_GPU_SOURCE})                                                               
target_link_libraries(cvo_gpu_img_lib PRIVATE lie_group_utils cvo_utils_lib  )                               
target_compile_definitions(cvo_gpu_img_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5)   # the dimension of the feature/semantics are declared here              
set_target_properties(cvo_gpu_img_lib PROPERTIES                                                               
POSITION_INDEPENDENT_CODE ON                                                                                 
CUDA_SEPERABLE_COMPILATION ON                                                                                 
COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>") 
```

#### Examples on calling the functions:
1. [Example code for aligning two geometric point clouds, without color/semantics](https://github.com/UMich-CURLY/unified_cvo/blob/multiframe/src/experiments/main_cvo_gpu_align_two_pcd.cpp)
```
add_executable(cvo_align_gpu_two_pcd ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_align_two_pcd.cpp)
target_include_directories(cvo_align_gpu_two_pcd PUBLIC
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
      target_link_libraries(cvo_align_gpu_two_pcd cvo_gpu_img_lib cvo_gpu_img_lib cvo_utils_lib boost_filesystem boost_system pcl_io pcl_common)

```

2. [Example code for aligning two color point clouds, without semantics](https://github.com/UMich-CURLY/unified_cvo/blob/multiframe/src/experiments/main_cvo_gpu_align_two_color_pcd.cpp)
```
add_executable(cvo_align_gpu_two_color_pcd ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_align_two_color_pcd.cpp)
target_include_directories(cvo_align_gpu_two_color_pcd PUBLIC
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_link_libraries(cvo_align_gpu_two_color_pcd cvo_gpu_img_lib cvo_gpu_img_lib cvo_utils_lib boost_filesystem boost_system pcl_io pcl_common)

```

3. [Example code for aligning KITTI stereo semantic point clouds](https://github.com/UMich-CURLY/unified_cvo/blob/release/src/experiments/https://github.com/UMich-CURLY/unified_cvo/blob/multiframe/src/experiments/main_cvo_gpu_align_semantic_image.cpp) 

```
add_executable(cvo_align_gpu_semantic_img ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_align_semantic_image.cpp)
target_include_directories(cvo_align_gpu_semantic_img PUBLIC
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_link_libraries(cvo_align_gpu_semantic_img cvo_gpu_img_lib cvo_utils_lib kitti  boost_filesystem boost_system)

```

#### Example calibration file (cvo_calib.txt)     
Calibration files are required for each data sequence. Note that for different sequences, the calibrations could be different. We assume the input images are already rectified. 

* Stereo camera format: `fx fy cx cy  baseline  image_width image_height`. Then you `cvo_calib.txt` file in the sequence's folder should contain  `707.0912 707.0912 601.8873 183.1104 0.54 1226 370`
  
* RGB-D camera format:  `fx fy cx cy  depthscale image_width image_height`. Then you `cvo_calib.txt` file in the sequence's folder should contain `517.3 516.5 318.6 255.3 5000.0 640 480`



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
is_using_range_ell: 0
is_using_kdtree: 0
is_exporting_association: 0
nearest_neighbors_max: 512
multiframe_using_cpu: 0
is_using_geometric_type: 0

```


#### Headers 

Core Library: `include/unified_cvo/cvo/CvoGPU.hpp`. This header file is the main interfacing of using the library. The `align` functions perform the registration. The `function_angle` functions measure the overlap of the two point clouds. 

Customized PCL PointCloud: `include/unified_cvo/utils/PointSegmentedDistribution.hpp`.  This customized point definition takes number of classes and number of intensity channels as template arguments. These two are specified as target compiler definitions in the `CMakeLists.txt`

Point Selector and Cvo PointCloud constructor: `include/unified_cvo/utils/CvoPointCloud.hpp` . Ways of contructing it are available 


---
 
 ### Citations
 If you find this repository useful, please cite 
 ```
 @inproceedings{zhang2021new,
  title={A New Framework for Registration of Semantic Point Clouds from Stereo and {RGB-D} Cameras},
  author={Zhang, Ray and Lin, Tzu-Yuan and Lin, Chien Erh and Parkison, Steven A and Clark, William and Grizzle, Jessy W and Eustice, Ryan M and Ghaffari, Maani},
  booktitle={Proceedings of the IEEE International Conference on Robotics and Automation},
  year={2021},
  pages={12214-12221},
  organization={IEEE},
  doi={10.1109/ICRA48506.2021.9561929}
  }
 ```
 and 
 
```
@article{clark2021nonparametric,
  title={Nonparametric Continuous Sensor Registration},
  author={Clark, William and Ghaffari, Maani and Bloch, Anthony},
  journal={Journal of Machine Learning Research},
  volume={22},
  number={271},
  pages={1--50},
  year={2021}
}
```
