SET(PROJECT_NAME OUTDOOR_CVO)

PROJECT(${PROJECT_NAME})
cmake_minimum_required(VERSION 3.8)
option(${PROJECT_NAME}_ENABLE_CUDA "ENABLE_CUDA" ON)
option(${PROJECT_NAME}_ENABLE_ACVO "ENABLE_ACVO" OFF)
include(CheckLanguage)
if (${PROJECT_NAME}_ENABLE_CUDA)
  enable_language(CUDA)
  set(CUDACXX nvcc)
  check_language(CUDA)
  set(CUDA_VERBOSE_BUILD ON )
  set(    CUDA_SEPERABLE_COMPILATION ON)
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)
#set(CMAKE_CXX_EXTENSIONS OFF)
#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#set (CMAKE_POSITION_INDEPENDENT_CODE FALSE)
add_compile_options(-std=c++14)

# Release Mode

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -msse  -msse2 -msse3 -msse4 -mavx2 -ftree-vectorize" )
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O2" )

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -lmkl_intel_lp64 -lmkl_core -lmkl -liomp -msse  -msse2 -msse3 -msse4 -mavx2 -ftree-vectorize" )
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -g  -msse -msse2 -msse3 -msse4 -mavx2 -ftree-vectorize" )

#set(BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# required libraries
find_package(SuiteParse REQUIRED)
find_package(Sophus REQUIRED)
message("Sophus dir: ${Sophus_INCLUDE_DIRS}")

# CUDA PACKAGE
find_package(CUDA REQUIRED)
#set(CUDA_SEPARABLE_COMPILATION ON)
#$set(CUDA_PROPAGATE_HOST_FLAGS ON)
#set(CUDA_HOST_COMPILER g++)
#list( APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_75,code=sm_75;")


#set(EIGEN_INCLUDE_DIR /usr/local/include/eigen3)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread timer chrono)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GTSAM REQUIRED)
find_package(OpenMP)
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(OpenMP)
if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        message("OpenMP enabled")
else()
	message("OpenMP not found")
endif()
add_definitions(-DOPENMP)
add_definitions(-DENABLE_SSE)
set(CMAKE_CXX_FLAGS
#  #"${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -fPIC -march=native"
#  "${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -march=native"
"${CMAKE_CXX_FLAGS} ${SSE_FLAGS} "
)

find_package(CUDA)
#SET(CUDA_SEPARABLE_COMPILATION ON)
#set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14; -Xcompiler -fPIC; -rdc=true; -Wno-deprecated-gpu-targets; -lgsl; -lgslcblas; -lm; -O2; -use_fast_math; --extended-lambda; -gencode arch=compute_75,code=sm_75;)
#list(APPEND CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS}  "-Xcompiler -std=c++14  -O3 -rdc=true -Wno-deprecated-gpu-targets -lgsl -lgslcblas -lm -use_fast_math"  )
#<<<<<<< HEAD
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  --extended-lambda -gencode arch=compute_61,code=sm_61 -gencode arch=compute_52,code=sm_52 "  )
#foreach(flag IN LISTS " -rdc=true; -Wno-deprecated-gpu-targets; -lgsl; -lgslcblas; -lm; -use_fast_math; -fPIC"  )
foreach(flag IN LISTS " -rdc=true; -Wno-deprecated-gpu-targets; -lgsl; -lgslcblas; -lm; -fPIC"  )
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>")
endforeach()

get_property(core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS)
string(REPLACE "-fPIC" "" new_core_options ${core_options})
set_property(TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options})
set_property(TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON")

#set (CUDA_NVCC_FLAGS "${CMAKE_CUDA_FLAGS}  --extended-lambda  -gencode arch=compute_75,code=sm_75"  )
#set (CMAKE_CUDA_FLAGS "  --extended-lambda  -gencode arch=compute_75,code=sm_75"  )

# thirdparty elas
set(ELAS_LIB_SOURCE
  ${PROJECT_SOURCE_DIR}/thirdparty/libelas/libelas/descriptor.cpp
  ${PROJECT_SOURCE_DIR}/thirdparty/libelas/libelas/elas.cpp
  ${PROJECT_SOURCE_DIR}/thirdparty/libelas/libelas/filter.cpp
  ${PROJECT_SOURCE_DIR}/thirdparty/libelas/libelas/matrix.cpp
  ${PROJECT_SOURCE_DIR}/thirdparty/libelas/libelas/triangle.cpp
  )

set(CUKDTREE_LIB_SOURCE
  ${PROJECT_SOURCE_DIR}/thirdparty/cugicp/cukdtree/cukdtree.cuh
  )

set(CVO_UTILS_SOURCE
  ${PROJECT_SOURCE_DIR}/src/StaticStereo.cpp
  ${PROJECT_SOURCE_DIR}/src/CvoPixelSelector.cpp
  ${PROJECT_SOURCE_DIR}/src/LidarPointSelector.cpp
  ${PROJECT_SOURCE_DIR}/src/LoamScanRegistration.cpp
  ${PROJECT_SOURCE_DIR}/src/LeGoLoamPointSelection.cpp
  ${PROJECT_SOURCE_DIR}/src/RawImage.cpp
  ${PROJECT_SOURCE_DIR}/src/CvoPointCloud.cpp
  
  )

set(CVO_UTILS_COVARIANCE_SOURCE
  ${PROJECT_SOURCE_DIR}/src/StaticStereo.cpp
  ${PROJECT_SOURCE_DIR}/src/CvoPixelSelector.cpp
  ${PROJECT_SOURCE_DIR}/src/LidarPointSelector.cpp
  ${PROJECT_SOURCE_DIR}/src/LoamScanRegistration.cpp
  ${PROJECT_SOURCE_DIR}/src/LeGoLoamPointSelection.cpp
  ${PROJECT_SOURCE_DIR}/src/RawImage.cpp
  ${PROJECT_SOURCE_DIR}/src/CvoPointCloud.cpp
  ${PROJECT_SOURCE_DIR}/src/utils/CvoPointCovariance.cu
  )


set(BKI_MAPPING_SOURCE
  ${PROJECT_SOURCE_DIR}/src/mapping/point3f.cpp
  ${PROJECT_SOURCE_DIR}/src/mapping/bkiblock.cpp
  ${PROJECT_SOURCE_DIR}/src/mapping/bkioctomap.cpp
  ${PROJECT_SOURCE_DIR}/src/mapping/bkioctree_node.cpp
  ${PROJECT_SOURCE_DIR}/src/mapping/bkioctree.cpp
  )

set(TUM_HANDLER_SOURCE
  ${PROJECT_SOURCE_DIR}/src/TumHandler.cpp  
  )

set(KITTI_HANDLER_SOURCE
  ${PROJECT_SOURCE_DIR}/src/KittiHandler.cpp  
  )

# Sources files
set(POSE_GRAPH_SOURCE
  ${PROJECT_SOURCE_DIR}/src/Frame.cpp
  ${PROJECT_SOURCE_DIR}/src/PoseGraph.cpp
  )

set(LIE_GROUP_SOURCE
  ${PROJECT_SOURCE_DIR}/src/cvo/LieGroup.cpp
  )

set (ACVO_GPU_SOURCE
  ${PROJECT_SOURCE_DIR}/src/cvo/AdaptiveCvoGPU.cu
  ${PROJECT_SOURCE_DIR}/src/cvo/CvoState.cu
  )



set (CVO_GPU_SOURCE
  ${PROJECT_SOURCE_DIR}/src/cvo/CvoGPU.cu
  ${PROJECT_SOURCE_DIR}/src/cvo/CvoState.cu
  )

set (CVO_CPU_SOURCE
  ${PROJECT_SOURCE_DIR}/src/Cvo.cpp
  )


include_directories(
  ${PROJECT_SOURCE_DIR}/include/outdoor_cvo/
  ${EIGEN_INCLUDE_DIR}
  ${TBB_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CSPARSE_INCLUDE_DIR}
  ${CHOLMOD_INCLUDE_DIR}
  ${Sophus_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  # thirdparty
  ${PROJECT_SOURCE_DIR}/thirdparty/libelas/
  ${PROJECT_SOURCE_DIR}/thirdparty/cugicp/
) 

message("CXX_FLAGS are ${CMAKE_CXX_FLAGS}")
message("CUDA_NVCC_FLAGS are ${CUDA_NVCC_FLAGS}")

add_library(elas SHARED ${ELAS_LIB_SOURCE})

add_library(tum SHARED ${TUM_HANDLER_SOURCE})
add_library(kitti SHARED ${KITTI_HANDLER_SOURCE})

if (${PROJECT_NAME}_ENABLE_CUDA)
  add_library(cukdtree_lib SHARED ${CUKDTREE_LIB_SOURCE})
  set_target_properties(cukdtree_lib PROPERTIES LINKER_LANGUAGE CUDA  CUDA_SOURCE_PROPERTY_FORMAT OBJ ) 
endif()  

add_library(lie_group_utils SHARED ${LIE_GROUP_SOURCE})

add_library(cvo_utils_lib SHARED ${CVO_UTILS_SOURCE})
target_link_libraries(cvo_utils_lib opencv_core opencv_photo opencv_highgui opencv_features2d elas opencv_imgcodecs pcl_filters pcl_features pcl_io pcl_search ${TBB_LIBRARIES})

add_library(cvo_utils_range_ell_lib SHARED ${CVO_UTILS_SOURCE})
target_compile_definitions(cvo_utils_range_ell_lib PRIVATE -DIS_USING_RANGE_ELL)
target_link_libraries(cvo_utils_range_ell_lib opencv_core opencv_photo opencv_features2d opencv_highgui elas opencv_imgcodecs pcl_filters pcl_features pcl_io pcl_search ${TBB_LIBRARIES})


if (${PROJECT_NAME}_ENABLE_CUDA)
add_library(cvo_utils_cov_lib SHARED ${CVO_UTILS_COVARIANCE_SOURCE})
target_compile_definitions(cvo_utils_cov_lib PRIVATE -DIS_USING_COVARIANCE)
set_target_properties(cvo_utils_cov_lib PROPERTIES
  LANGUAGE CUDA
  CUDA_SOURCE_PROPERTY_FORMAT OBJ)
target_link_libraries(cvo_utils_cov_lib cukdtree_lib opencv_core opencv_photo opencv_features2d opencv_highgui elas opencv_imgcodecs pcl_filters pcl_features pcl_io pcl_search ${TBB_LIBRARIES})
endif()

add_library(cvo_utils_normal_lib SHARED ${CVO_UTILS_SOURCE})
target_compile_definitions(cvo_utils_normal_lib PRIVATE -DIS_USING_NORMALS )
target_link_libraries(cvo_utils_normal_lib  opencv_core opencv_highgui opencv_photo elas opencv_imgcodecs opencv_features2d pcl_filters pcl_features  pcl_io ${TBB_LIBRARIES})

add_library(cvo_utils_loam_lib SHARED ${CVO_UTILS_SOURCE})
target_compile_definitions(cvo_utils_loam_lib PRIVATE -DIS_USING_LOAM)
target_link_libraries(cvo_utils_loam_lib  opencv_core opencv_highgui opencv_photo elas opencv_imgcodecs pcl_filters pcl_features opencv_features2d pcl_search pcl_io ${TBB_LIBRARIES})

add_library(bki_mapping_lib SHARED ${BKI_MAPPING_SOURCE})
target_link_libraries(bki_mapping_lib cvo_utils_lib pcl_filters pcl_io pcl_search pcl_visualization )

#if (${PROJECT_NAME}_ENABLE_CUDA)
#add_library(bki_mapping_cov_lib SHARED ${BKI_MAPPING_SOURCE})
#target_link_libraries(bki_mapping_lib cvo_utils_cov_lib pcl_filters pcl_io pcl_search pcl_visualization )
#endif()

#############################################
# CVO and Adaptive-CVO cpu
add_library(cvo_cpu_lib SHARED ${CVO_CPU_SOURCE} )
target_link_libraries(cvo_cpu_lib PRIVATE  lie_group_utils ${TBB_LIBRARIES} pcl_io pcl_search pcl_visualization cxsparse)

add_library(cvo_cpu_normal_lib SHARED ${CVO_CPU_SOURCE} )
target_compile_definitions(cvo_cpu_normal_lib PRIVATE -DIS_USING_NORMALS)
target_link_libraries(cvo_cpu_normal_lib PRIVATE  cvo_utils_normal_lib lie_group_utils ${TBB_LIBRARIES} pcl_io pcl_search pcl_visualization cxsparse)

add_library(cvo_cpu_loam_lib SHARED ${CVO_CPU_SOURCE} )
target_compile_definitions(cvo_cpu_loam_lib PRIVATE -DIS_USING_LOAM)
target_link_libraries(cvo_cpu_loam_lib PRIVATE cvo_utils_loam_lib  lie_group_utils ${TBB_LIBRARIES} pcl_io pcl_search pcl_visualization cxsparse)

add_library(cvo_cpu_semantic_lib SHARED ${CVO_CPU_SOURCE} )
target_compile_definitions(cvo_cpu_semantic_lib PRIVATE -DIS_USING_SEMANTICS)
target_link_libraries(cvo_cpu_semantic_lib lie_group_utils ${TBB_LIBRARIES} pcl_io pcl_search pcl_visualization cxsparse)


if (${PROJECT_NAME}_ENABLE_ACVO)
add_library(acvo_cpu_lib SHARED ${ACVO_CPU_SOURCE} )
target_link_libraries(acvo_cpu_lib PRIVATE  lie_group_utils ${TBB_LIBRARIES} pcl_io  cxsparse)

add_library(acvo_cpu_semantic_lib SHARED ${ACVO_CPU_SOURCE} )
target_compile_definitions(acvo_cpu_semantic_lib PRIVATE -DIS_USING_SEMANTICS)
target_link_libraries(acvo_cpu_semantic_lib lie_group_utils ${TBB_LIBRARIES} pcl_io  cxsparse)
endif()

add_library(pose_graph_lib SHARED ${POSE_GRAPH_SOURCE})
target_link_libraries(pose_graph_lib PRIVATE cvo_utils_lib bki_mapping_lib cvo_cpu_lib opencv_core opencv_imgcodecs boost_filesystem  gtsam gtsam_unstable pcl_io pcl_search pcl_visualization)

#add_library(pose_graph_cov_lib SHARED ${POSE_GRAPH_SOURCE})
#target_compile_definitions(pose_graph_cov_lib PRIVATE -DIS_USING_COVARIANCE)
#target_link_libraries(pose_graph_cov_lib PRIVATE cvo_utils_cov_lib bki_mapping_cov_lib cvo_cpu_lib  opencv_core opencv_imgcodecs boost_filesystem  gtsam gtsam_unstable pcl_io pcl_search pcl_visualization)


add_library(pose_graph_normal_lib SHARED ${POSE_GRAPH_SOURCE})
target_link_libraries(pose_graph_normal_lib PRIVATE cvo_utils_normal_lib bki_mapping_lib cvo_cpu_lib opencv_core opencv_imgcodecs boost_filesystem  gtsam gtsam_unstable pcl_io )

add_library(pose_graph_loam_lib SHARED ${POSE_GRAPH_SOURCE})
target_compile_definitions(pose_graph_loam_lib PRIVATE -DIS_USING_LOAM -DIS_USING_NORMALS)
target_link_libraries(pose_graph_loam_lib PRIVATE cvo_utils_loam_lib bki_mapping_lib cvo_cpu_lib opencv_core opencv_imgcodecs boost_filesystem  gtsam gtsam_unstable pcl_io )


#add_executable(cvo_f2f ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_align_f2f.cpp )
#target_link_libraries(cvo_f2f cvo_cpu_lib boost_system pose_graph_lib cvo_utils_lib kitti boost_thread cxsparse  ${OpenCV_LIBS}  ${TBB_LIBRARIES} ${PCL_LIBRARIES})

#add_executable(cvo_align ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_align.cpp )
#target_link_libraries(cvo_align cvo_cpu_lib cvo_utils_lib boost_system boost_thread kitti pose_graph_lib cxsparse  ${OpenCV_LIBS}  ${TBB_LIBRARIES} ${PCL_LIBRARIES})

#add_executable(cvo_align_normal ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_align.cpp )
#target_compile_definitions(cvo_align_normal PRIVATE -DIS_USING_NORMALS)
#target_link_libraries(cvo_align_normal cvo_cpu_normal_lib cvo_utils_normal_lib boost_system kitti boost_thread pose_graph_normal_lib cxsparse  ${OpenCV_LIBS}  ${TBB_LIBRARIES} ${PCL_LIBRARIES})


#add_executable(cvo_align_loam ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_align.cpp )
#target_compile_definitions(cvo_align_loam PRIVATE -DIS_USING_LOAM)
#target_link_libraries(cvo_align_loam cvo_cpu_loam_lib boost_system  cvo_utils_loam_lib boost_thread  kitti pose_graph_loam_lib cxsparse  ${OpenCV_LIBS}  ${TBB_LIBRARIES} ${PCL_LIBRARIES})




#############################################
# Adaptive CVO GPU
if (${PROJECT_NAME}_ENABLE_CUDA AND ${PROJECT_NAME}_ENABLE_ACVO)
add_library(acvo_gpu_lib  ${ACVO_GPU_SOURCE})
set_target_properties(acvo_gpu_lib PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPERABLE_COMPILATION ON 
  COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")
target_compile_definitions(acvo_gpu_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5)
target_link_libraries(acvo_gpu_lib PRIVATE lie_group_utils cvo_utils_lib pcl_filters pcl_io pcl_search pcl_visualization)

add_library(acvo_gpu_semantic_lib ${ACVO_GPU_SOURCE} )
target_compile_definitions(acvo_gpu_semantic_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5 -DIS_USING_SEMANTICS)
target_link_libraries(acvo_gpu_semantic_lib PRIVATE  lie_group_utils cvo_utils_lib)
set_target_properties(acvo_gpu_semantic_lib PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPERABLE_COMPILATION ON
  COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")

add_library(acvo_gpu_lidar_lib ${ACVO_GPU_SOURCE})
target_compile_definitions(acvo_gpu_lidar_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1)
target_link_libraries(acvo_gpu_lidar_lib PRIVATE  lie_group_utils cvo_utils_lib )
set_target_properties(acvo_gpu_lidar_lib PROPERTIES POSITION_INDEPENDENT_CODE ON
  CUDA_SEPERABLE_COMPILATION ON
  COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")


add_library(acvo_gpu_semantic_lidar_lib ${ACVO_GPU_SOURCE})
target_compile_definitions(acvo_gpu_semantic_lidar_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_USING_SEMANTICS)
target_link_libraries(acvo_gpu_semantic_lidar_lib PRIVATE  lie_group_utils cvo_utils_lib)
set_target_properties(acvo_gpu_semantic_lidar_lib PROPERTIES POSITION_INDEPENDENT_CODE ON
  CUDA_SEPERABLE_COMPILATION ON
  COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")


add_library(acvo_gpu_geometric_lib ${ACVO_GPU_SOURCE})
target_compile_definitions(acvo_gpu_geometric_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_GEOMETRIC_ONLY)
target_link_libraries(acvo_gpu_geometric_lib PRIVATE lie_group_utils cvo_utils_lib )
set_target_properties(acvo_gpu_geometric_lib PROPERTIES POSITION_INDEPENDENT_CODE ON
  CUDA_SEPERABLE_COMPILATION ON
  COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")

endif()



#####################################
## Non-Adaptive CVO GPU
if (${PROJECT_NAME}_ENABLE_CUDA)
add_library(cvo_gpu_lib ${CVO_GPU_SOURCE})
target_link_libraries(cvo_gpu_lib PRIVATE lie_group_utils cvo_utils_lib  )
target_compile_definitions(cvo_gpu_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5)
set_target_properties(cvo_gpu_lib PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPERABLE_COMPILATION ON
  COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")


add_library(cvo_gpu_img_intensity_lib ${CVO_GPU_SOURCE})
target_compile_definitions(cvo_gpu_img_intensity_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5)
target_link_libraries(cvo_gpu_img_intensity_lib PRIVATE lie_group_utils cvo_utils_lib)
set_target_properties(cvo_gpu_img_intensity_lib PROPERTIES POSITION_INDEPENDENT_CODE ON
    CUDA_SEPERABLE_COMPILATION ON
    COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")
  


add_library(cvo_gpu_img_geometric_lib ${CVO_GPU_SOURCE})
target_link_libraries(cvo_gpu_img_geometric_lib PRIVATE lie_group_utils cvo_utils_lib  )
target_compile_definitions(cvo_gpu_img_geometric_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5 -DIS_GEOMETRIC_ONLY)
set_target_properties(cvo_gpu_img_geometric_lib PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPERABLE_COMPILATION ON
  COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")

add_library(cvo_gpu_range_ell_lib ${CVO_GPU_SOURCE})
target_link_libraries(cvo_gpu_range_ell_lib PRIVATE lie_group_utils cvo_utils_range_ell_lib  )
target_compile_definitions(cvo_gpu_range_ell_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5 -DIS_USING_RANGE_ELL)
set_target_properties(cvo_gpu_range_ell_lib PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPERABLE_COMPILATION ON
  COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")

#add_library(cvo_gpu_exec ${CVO_LIB_SOURCE})
#target_compile_definitions(cvo_gpu_exec PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5)
#target_link_libraries(cvo_gpu_exec  elas cvo_gpu_lib ${Boost_LIBRARIES} ${TBB_LIBRARIES}  ${Sophus_LIBRARIES} ${PCL_LIBRARIES}  ${OpenCV_LIBS} ${TBB_LIBRARIES} gtsam  gtsam_unstable )

add_library(cvo_gpu_lidar_lib SHARED ${CVO_GPU_SOURCE})
target_compile_definitions(cvo_gpu_lidar_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_GEOMETRIC_ONLY)
target_link_libraries(cvo_gpu_lidar_lib PRIVATE lie_group_utils cvo_utils_lib)
set_target_properties(cvo_gpu_lidar_lib PROPERTIES
  POSITION_INDEPENDENT_CODE ON
    CUDA_SEPERABLE_COMPILATION ON
    COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")

  
#add_library(cvo_gpu_lidar_cov_lib SHARED ${CVO_GPU_SOURCE})
#target_compile_definitions(cvo_gpu_lidar_cov_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_GEOMETRIC_ONLY -DIS_USING_COVARIANCE)
#target_link_libraries(cvo_gpu_lidar_cov_lib PRIVATE lie_group_utils cvo_utils_cov_lib cukdtree_lib)
#set_target_properties(cvo_gpu_lidar_cov_lib PROPERTIES
#  POSITION_INDEPENDENT_CODE ON
#  CUDA_SEPERABLE_COMPILATION ON
#  COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")
  

#add_library(cvo_gpu_lidar_normal_lib SHARED ${CVO_GPU_SOURCE})
##target_compile_definitions(cvo_gpu_lidar_normal_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_GEOMETRIC_ONLY -DIS_USING_NORMALS)
#target_link_libraries(cvo_gpu_lidar_normal_lib PRIVATE lie_group_utils cvo_utils_normal_lib )
#set_target_properties(cvo_gpu_lidar_normal_lib PROPERTIES
#  POSITION_INDEPENDENT_CODE ON
#    CUDA_SEPERABLE_COMPILATION ON
#    COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")

add_library(cvo_gpu_lidar_range_ell_lib SHARED ${CVO_GPU_SOURCE})
target_compile_definitions(cvo_gpu_lidar_range_ell_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_GEOMETRIC_ONLY -DIS_USING_RANGE_ELL)
target_link_libraries(cvo_gpu_lidar_range_ell_lib PRIVATE lie_group_utils cvo_utils_range_ell_lib )
set_target_properties(cvo_gpu_lidar_range_ell_lib PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPERABLE_COMPILATION ON
  COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")
  
  

add_library(cvo_gpu_lidar_loam_lib SHARED ${CVO_GPU_SOURCE})
target_compile_definitions(cvo_gpu_lidar_loam_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_GEOMETRIC_ONLY -DIS_USING_LOAM )
target_link_libraries(cvo_gpu_lidar_loam_lib PRIVATE lie_group_utils cvo_utils_loam_lib )
set_target_properties(cvo_gpu_lidar_loam_lib PROPERTIES
  POSITION_INDEPENDENT_CODE ON
    CUDA_SEPERABLE_COMPILATION ON
    COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")

  

add_library(cvo_gpu_lidar_intensity_lib ${CVO_GPU_SOURCE})
target_compile_definitions(cvo_gpu_lidar_intensity_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_USING_RANGE_ELL)
target_link_libraries(cvo_gpu_lidar_intensity_lib PRIVATE lie_group_utils cvo_utils_lib)
set_target_properties(cvo_gpu_lidar_intensity_lib PROPERTIES POSITION_INDEPENDENT_CODE ON
    CUDA_SEPERABLE_COMPILATION ON
    COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")

add_library(cvo_gpu_lidar_intensity_loam_lib SHARED ${CVO_GPU_SOURCE})
target_compile_definitions(cvo_gpu_lidar_intensity_loam_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_USING_LOAM )
target_link_libraries(cvo_gpu_lidar_intensity_loam_lib PRIVATE lie_group_utils cvo_utils_loam_lib )
set_target_properties(cvo_gpu_lidar_intensity_loam_lib PROPERTIES
  POSITION_INDEPENDENT_CODE ON
    CUDA_SEPERABLE_COMPILATION ON
    COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")
  
add_library(cvo_gpu_lidar_semantic_lib ${CVO_GPU_SOURCE})
target_compile_definitions(cvo_gpu_lidar_semantic_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_USING_SEMANTICS)
target_link_libraries(cvo_gpu_lidar_semantic_lib PRIVATE lie_group_utils cvo_utils_loam_lib)
set_target_properties(cvo_gpu_lidar_semantic_lib PROPERTIES POSITION_INDEPENDENT_CODE ON
    CUDA_SEPERABLE_COMPILATION ON
    COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")

add_library(cvo_gpu_img_semantic_lib ${CVO_GPU_SOURCE})
target_compile_definitions(cvo_gpu_img_semantic_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5 -DIS_USING_SEMANTICS)
target_link_libraries(cvo_gpu_img_semantic_lib PRIVATE lie_group_utils cvo_utils_lib)
set_target_properties(cvo_gpu_img_semantic_lib PROPERTIES POSITION_INDEPENDENT_CODE ON
    CUDA_SEPERABLE_COMPILATION ON
    COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")



add_library(cvo_gpu_lidar_semantic_loam_lib SHARED ${CVO_GPU_SOURCE})
target_compile_definitions(cvo_gpu_lidar_semantic_loam_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_USING_LOAM -DIS_USING_SEMANTICS)
target_link_libraries(cvo_gpu_lidar_semantic_loam_lib PRIVATE lie_group_utils cvo_utils_loam_lib )
set_target_properties(cvo_gpu_lidar_semantic_loam_lib PROPERTIES
  POSITION_INDEPENDENT_CODE ON
    CUDA_SEPERABLE_COMPILATION ON
    COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")
endif()


#######################################
# tests GPU
if (${PROJECT_NAME}_ENABLE_CUDA)
if(${PROJECT_NAME}_ENABLE_ACVO)

#add_executable(acvo_align_gpu ${PROJECT_SOURCE_DIR}/src/experiments/main_acvo_gpu_align.cpp)
#target_compile_definitions(acvo_align_gpu PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5)
#target_link_libraries(acvo_align_gpu acvo_gpu_lib cvo_cpu_lib cvo_utils_lib boost_filesystem kitti boost_system)

#add_executable(acvo_align_semantic_gpu ${PROJECT_SOURCE_DIR}/src/experiments/main_acvo_gpu_align.cpp)
#target_compile_definitions(acvo_align_semantic_gpu PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5 -DIS_USING_SEMANTICS)
#target_link_libraries(acvo_align_semantic_gpu acvo_gpu_semantic_lib  cvo_utils_lib kitti boost_filesystem cvo_cpu_lib boost_system)


#add_executable(acvo_align_gpu_raw_img ${PROJECT_SOURCE_DIR}/src/experiments/main_acvo_gpu_align_raw_image.cpp)
#target_compile_definitions(acvo_align_gpu_raw_img PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5)
#target_link_libraries(acvo_align_gpu_raw_img acvo_gpu_lib cvo_utils_lib pose_graph_lib kitti  boost_filesystem boost_system)

#add_executable(acvo_align_semantic_gpu_raw_img ${PROJECT_SOURCE_DIR}/src/experiments/main_acvo_semantic_gpu_align_raw_image.cpp)
#target_compile_definitions(acvo_align_semantic_gpu_raw_img PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5 -DIS_USING_SEMANTICS)
#target_link_libraries(acvo_align_semantic_gpu_raw_img acvo_gpu_semantic_lib cvo_utils_lib kitti pose_graph_lib  boost_filesystem boost_system)
 
#add_executable(acvo_align_semantic_gpu_raw_lidar ${PROJECT_SOURCE_DIR}/src/experiments/main_semantic_acvo_gpu_lidar_align_raw.cpp)
#target_compile_definitions(acvo_align_semantic_gpu_raw_lidar PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_USING_SEMANTICS)
#target_link_libraries(acvo_align_semantic_gpu_raw_lidar acvo_gpu_semantic_lidar_lib kitti cvo_utils_lib pose_graph_lib  boost_filesystem boost_system)
 

#add_executable(acvo_geometric_gpu_raw_lidar ${PROJECT_SOURCE_DIR}/src/experiments/main_semantic_acvo_gpu_lidar_align_raw.cpp)
#target_compile_definitions(acvo_geometric_gpu_raw_lidar PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_GEOMETRIC_ONLY)
#target_link_libraries(acvo_geometric_gpu_raw_lidar acvo_gpu_geometric_lib kitti cvo_utils_lib pose_graph_lib  boost_filesystem boost_system)

#add_executable(acvo_intensity_gpu_raw_lidar ${PROJECT_SOURCE_DIR}/src/experiments/main_semantic_acvo_gpu_lidar_align_raw.cpp)
#target_compile_definitions(acvo_intensity_gpu_raw_lidar PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 )
#target_link_libraries(acvo_intensity_gpu_raw_lidar acvo_gpu_lidar_lib kitti cvo_utils_lib pose_graph_lib  boost_filesystem boost_system)


add_executable(acvo_semantic_lidar_align_gpu ${PROJECT_SOURCE_DIR}/src/experiments/main_acvo_gpu_lidar_align.cpp)
target_compile_definitions(acvo_semantic_lidar_align_gpu PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_USING_SEMANTICS)
target_link_libraries(acvo_semantic_lidar_align_gpu  acvo_gpu_semantic_lidar_lib cvo_cpu_semantic_lib cvo_utils_lib boost_filesystem boost_system kitti)


endif()

add_executable(cvo_align_gpu_intensity_img ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_align_raw_image.cpp)
target_compile_definitions(cvo_align_gpu_intensity_img PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5)
target_link_libraries(cvo_align_gpu_intensity_img cvo_gpu_img_intensity_lib cvo_utils_lib kitti  boost_filesystem boost_system)

add_executable(cvo_align_gpu_geometric_img ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_align_raw_image.cpp)
target_compile_definitions(cvo_align_gpu_geometric_img PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5 -DIS_GEOMETRIC_ONLY)
target_link_libraries(cvo_align_gpu_geometric_img cvo_gpu_img_geometric_lib cvo_utils_lib kitti  boost_filesystem boost_system)

add_executable(cvo_align_gpu_raw_img_range_ell ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_align_raw_image.cpp)
target_compile_definitions(cvo_align_gpu_raw_img_range_ell PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5 -DIS_USING_RANGE_ELL)
target_link_libraries(cvo_align_gpu_raw_img_range_ell cvo_gpu_range_ell_lib cvo_utils_range_ell_lib kitti  boost_filesystem boost_system)

add_executable(cvo_align_gpu_semantic_img ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_align_semantic_image.cpp)
target_compile_definitions(cvo_align_gpu_semantic_img PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5 -DIS_USING_SEMANTIC)
target_link_libraries(cvo_align_gpu_semantic_img cvo_gpu_img_semantic_lib cvo_utils_lib kitti  boost_filesystem boost_system)


#add_executable(cvo_align_gpu ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_align.cpp)
#target_compile_definitions(cvo_align_gpu PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5)
#target_link_libraries(cvo_align_gpu cvo_gpu_lib cvo_cpu_lib pose_graph_lib kitti cvo_utils_lib boost_filesystem boost_system)

#add_executable(cvo_align_gpu_rgbd ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_align_rgbd.cpp)
#target_compile_definitions(cvo_align_gpu_rgbd PRIVATE -DNUM_CLASSES=1 -DFEATURE_DIMENSIONS=5)
#target_link_libraries(cvo_align_gpu_rgbd cvo_gpu_lib cvo_cpu_lib pose_graph_lib tum cvo_utils_lib boost_filesystem boost_system)


add_executable(cvo_align_gpu_lidar_raw ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_lidar_raw_geometric.cpp)
target_compile_definitions(cvo_align_gpu_lidar_raw PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_GEOMETRIC_ONLY)
target_link_libraries(cvo_align_gpu_lidar_raw cvo_gpu_lidar_lib kitti cvo_utils_lib boost_filesystem boost_system)

#add_executable(cvo_align_gpu_lidar_raw_cov ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_lidar_raw_covariance.cpp)
#target_compile_definitions(cvo_align_gpu_lidar_raw_cov PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_GEOMETRIC_ONLY -DIS_USING_COVARIANCE)
#target_link_libraries(cvo_align_gpu_lidar_raw_cov cvo_gpu_lidar_cov_lib kitti cvo_utils_cov_lib boost_filesystem boost_system pcl_visualization ${VTK_LIBRARIES})


add_executable(cvo_align_gpu_lidar_range_ell ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_lidar_raw_geometric.cpp)
target_compile_definitions(cvo_align_gpu_lidar_range_ell PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_GEOMETRIC_ONLY -DIS_USING_RANGE_ELL)
target_link_libraries(cvo_align_gpu_lidar_range_ell cvo_gpu_lidar_range_ell_lib cvo_utils_range_ell_lib boost_filesystem boost_system kitti)


add_executable(cvo_align_gpu_lidar_loam ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_lidar_raw_geometric.cpp)
target_compile_definitions(cvo_align_gpu_lidar_loam PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_GEOMETRIC_ONLY -DIS_USING_LOAM)
target_link_libraries(cvo_align_gpu_lidar_loam cvo_gpu_lidar_loam_lib cvo_utils_loam_lib boost_filesystem boost_system kitti)


add_executable(cvo_align_gpu_lidar_intensity_raw ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_lidar_raw_intensity.cpp)
target_compile_definitions(cvo_align_gpu_lidar_intensity_raw PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1)
target_link_libraries(cvo_align_gpu_lidar_intensity_raw cvo_gpu_lidar_intensity_lib cvo_cpu_lib pose_graph_lib cvo_utils_lib boost_filesystem boost_system kitti)

add_executable(cvo_align_gpu_lidar_intensity_loam_raw ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_lidar_raw_intensity.cpp)
target_compile_definitions(cvo_align_gpu_lidar_intensity_loam_raw PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_USING_LOAM)
target_link_libraries(cvo_align_gpu_lidar_intensity_loam_raw cvo_gpu_lidar_intensity_loam_lib  cvo_utils_loam_lib boost_filesystem boost_system kitti)


add_executable(gicp_align_two ${PROJECT_SOURCE_DIR}/src/experiments/gicp_align_two.cpp)
target_link_libraries(gicp_align_two ${PCL_LIBRARIES})

add_executable(cvo_single_test ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_single_image_test.cpp)
#target_compile_definitions(cvo_single_test PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5 -DIS_GEOMETRIC_ONLY)
#target_compile_definitions(cvo_single_test PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5)
target_compile_definitions(cvo_single_test PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5 -DIS_USING_SEMANTICS)
target_link_libraries(cvo_single_test cvo_gpu_img_semantic_lib cvo_utils_lib kitti  boost_filesystem boost_system)

#add_executable(cvo_align_gpu_lidar_pcd ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_pcd.cpp)
#target_compile_definitions(cvo_align_gpu_lidar_pcd PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_GEOMETRIC_ONLY -DIS_USING_LOAM)
#target_link_libraries(cvo_align_gpu_lidar_pcd cvo_gpu_lidar_loam_lib pose_graph_loam_lib cvo_utils_loam_lib boost_filesystem boost_system pcl_io kitti)



#add_executable(cvo_align_gpu_lidar_semantic_loam ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_lidar_raw_semantic.cpp)
#target_compile_definitions(cvo_align_gpu_lidar_semantic_loam PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_USING_SEMANTICS)
#target_link_libraries(cvo_align_gpu_lidar_semantic_loam cvo_gpu_lidar_semantic_loam_lib  cvo_utils_loam_lib boost_filesystem boost_system kitti)

#add_executable(acvo_semantic_lidar_align_gpu ${PROJECT_SOURCE_DIR}/src/experiments/main_acvo_gpu_lidar_align.cpp)
#target_compile_definitions(acvo_semantic_lidar_align_gpu PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_USING_SEMANTICS)
#target_link_libraries(acvo_semantic_lidar_align_gpu  acvo_gpu_semantic_lidar_lib cvo_cpu_semantic_lib cvo_utils_lib boost_filesystem boost_system kitti)

add_executable(cvo_evaluate_indicator ${PROJECT_SOURCE_DIR}/src/experiments/main_evaluate_indicator.cpp)
target_compile_definitions(cvo_evaluate_indicator PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5 -DIS_GEOMETRIC_ONLY)
target_link_libraries(cvo_evaluate_indicator cvo_gpu_lib cvo_cpu_lib pose_graph_lib cvo_utils_lib boost_filesystem boost_system)

add_executable(cvo_indicator_in_sequence ${PROJECT_SOURCE_DIR}/src/experiments/main_indicator_in_sequence.cpp)
target_compile_definitions(cvo_indicator_in_sequence PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5 -DIS_GEOMETRIC_ONLY) # -DIS_USING_SEMANTICS
target_link_libraries(cvo_indicator_in_sequence cvo_gpu_lib cvo_cpu_lib cvo_utils_lib boost_filesystem boost_system kitti)

endif()

#add_executable(cvo_local_mapping ${PROJECT_SOURCE_DIR}/src/main_local_mapping.cpp )
#target_link_libraries(cvo_local_mapping cvo boost_system boost_thread cxsparse  ${OpenCV_LIBS}  ${TBB_LIBRARIES} ${PCL_LIBRARIES}  )

#################################################################
# exporting libraries 
#################################################################
export(PACKAGE OUTDOOR_CVO)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set_target_properties(cvo_gpu_lib PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set_target_properties(cvo_cpu_lib PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
#get_property(cvo_include_dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
#get_property(cvo_library_dirs TARGET OUTDOOR_CVO PROPERTY LIBRARY_OUTPUT_DIRECTORY)
#get_property(cvo_libraries TARGET OUTDOOR_CVO PROPERTY LIBRARY_OUTPUT_NAME)

#message("cvo_include_dirs: " ${cvo_include_dirs})
#message("cvo_library_dirs: " ${cvo_library_dirs})
#message("cvo_libraries: " ${cvo_libraries})

# Configure config file for local build tree
configure_file(${PROJECT_SOURCE_DIR}/cmake/cvoConfig.cmake.in
  "${PROJECT_BINARY_DIR}/cvoConfig.cmake" @ONLY)
