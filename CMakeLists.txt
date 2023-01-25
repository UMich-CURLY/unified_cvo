set(PROJECT_NAME UnifiedCvo)

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)
cmake_policy(SET CMP0060 NEW)

project(${PROJECT_NAME} VERSION 0.1)
set(${PROJECT_NAME}_VERSION 0.1)

cmake_minimum_required(VERSION 3.0)
option(${PROJECT_NAME}_ENABLE_CUDA "ENABLE_CUDA" ON)
option(${PROJECT_NAME}_ENABLE_ACVO "ENABLE_ACVO" OFF)
option(${PROJECT_NAME}_ENABLE_LIDAR "ENABLE_LIDAR" ON)
option(${PROJECT_NAME}_ENABLE_QT_VIEWER "ENABLE_QT_VIEWER" ON)
option(${PROJECT_NAME}_ENABLE_CUKDTREE "ENABLE_CUKDTREE" OFF)
include(CheckLanguage)
if (${PROJECT_NAME}_ENABLE_CUDA)
  enable_language(CUDA)
  set(CUDACXX nvcc)
  check_language(CUDA)
  set(CUDA_VERBOSE_BUILD ON )
  set(    CUDA_SEPERABLE_COMPILATION ON)
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

# Release Mode
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -msse  -msse2 -msse3 -msse4 -mavx2 -ftree-vectorize" )
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -lmkl_intel_lp64 -lmkl_core -lmkl -liomp -msse  -msse2 -msse3 -msse4 -mavx2 -ftree-vectorize" )
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -g  -msse -msse2 -msse3 -msse4 -mavx2 -ftree-vectorize" )
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}   -march=native" )
#set(BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

## build output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY lib)
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# required libraries
find_package(SuiteParse REQUIRED)
find_package(Sophus REQUIRED)
message("Sophus dir: ${Sophus_INCLUDE_DIRS}")

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread timer chrono filesystem)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(yaml-cpp)
#find_package(VTK REQUIRED)
#include(${VTK_USE_FILE})

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	message("OpenMP Found")
else()
	message("OpenMP not Found")
endif()
add_definitions(-DOPENMP)
#add_definitions(-DENABLE_SSE)
#set(CMAKE_CXX_FLAGS
#  #"${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -fPIC -march=native"
#  "${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -march=native"
#"${CMAKE_CXX_FLAGS} ${SSE_FLAGS} "
#)

find_package(CUDA)
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  --expt-extended-lambda  -gencode arch=compute_61,code=sm_61  -gencode arch=compute_52,code=sm_52 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 "  )
#set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  --expt-extended-lambda  -gencode arch=compute_86,code=sm_86 "  )
foreach(flag IN LISTS " -use_fast_math -rdc=true; -Wno-deprecated-gpu-targets; -lgsl; -lgslcblas; -lm; -fPIC"  )
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>")
endforeach()

get_property(core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS)
string(REPLACE "-fPIC" "" new_core_options ${core_options})
set_property(TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options})
set_property(TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON")

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

set(QT_VIEWER_SOURCE
  ${PROJECT_SOURCE_DIR}/thirdparty/cugicp/viewer/color_handler.cc
  ${PROJECT_SOURCE_DIR}/thirdparty/cugicp/viewer/viewer.cc
  )

set(CVO_UTILS_SOURCE
  ${PROJECT_SOURCE_DIR}/src/utils/StaticStereo.cpp
  ${PROJECT_SOURCE_DIR}/src/utils/CvoPixelSelector.cpp
  ${PROJECT_SOURCE_DIR}/src/utils/LidarPointSelector.cpp
  ${PROJECT_SOURCE_DIR}/src/utils/LoamScanRegistration.cpp
  ${PROJECT_SOURCE_DIR}/src/utils/LeGoLoamPointSelection.cpp
  ${PROJECT_SOURCE_DIR}/src/utils/RawImage.cpp
  ${PROJECT_SOURCE_DIR}/src/utils/ImageStereo.cpp
  )

set(CVO_UTILS_COVARIANCE_SOURCE
  ${PROJECT_SOURCE_DIR}/src/utils/StaticStereo.cpp
  ${PROJECT_SOURCE_DIR}/src/utils/CvoPixelSelector.cpp
  ${PROJECT_SOURCE_DIR}/src/utils/LidarPointSelector.cpp
  ${PROJECT_SOURCE_DIR}/src/utils/LoamScanRegistration.cpp
  ${PROJECT_SOURCE_DIR}/src/utils/LeGoLoamPointSelection.cpp
  ${PROJECT_SOURCE_DIR}/src/utils/RawImage.cpp
  ${PROJECT_SOURCE_DIR}/src/utils/VoxelMap.cpp  
  ${PROJECT_SOURCE_DIR}/src/utils/CvoPointCloud.cpp
  ${PROJECT_SOURCE_DIR}/src/utils/CvoPointCovariance.cu
  ${PROJECT_SOURCE_DIR}/src/utils/ImageStereo.cpp  
  )


set(BKI_MAPPING_SOURCE
  ${PROJECT_SOURCE_DIR}/src/mapping/point3f.cpp
  ${PROJECT_SOURCE_DIR}/src/mapping/bkiblock.cpp
  ${PROJECT_SOURCE_DIR}/src/mapping/bkioctomap.cpp
  ${PROJECT_SOURCE_DIR}/src/mapping/bkioctree_node.cpp
  ${PROJECT_SOURCE_DIR}/src/mapping/bkioctree.cpp
  )

set(TUM_HANDLER_SOURCE
  ${PROJECT_SOURCE_DIR}/src/dataset_handler/TumHandler.cpp  
  )

set(KITTI_HANDLER_SOURCE
  ${PROJECT_SOURCE_DIR}/src/dataset_handler/KittiHandler.cpp  
  )

set(CNPY_LIB_SOURCE
  ${PROJECT_SOURCE_DIR}/thirdparty/cnpy/cnpy.cpp
  )

set(TARTAN_HANDLER_SOURCE
  ${PROJECT_SOURCE_DIR}/src/dataset_handler/TartanAirHandler.cpp
  )

# Sources files
#set(POSE_GRAPH_SOURCE
#  ${PROJECT_SOURCE_DIR}/src/graph_optimizer/Frame.cpp
#  ${PROJECT_SOURCE_DIR}/src/graph_optimizer/PoseGraph.cpp
#  )

set(LIE_GROUP_SOURCE
  ${PROJECT_SOURCE_DIR}/src/cvo/LieGroup.cpp
  )

set (ACVO_GPU_SOURCE
  ${PROJECT_SOURCE_DIR}/src/cvo/AdaptiveCvoGPU.cu
  ${PROJECT_SOURCE_DIR}/src/cvo/CvoState.cu
  )

set (ACVO_CPU_SOURCE
  ${PROJECT_SOURCE_DIR}/src/cvo/ACvo.cpp
  )

set (CVO_GPU_SOURCE
  ${PROJECT_SOURCE_DIR}/src/cvo/CvoGPU.cu
  ${PROJECT_SOURCE_DIR}/src/cvo/CvoGPU_impl.cu
  ${PROJECT_SOURCE_DIR}/src/cvo/CvoState.cu
  ${PROJECT_SOURCE_DIR}/src/cvo/CvoGPU.cpp
  ${PROJECT_SOURCE_DIR}/src/cvo/SparseKernelMat.cu
  ${PROJECT_SOURCE_DIR}/src/cvo/IRLS_State_CPU.cpp
  ${PROJECT_SOURCE_DIR}/src/cvo/IRLS_State_GPU.cu
  ${PROJECT_SOURCE_DIR}/src/cvo/CvoFrameGPU.cu
  ${PROJECT_SOURCE_DIR}/src/cvo/CvoFrame.cpp
  ${PROJECT_SOURCE_DIR}/src/cvo/IRLS.cpp
  ${PROJECT_SOURCE_DIR}/src/cvo/IRLS_State_GPU.cpp
  ${PROJECT_SOURCE_DIR}/src/utils/CvoPoint.cpp
  ${PROJECT_SOURCE_DIR}/src/utils/VoxelMap_CvoPoint.cpp
  ${PROJECT_SOURCE_DIR}/src/utils/VoxelMap_pcl.cpp
  ${PROJECT_SOURCE_DIR}/src/utils/CvoPointCloud.cpp  
  )

set (CVO_CPU_SOURCE
  ${PROJECT_SOURCE_DIR}/src/cvo/Cvo.cpp
  )

set (CVO_IRLS_SOURCE
  ${PROJECT_SOURCE_DIR}/src/cvo/IRLS_State_CPU.cpp
  ${PROJECT_SOURCE_DIR}/src/cvo/IRLS_State_GPU.cu
  ${PROJECT_SOURCE_DIR}/src/cvo/CvoFrameGPU.cu
  ${PROJECT_SOURCE_DIR}/src/cvo/CvoFrame.cpp
  ${PROJECT_SOURCE_DIR}/src/cvo/IRLS.cpp)

set (CVO_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/
  ${EIGEN_INCLUDE_DIR}
  ${TBB_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CSPARSE_INCLUDE_DIR}
  ${CHOLMOD_INCLUDE_DIR}
  ${Sophus_INCLUDE_DIRS}
  #${GTSAM_INCLUDE_DIR}
  # thirdparty
  ${PROJECT_SOURCE_DIR}/thirdparty/libelas/
  ${PROJECT_SOURCE_DIR}/thirdparty/cugicp/
  ${PROJECT_SOURCE_DIR}/thirdparty/cnpy/
  )

set (CVO_PUBLIC_HEADERS
  ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/
  ${PROJECT_SOURCE_DIR}/thirdparty/libelas/
  ${PROJECT_SOURCE_DIR}/thirdparty/cugicp/  
  )

message("CXX_FLAGS are ${CMAKE_CXX_FLAGS}")
message("CUDA_NVCC_FLAGS are ${CUDA_NVCC_FLAGS}")

add_library(elas SHARED ${ELAS_LIB_SOURCE})
target_include_directories(elas PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	"$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
set_target_properties(elas PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

add_library(tum SHARED ${TUM_HANDLER_SOURCE})
target_compile_definitions(tum PUBLIC -DCVO_NDEBUG)  
target_include_directories(tum PUBLIC
  #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}>
  "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"
  #$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include> 
  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )

add_library(kitti SHARED ${KITTI_HANDLER_SOURCE})
target_include_directories(kitti PUBLIC
  #$<BUILD_INTERFACE:${CMAE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}>
  "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"
  #$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include> 
  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_link_libraries(kitti PUBLIC opencv_core opencv_photo opencv_highgui opencv_features2d opencv_imgcodecs pcl_filters pcl_features pcl_io pcl_search pcl_common boost_filesystem
  )

add_library(cnpy SHARED ${CNPY_LIB_SOURCE})
target_include_directories(cnpy PUBLIC
  #$<BUILD_INTERFACE:${CMAE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}>
  "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"
  #$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include> 
  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_link_libraries(cnpy PUBLIC )

add_library(tartan SHARED ${TARTAN_HANDLER_SOURCE})
target_include_directories(tartan PUBLIC
  "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"
  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_link_libraries(tartan PUBLIC opencv_core cnpy opencv_photo opencv_highgui opencv_features2d opencv_imgcodecs pcl_filters pcl_features pcl_io pcl_search pcl_common boost_system boost_filesystem)

if (${PROJECT_NAME}_ENABLE_CUDA)
  add_library(cukdtree_lib SHARED ${CUKDTREE_LIB_SOURCE})
  target_compile_definitions(cukdtree_lib PUBLIC -DCVO_POINT_NEIGHBORS=256)  
  target_include_directories(cukdtree_lib PUBLIC
    #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
    #$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
    "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
  set_target_properties(cukdtree_lib PROPERTIES LINKER_LANGUAGE CUDA  CUDA_SOURCE_PROPERTY_FORMAT OBJ )
endif()  

if (${PROJECT_NAME}_ENABLE_QT_VIEWER)
  add_library(pcd_qt_viewer_lib SHARED ${QT_VIEWER_SOURCE})
  target_include_directories(pcd_qt_viewer_lib PUBLIC
    #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
    #$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
    "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
  target_link_libraries(pcd_qt_viewer_lib PUBLIC pcl_common pcl_visualization)

endif()

add_library(lie_group_utils SHARED ${LIE_GROUP_SOURCE})
target_include_directories(lie_group_utils PUBLIC
  #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}>
  "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"
  #$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include> 
  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${UnifiedCvo_VERSION}> )

      
add_library(cvo_utils_lib SHARED ${CVO_UTILS_SOURCE})
set_target_properties(cvo_utils_lib PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
#add_dependencies(cvo_utils_lib elas)
#target_compile_definitions(cvo_utils_lib PUBLIC ${PCL_DEFINITIONS})   
target_include_directories(cvo_utils_lib PUBLIC
  #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/UnifiedCvo>
  "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"  
  #$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include> 
  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_link_libraries(cvo_utils_lib
  PUBLIC opencv_core opencv_photo opencv_highgui opencv_features2d opencv_imgcodecs pcl_filters pcl_features pcl_io pcl_search pcl_common ${TBB_LIBRARIES}
  PRIVATE
  elas)


#add_library(cvo_utils_range_ell_lib SHARED ${CVO_UTILS_SOURCE})
#target_compile_definitions(cvo_utils_range_ell_lib PRIVATE -DIS_USING_RANGE_ELL)
#target_link_libraries(cvo_utils_range_ell_lib opencv_core opencv_photo opencv_features2d opencv_highgui elas opencv_imgcodecs pcl_filters pcl_features pcl_io pcl_search ${TBB_LIBRARIES})


add_library(bki_mapping_lib SHARED ${BKI_MAPPING_SOURCE})
target_include_directories(bki_mapping_lib PUBLIC
  #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
  #$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
  "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"  
  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_link_libraries(bki_mapping_lib cvo_utils_lib cvo_gpu_img_lib pcl_filters pcl_io pcl_search pcl_visualization )

#if (${PROJECT_NAME}_ENABLE_CUDA)
#add_library(bki_mapping_cov_lib SHARED ${BKI_MAPPING_SOURCE})
#target_link_libraries(bki_mapping_lib cvo_utils_cov_lib pcl_filters pcl_io pcl_search pcl_visualization )
#endif()

#############################################
# CVO and Adaptive-CVO cpu
add_library(cvo_cpu_lib SHARED ${CVO_CPU_SOURCE} )
target_include_directories(cvo_cpu_lib PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
	"$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"        
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
      target_link_libraries(cvo_cpu_lib PRIVATE  lie_group_utils ${TBB_LIBRARIES} pcl_io pcl_search pcl_visualization cxsparse)

#add_library(cvo_irls_lib SHARED ${CVO_IRLS_SOURCE} )
#target_include_directories(cvo_irls_lib PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
#	"$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"        
#        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
#target_compile_definitions(cvo_irls_lib PUBLIC -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5 -DCUDA_BLOCK_SIZE=512 -DCVO_POINT_NEIGHBORS=256 ${PCL_DEFINITIONS})      
#target_link_libraries(cvo_irls_lib PRIVATE  lie_group_utils ${TBB_LIBRARIES} pcl_io pcl_search pcl_visualization cxsparse Ceres::ceres yaml-cpp)
      
    

if (${PROJECT_NAME}_ENABLE_LIDAR)
#add_library(cvo_cpu_loam_lib SHARED ${CVO_CPU_SOURCE} )
#target_compile_definitions(cvo_cpu_loam_lib PRIVATE -DIS_USING_LOAM)
#target_link_libraries(cvo_cpu_loam_lib PRIVATE cvo_utils_lib  lie_group_utils ${TBB_LIBRARIES} pcl_io pcl_search pcl_visualization cxsparse)
endif()
#add_library(cvo_cpu_semantic_lib SHARED ${CVO_CPU_SOURCE} )
#target_compile_definitions(cvo_cpu_semantic_lib PRIVATE -DIS_USING_SEMANTICS)
#target_link_libraries(cvo_cpu_semantic_lib lie_group_utils ${TBB_LIBRARIES} pcl_io pcl_search pcl_visualization cxsparse)


if (${PROJECT_NAME}_ENABLE_ACVO)
  add_library(acvo_cpu_lib SHARED ${ACVO_CPU_SOURCE} )
  target_include_directories(acvo_cpu_lib PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_link_libraries(acvo_cpu_lib PRIVATE  lie_group_utils ${TBB_LIBRARIES} pcl_io  cxsparse)

add_library(acvo_cpu_semantic_lib SHARED ${ACVO_CPU_SOURCE} )
target_include_directories(acvo_cpu_semantic_lib PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
      target_compile_definitions(acvo_cpu_semantic_lib PRIVATE -DIS_USING_SEMANTICS)      
target_link_libraries(acvo_cpu_semantic_lib lie_group_utils ${TBB_LIBRARIES} pcl_io  cxsparse)
endif()

#add_library(pose_graph_lib SHARED ${POSE_GRAPH_SOURCE})
#target_link_libraries(pose_graph_lib PRIVATE cvo_utils_lib bki_mapping_lib cvo_cpu_lib opencv_core opencv_imgcodecs boost_filesystem  gtsam gtsam_unstable pcl_io pcl_search pcl_visualization)

#add_library(pose_graph_cov_lib SHARED ${POSE_GRAPH_SOURCE})
#target_compile_definitions(pose_graph_cov_lib PRIVATE -DIS_USING_COVARIANCE)
#target_link_libraries(pose_graph_cov_lib PRIVATE cvo_utils_cov_lib bki_mapping_cov_lib cvo_cpu_lib  opencv_core opencv_imgcodecs boost_filesystem  gtsam gtsam_unstable pcl_io pcl_search pcl_visualization)


#add_library(pose_graph_normal_lib SHARED ${POSE_GRAPH_SOURCE})
#target_link_libraries(pose_graph_normal_lib PRIVATE cvo_utils_normal_lib bki_mapping_lib cvo_cpu_lib open#cv_core opencv_imgcodecs boost_filesystem  gtsam gtsam_unstable pcl_io )
if (${PROJECT_NAME}_ENABLE_LIDAR)
#add_library(pose_graph_loam_lib SHARED ${POSE_GRAPH_SOURCE})
#target_compile_definitions(pose_graph_loam_lib PRIVATE -DIS_USING_LOAM)
#target_link_libraries(pose_graph_loam_lib PRIVATE cvo_utils_lib bki_mapping_lib cvo_cpu_lib opencv_core opencv_imgcodecs boost_filesystem  gtsam gtsam_unstable pcl_io )
endif()

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
  target_include_directories(acvo_gpu_lib PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
	"$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"        
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
set_target_properties(acvo_gpu_lib PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPERABLE_COMPILATION ON 
  COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")
target_compile_definitions(acvo_gpu_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5)
target_link_libraries(acvo_gpu_lib PRIVATE lie_group_utils cvo_utils_lib pcl_filters pcl_io pcl_search pcl_visualization)

add_library(acvo_gpu_semantic_lib ${ACVO_GPU_SOURCE} )
target_include_directories(acvo_gpu_semantic_lib PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_compile_definitions(acvo_gpu_semantic_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5 -DIS_USING_SEMANTICS)
target_link_libraries(acvo_gpu_semantic_lib PRIVATE  lie_group_utils cvo_utils_lib)
set_target_properties(acvo_gpu_semantic_lib PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPERABLE_COMPILATION ON
  COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")
if (${PROJECT_NAME}_ENABLE_LIDAR)
  add_library(acvo_gpu_lidar_lib ${ACVO_GPU_SOURCE})
  target_include_directories(acvo_gpu_lidar_lib PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
	"$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"        
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_compile_definitions(acvo_gpu_lidar_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1)
target_link_libraries(acvo_gpu_lidar_lib PRIVATE  lie_group_utils cvo_utils_lib )
set_target_properties(acvo_gpu_lidar_lib PROPERTIES POSITION_INDEPENDENT_CODE ON
  CUDA_SEPERABLE_COMPILATION ON
  COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")


add_library(acvo_gpu_semantic_lidar_lib ${ACVO_GPU_SOURCE})
target_include_directories(acvo_gpu_semantic_lidar_lib PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
	"$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"        
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_compile_definitions(acvo_gpu_semantic_lidar_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_USING_SEMANTICS)
target_link_libraries(acvo_gpu_semantic_lidar_lib PRIVATE  lie_group_utils cvo_utils_lib)
set_target_properties(acvo_gpu_semantic_lidar_lib PROPERTIES POSITION_INDEPENDENT_CODE ON
  CUDA_SEPERABLE_COMPILATION ON
  COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")


add_library(acvo_gpu_geometric_lib ${ACVO_GPU_SOURCE})
target_include_directories(acvo_gpu_geometric_lib PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"        
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_compile_definitions(acvo_gpu_geometric_lib PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_GEOMETRIC_ONLY)
target_link_libraries(acvo_gpu_geometric_lib PRIVATE lie_group_utils cvo_utils_lib )
set_target_properties(acvo_gpu_geometric_lib PROPERTIES POSITION_INDEPENDENT_CODE ON
  CUDA_SEPERABLE_COMPILATION ON
  COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")
endif()
endif()



#####################################
## Non-Adaptive CVO GPU
if (${PROJECT_NAME}_ENABLE_CUDA)
add_library(cvo_gpu_img_lib SHARED ${CVO_GPU_SOURCE})
add_dependencies(cvo_gpu_img_lib cvo_utils_lib lie_group_utils)
target_link_libraries(cvo_gpu_img_lib PRIVATE lie_group_utils cvo_utils_lib yaml-cpp  Ceres::ceres )
#target_compile_definitions(cvo_gpu_img_lib ${PCL_DEFINITIONS})   
#set_target_properties(cvo_gpu_img_lib PROPERTIES PUBLIC_HEADER "${CVO_PUBLIC_HEADERS}")
target_include_directories(cvo_gpu_img_lib PUBLIC
  #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/UnifiedCvo>
  "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"        
  #$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include> 
  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_compile_definitions(cvo_gpu_img_lib PUBLIC -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5 -DCUDA_BLOCK_SIZE=512 -DCVO_POINT_NEIGHBORS=256
#  ${PCL_DEFINITIONS}
)
set_target_properties(cvo_gpu_img_lib PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPERABLE_COMPILATION ON
  COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")

add_library(cvo_gpu_rgb_lib SHARED ${CVO_GPU_SOURCE})
add_dependencies(cvo_gpu_rgb_lib cvo_utils_lib lie_group_utils)
target_link_libraries(cvo_gpu_rgb_lib PRIVATE lie_group_utils cvo_utils_lib yaml-cpp  Ceres::ceres )
#target_compile_definitions(cvo_gpu_img_lib ${PCL_DEFINITIONS})   
#set_target_properties(cvo_gpu_img_lib PROPERTIES PUBLIC_HEADER "${CVO_PUBLIC_HEADERS}")
target_include_directories(cvo_gpu_rgb_lib PUBLIC
  #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/UnifiedCvo>
  "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"        
  #$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include> 
  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_compile_definitions(cvo_gpu_rgb_lib PUBLIC -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=3 -DCUDA_BLOCK_SIZE=512 -DCVO_POINT_NEIGHBORS=256
#  ${PCL_DEFINITIONS}
)
set_target_properties(cvo_gpu_rgb_lib PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPERABLE_COMPILATION ON
  COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")


add_library(cvo_gpu_img_deep_feature_lib SHARED ${CVO_GPU_SOURCE})
add_dependencies(cvo_gpu_img_deep_feature_lib cvo_utils_lib lie_group_utils)
target_link_libraries(cvo_gpu_img_deep_feature_lib PRIVATE lie_group_utils cvo_utils_lib yaml-cpp  Ceres::ceres )
#target_compile_definitions(cvo_gpu_img_lib ${PCL_DEFINITIONS})   
#set_target_properties(cvo_gpu_img_lib PROPERTIES PUBLIC_HEADER "${CVO_PUBLIC_HEADERS}")
target_include_directories(cvo_gpu_img_deep_feature_lib PUBLIC
  #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/UnifiedCvo>
  "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"        
  #$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include> 
  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_compile_definitions(cvo_gpu_img_deep_feature_lib PUBLIC -DNUM_CLASSES=256 -DFEATURE_DIMENSIONS=5 -DCUDA_BLOCK_SIZE=512 -DCVO_POINT_NEIGHBORS=256
  # ${PCL_DEFINITIONS}
)
set_target_properties(cvo_gpu_img_deep_feature_lib PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPERABLE_COMPILATION ON
  COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")


if(${PROJECT_NAME}_ENABLE_LIDAR)
  add_library(cvo_gpu_lidar_lib SHARED ${CVO_GPU_SOURCE})
target_include_directories(cvo_gpu_lidar_lib PUBLIC
  #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/UnifiedCvo>
  "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"        
  #$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include> 
  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
add_dependencies(cvo_gpu_lidar_lib cvo_utils_lib lie_group_utils)
target_compile_definitions(cvo_gpu_lidar_lib PUBLIC -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DCUDA_BLOCK_SIZE=512 -DCVO_POINT_NEIGHBORS=256)
target_link_libraries(cvo_gpu_lidar_lib PRIVATE lie_group_utils cvo_utils_lib cvo_gpu_img_lib yaml-cpp Ceres::ceres)
set_target_properties(cvo_gpu_lidar_lib PROPERTIES
  POSITION_INDEPENDENT_CODE ON
    CUDA_SEPERABLE_COMPILATION ON
    COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fPIC>")
endif()  
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


#add_executable(acvo_semantic_lidar_align_gpu ${PROJECT_SOURCE_DIR}/src/experiments/main_acvo_gpu_lidar_align.cpp)
#target_compile_definitions(acvo_semantic_lidar_align_gpu PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=1 -DIS_USING_SEMANTICS)
#target_link_libraries(acvo_semantic_lidar_align_gpu  acvo_gpu_semantic_lidar_lib cvo_cpu_semantic_lib cvo_utils_lib boost_filesystem boost_system kitti)


endif()

add_executable(cvo_align_gpu_img ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_align_raw_image.cpp)
target_include_directories(cvo_align_gpu_img PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_link_libraries(cvo_align_gpu_img cvo_gpu_img_lib cvo_utils_lib kitti  boost_filesystem boost_system)

add_executable(cvo_irls_pcd ${PROJECT_SOURCE_DIR}/src/experiments/main_multi_frame_irls_pcd.cpp)
target_include_directories(cvo_irls_pcd PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_link_libraries(cvo_irls_pcd cvo_gpu_img_lib  cvo_utils_lib boost_filesystem boost_system)


#add_executable(cvo_irls_rand_pcd ${PROJECT_SOURCE_DIR}/src/experiments/main_multi_frame_irls_bunny_random.cpp)
#target_include_directories(cvo_irls_rand_pcd PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
#        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
#        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
#target_link_libraries(cvo_irls_rand_pcd cvo_gpu_img_lib  cvo_utils_lib boost_filesystem boost_system pcl_filters pcl_common)

add_executable(cvo_irls_rand_pcd ${PROJECT_SOURCE_DIR}/src/experiments/main_multi_frame_irls_toy_random.cpp)
target_include_directories(cvo_irls_rand_pcd PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_link_libraries(cvo_irls_rand_pcd cvo_gpu_img_lib  cvo_utils_lib boost_filesystem boost_system pcl_filters pcl_common)


add_executable(cvo_align_gpu_two_pcd ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_align_two_pcd.cpp)
target_include_directories(cvo_align_gpu_two_pcd PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
      target_link_libraries(cvo_align_gpu_two_pcd cvo_gpu_rgb_lib cvo_utils_lib boost_filesystem boost_system pcl_io pcl_common)

add_executable(kitti_depth_filtering ${PROJECT_SOURCE_DIR}/src/experiments/main_depth_filtering.cpp)
target_include_directories(kitti_depth_filtering PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
      target_link_libraries(kitti_depth_filtering kitti cvo_gpu_img_lib cvo_utils_lib boost_filesystem boost_system pcl_io pcl_common)
      

add_executable(cvo_align_gpu_two_color_pcd ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_align_two_color_pcd.cpp)
target_include_directories(cvo_align_gpu_two_color_pcd PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_link_libraries(cvo_align_gpu_two_color_pcd cvo_gpu_rgb_lib cvo_utils_lib boost_filesystem boost_system pcl_io pcl_common)


add_executable(cvo_irls_tum ${PROJECT_SOURCE_DIR}/src/experiments/main_multi_frame_irls_tum.cpp)
target_include_directories(cvo_irls_tum PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_link_libraries(cvo_irls_tum cvo_gpu_img_lib cvo_gpu_img_lib cvo_utils_lib tum boost_filesystem boost_system)

add_executable(cvo_irls_tartan ${PROJECT_SOURCE_DIR}/src/experiments/main_multi_frame_irls_tartan.cpp)
target_include_directories(cvo_irls_tartan PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_link_libraries(cvo_irls_tartan cvo_gpu_img_lib cvo_utils_lib tartan boost_filesystem boost_system pcl_filters)

add_executable(cvo_covis_tartan ${PROJECT_SOURCE_DIR}/src/experiments/main_covisMap_test.cpp)
target_include_directories(cvo_covis_tartan PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_link_libraries(cvo_covis_tartan cvo_gpu_img_lib cvo_gpu_img_lib cvo_utils_lib tartan boost_filesystem boost_system pcl_filters)


add_executable(cvo_irls_tartan_translation ${PROJECT_SOURCE_DIR}/src/experiments/main_multi_frame_irls_translation_only_tartan.cpp)
target_include_directories(cvo_irls_tartan_translation PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_link_libraries(cvo_irls_tartan_translation cvo_gpu_img_lib cvo_gpu_img_lib cvo_utils_lib tartan boost_filesystem boost_system pcl_filters)


add_executable(cvo_irls_kitti ${PROJECT_SOURCE_DIR}/src/experiments/main_multi_frame_irls_kitti.cpp)
target_include_directories(cvo_irls_kitti PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_link_libraries(cvo_irls_kitti cvo_gpu_img_lib cvo_utils_lib kitti boost_filesystem boost_system)



add_executable(cvo_align_gpu_semantic_img ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_align_semantic_image.cpp)
target_include_directories(cvo_align_gpu_semantic_img PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_link_libraries(cvo_align_gpu_semantic_img cvo_gpu_img_lib cvo_utils_lib kitti  boost_filesystem boost_system)

add_executable(cvo_align_gpu_rgbd ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_align_rgbd.cpp)
target_include_directories(cvo_align_gpu_rgbd PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_link_libraries(cvo_align_gpu_rgbd cvo_gpu_img_lib cvo_cpu_lib tum cvo_utils_lib boost_filesystem boost_system)

add_executable(cvo_align_gpu_rgbd_tartan ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_align_tartan.cpp)
target_include_directories(cvo_align_gpu_rgbd_tartan PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_link_libraries(cvo_align_gpu_rgbd_tartan cvo_gpu_img_lib cvo_cpu_lib tum cvo_utils_lib boost_filesystem boost_system tartan)

if (${PROJECT_NAME}_ENABLE_QT_VIEWER)
  add_executable(stack_pcd_viewer ${PROJECT_SOURCE_DIR}/src/experiments/main_stack_pcd_viewer.cpp)
  target_include_directories(stack_pcd_viewer PUBLIC
    #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
    #$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
    "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
  target_link_libraries(stack_pcd_viewer cvo_utils_lib boost_filesystem boost_system  pcl_common pcl_io pcl_visualization  pcd_qt_viewer_lib )

  add_executable(ellipse_viewer ${PROJECT_SOURCE_DIR}/src/experiments/main_ellipse_viewer.cpp)
  target_compile_definitions(ellipse_viewer PRIVATE -DNUM_CLASSES=19 -DFEATURE_DIMENSIONS=5)
  target_include_directories(ellipse_viewer PUBLIC
    #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
    #$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
    "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
  target_link_libraries(ellipse_viewer cvo_gpu_img_lib cvo_utils_lib boost_filesystem boost_system  pcl_common pcl_io pcl_visualization  pcd_qt_viewer_lib  kitti)
  
  
endif()

if (${PROJECT_NAME}_ENABLE_QT_VIEWER)
  add_executable(traj_playback ${PROJECT_SOURCE_DIR}/src/experiments/main_traj_playback.cpp)
  target_include_directories(traj_playback PUBLIC
    #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
    #$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
    "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
  target_link_libraries(traj_playback cvo_utils_lib boost_filesystem boost_system boost_filesystem pcl_common pcl_io pcl_visualization  pcd_qt_viewer_lib )
  
endif()


if (${PROJECT_NAME}_ENABLE_LIDAR)
  add_executable(cvo_align_gpu_lidar_raw ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_lidar_raw_geometric.cpp)
  target_include_directories(cvo_align_gpu_lidar_raw PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_link_libraries(cvo_align_gpu_lidar_raw cvo_gpu_lidar_lib kitti cvo_utils_lib boost_filesystem boost_system)
endif()

add_executable(gicp_align_two ${PROJECT_SOURCE_DIR}/src/experiments/gicp_align_two.cpp)
target_include_directories(gicp_align_two PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_link_libraries(gicp_align_two ${PCL_LIBRARIES})


add_executable(cvo_single_semantic_test ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_single_image_test.cpp)
target_include_directories(cvo_single_semantic_test PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_link_libraries(cvo_single_semantic_test cvo_gpu_img_lib cvo_utils_lib kitti  boost_filesystem boost_system)

add_executable(cvo_indicator_in_sequence ${PROJECT_SOURCE_DIR}/src/experiments/main_indicator_in_sequence.cpp)
target_include_directories(cvo_indicator_in_sequence PUBLIC
	#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> 
	#$<BUILD_INTERFACE:${UnifiedCvo_BINARY_DIR}/include>
        "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}> )
target_link_libraries(cvo_indicator_in_sequence cvo_gpu_img_lib cvo_utils_lib boost_filesystem boost_system kitti)

add_executable(cu_kd_tree_test ${PROJECT_SOURCE_DIR}/src/experiments/cu_kd_tree_test.cu)
target_include_directories(cu_kd_tree_test PUBLIC
      "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>"                  
    )
target_link_libraries(cu_kd_tree_test cvo_utils_lib cukdtree_lib ${PCL_LIBRARIES})

add_executable(cvo_align_two_nn_feature ${PROJECT_SOURCE_DIR}/src/experiments/main_cvo_gpu_align_two_nn_feature_pcd.cpp)
target_include_directories(cvo_align_two_nn_feature PUBLIC
      "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>")
target_link_libraries(cvo_align_two_nn_feature cvo_gpu_img_deep_feature_lib cvo_utils_lib kitti  boost_filesystem boost_system)

add_executable(gen_kitti_depth_npy ${PROJECT_SOURCE_DIR}/src/experiments/main_depth_gen.cpp )
target_include_directories(gen_kitti_depth_npy PUBLIC  "$<BUILD_INTERFACE:${CVO_INCLUDE_DIRS}>")
target_link_libraries(gen_kitti_depth_npy cvo_utils_lib kitti boost_filesystem cnpy)

endif()

#add_executable(cvo_local_mapping ${PROJECT_SOURCE_DIR}/src/main_local_mapping.cpp )
#target_link_libraries(cvo_local_mapping cvo boost_system boost_thread cxsparse  ${OpenCV_LIBS}  ${TBB_LIBRARIES} ${PCL_LIBRARIES}  )
add_executable(eth3d_local_mapping ${PROJECT_SOURCE_DIR}/src/experiments/main_local_mapping_eth3d.cpp )
target_link_libraries(eth3d_local_mapping cvo_gpu_img_lib boost_system boost_thread cxsparse  ${OpenCV_LIBS}  ${TBB_LIBRARIES} ${PCL_LIBRARIES} tum bki_mapping_lib)


#################################################################
# install
# Install the project in CMAKE_INSTALL_PREFIX (defaults to /usr/local on linux). 
# To change the install location, run 
#   cmake -DCMAKE_INSTALL_PREFIX=<desired-install-path> ..
#################################################################

# install directory
#set(${PROJECT_NAME}_TARGET_LIBS 
set(${PROJECT_NAME}_INCLUDE_DEST "include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}")
set(${PROJECT_NAME}_LIB_DEST "lib/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}")
#set(UNIFIED_CVO_BIN_DEST "bin/")


export(                                                                                                                                                                                                    
  TARGETS    elas tum kitti cnpy tartan lie_group_utils cvo_utils_lib cvo_gpu_img_lib  #cvo_irls_lib
  NAMESPACE  ${PROJECT_NAME}::        
  FILE ${PROJECT_NAME}Targets.cmake                                                                                                                                                                     
  )                                                                                                                                                                                                        
          

install(
    # targets to install
    TARGETS elas tum kitti cnpy tartan lie_group_utils cvo_utils_lib cvo_gpu_img_lib #cvo_irls_lib
    # name of the CMake "export group" containing the targets we want to install
    EXPORT ${PROJECT_NAME}Targets
    # Dynamic, static library and include destination locations after running
    # "make install"
    LIBRARY DESTINATION ${${PROJECT_NAME}_LIB_DEST}
    ARCHIVE DESTINATION ${${PROJECT_NAME}_LIB_DEST}
    #INCLUDES DESTINATION ${${PROJECT_NAME}_INCLUDE_DEST}
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}" 
    )
  
  message("install_include dir is ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION}")
  
  

  install(
    # The export we want to save (matches name defined above containing the
    # install targets)
    EXPORT ${PROJECT_NAME}Targets
    # CMake file in which to store the export's information
    FILE  ${PROJECT_NAME}Targets.cmake
    # Namespace prepends all targets in the export (when we import later, we
    # will use CMakeDemo::CMakeDemo)
    NAMESPACE ${PROJECT_NAME}::
    # where to place the resulting file (here, we're putting it with the library)
    DESTINATION cmake/ #${${PROJECT_NAME}_LIB_DEST}
    )  


  install(DIRECTORY ${CVO_PUBLIC_HEADERS} 	  
	  DESTINATION ${${PROJECT_NAME}_INCLUDE_DEST})


# Write a "version file" in case someone wants to only load a particular version 
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${${PROJECT_NAME}_VERSION}
  COMPATIBILITY AnyNewerVersion
  )
configure_package_config_file(                                                                                                                                                                             
  ${PROJECT_NAME}Config.cmake.in                                                                                                                                                                           
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake                                                                                                                                                  
  INSTALL_DESTINATION  cmake/                                                                                                                                                                              
  PATH_VARS PROJECT_NAME                                                                                                                                                                                   
  ) 
  
install(
  FILES 
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION cmake/ #${${PROJECT_NAME}_LIB_DEST}
  )

#################################################################
# exporting libraries 
#################################################################
#export(PACKAGE UNIFIED_CVO)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
#set_target_properties(cvo_gpu_img_lib PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
#set_target_properties(cvo_cpu_lib PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
#get_property(cvo_include_dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
#get_property(cvo_library_dirs TARGET OUTDOOR_CVO PROPERTY LIBRARY_OUTPUT_DIRECTORY)
#get_property(cvo_libraries TARGET OUTDOOR_CVO PROPERTY LIBRARY_OUTPUT_NAME)

#message("cvo_include_dirs: " ${cvo_include_dirs})
#message("cvo_library_dirs: " ${cvo_library_dirs})
#message("cvo_libraries: " ${cvo_libraries})

# Configure config file for local build tree
#configure_file(${PROJECT_SOURCE_DIR}/cmake/cvoConfig.cmake.in
#  "${PROJECT_BINARY_DIR}/cvoConfig.cmake" @ONLY)
