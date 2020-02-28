SET(PROJECT_NAME OUTDOOR_CVO)

PROJECT(${PROJECT_NAME})
cmake_minimum_required(VERSION 3.5.1)
#set(CMAKE_VERBOSE_MAKEFILE ON)

add_compile_options(-std=c++17)
# Release Mode
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -msse  -msse2 -msse3 -msse4 -mavx2 -ftree-vectorize" )
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -lmkl_intel_lp64 -lmkl_core -lmkl -liomp -msse  -msse2 -msse3 -msse4 -mavx2 -ftree-vectorize" )
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++17 -g  -msse -msse2 -msse3 -msse4 -mavx2 -ftree-vectorize" )

#set(BUILD_TYPE RelWithDebInfo)

set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# required libraries
find_package(SuiteParse REQUIRED)
find_package(Sophus REQUIRED)
message("Sophus dir: ${Sophus_INCLUDE_DIRS}")

#set(EIGEN_INCLUDE_DIR /usr/include/eigen3)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread timer chrono)
if (Boost_FOUND)
message("Find Boost! ${Boost_LIBRARIES}")
endif()

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GTSAM REQUIRED)
message("gtsam dir: ${GTSAM_INCLUDE_DIR}")

find_package(OpenMP)
if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        message("OpenMP enabled")
else()
	message("OpenMP not found")
endif()

## Enable OpenMP
add_definitions(-DOPENMP)
#add_definitions(-DDEBUG)

# flags
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS
	"${CMAKE_CXX_FLAGS} ${SSE_FLAGS}  -fPIC -march=native"
	#"${CMAKE_CXX_FLAGS} ${SSE_FLAGS}  -fPIC "
)

# Sources files
set(CVO_LIB_SOURCE
  ${PROJECT_SOURCE_DIR}/src/Cvo.cpp
  ${PROJECT_SOURCE_DIR}/src/ACvo.cpp
  ${PROJECT_SOURCE_DIR}/src/CvoPointCloud.cpp
  ${PROJECT_SOURCE_DIR}/src/Frame.cpp
  ${PROJECT_SOURCE_DIR}/src/KittiHandler.cpp
  ${PROJECT_SOURCE_DIR}/src/PoseGraph.cpp
  ${PROJECT_SOURCE_DIR}/src/StaticStereo.cpp
  ${PROJECT_SOURCE_DIR}/src/CvoPixelSelector.cpp
  ${PROJECT_SOURCE_DIR}/src/RawImage.cpp
  ${PROJECT_SOURCE_DIR}/src/mapping/bkiblock.cpp
  ${PROJECT_SOURCE_DIR}/src/mapping/bkioctomap.cpp
  ${PROJECT_SOURCE_DIR}/src/mapping/bkioctree_node.cpp
  ${PROJECT_SOURCE_DIR}/src/mapping/bkioctree.cpp
  ${PROJECT_SOURCE_DIR}/src/mapping/point3f.cpp
  ${PROJECT_SOURCE_DIR}/src/CvoPointCloud.cpp
  ${PROJECT_SOURCE_DIR}/include/outdoor_cvo/cvo/LieGroup.cpp
  )

# thirdparty elas
set(ELAS_LIB_SOURCE
  ${PROJECT_SOURCE_DIR}/thirdparty/libelas/descriptor.cpp
  ${PROJECT_SOURCE_DIR}/thirdparty/libelas/elas.cpp
  ${PROJECT_SOURCE_DIR}/thirdparty/libelas/filter.cpp
  ${PROJECT_SOURCE_DIR}/thirdparty/libelas/matrix.cpp
  ${PROJECT_SOURCE_DIR}/thirdparty/libelas/triangle.cpp
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
  ${PROJECT_SOURCE_DIR}/thirdparty/
) 

message("CXX_FLAGS are ${CMAKE_CXX_FLAGS}")

# compile main library.
add_library(elas ${ELAS_LIB_SOURCE})

add_library(cvo ${CVO_LIB_SOURCE})
target_link_libraries(cvo elas ${Boost_LIBRARIES} ${TBB_LIBRARIES}  ${Sophus_LIBRARIES} ${PCL_LIBRARIES}  ${OpenCV_LIBS} ${TBB_LIBRARIES} gtsam  gtsam_unstable )

add_library(acvo ${CVO_LIB_SOURCE})
target_link_libraries(acvo elas ${Boost_LIBRARIES} ${TBB_LIBRARIES}  ${Sophus_LIBRARIES} ${PCL_LIBRARIES}  ${OpenCV_LIBS} ${TBB_LIBRARIES} gtsam  gtsam_unstable )

add_executable(cvo_kitti_color_stereo ${PROJECT_SOURCE_DIR}/src/main_kitti_stereo_color.cpp )
target_link_libraries(cvo_kitti_color_stereo cvo  ${OpenCV_LIBS}  )

add_executable(cvo_kitti_stereo_semantic ${PROJECT_SOURCE_DIR}/src/main_kitti_stereo_semantic.cpp )
target_link_libraries(cvo_kitti_stereo_semantic cvo  ${OpenCV_LIBS} )
add_executable(cvo_kitti_stereo_odom_file ${PROJECT_SOURCE_DIR}/src/main_kitti_stereo_odom_file.cpp )
target_link_libraries(cvo_kitti_stereo_odom_file cvo  ${OpenCV_LIBS} gtsam gtsam_unstable)

add_executable(cvo_align ${PROJECT_SOURCE_DIR}/src/main_cvo_align.cpp )
target_link_libraries(cvo_align cvo boost_system boost_thread cxsparse  ${OpenCV_LIBS}  ${TBB_LIBRARIES} ${PCL_LIBRARIES})

add_executable(cvo_f2f ${PROJECT_SOURCE_DIR}/src/main_cvo_align_f2f.cpp )
target_link_libraries(cvo_f2f cvo acvo boost_system boost_thread cxsparse  ${OpenCV_LIBS}  ${TBB_LIBRARIES} ${PCL_LIBRARIES})

add_executable(cvo_map2map ${PROJECT_SOURCE_DIR}/src/main_map2map_batch.cpp)
target_link_libraries(cvo_map2map cvo boost_system)

add_executable(cvo_local_mapping ${PROJECT_SOURCE_DIR}/src/main_local_mapping.cpp )
target_link_libraries(cvo_local_mapping cvo boost_system boost_thread cxsparse  ${OpenCV_LIBS}  ${TBB_LIBRARIES} ${PCL_LIBRARIES}  )
