SET(PROJECT_NAME OUTDOOR_CVO)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
#set(CMAKE_VERBOSE_MAKEFILE ON)


# Release Mode
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -mkl -msse -msse2 -msse3 -msse4 -mavx2 -ftree-vectorize" )
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g  -msse -msse2 -msse3 -msse4 -mavx2 -ftree-vectorize" )

#set(BUILD_TYPE RelWithDebInfo)

set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# required libraries
find_package(SuiteParse REQUIRED)
find_package(Sophus REQUIRED)
message("Sophus dir: ${Sophus_INCLUDE_DIRS}")

set(EIGEN_INCLUDE_DIR /usr/include/eigen3)
find_package(TBB REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread timer chrono) 
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GTSAM REQUIRED)
message("gtsam dir: ${GTSAM_INCLUDE_DIR}")

find_package(OpenMP)
if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        message("OpenMP enabled")
endif()

## Enable OpenMP
add_definitions(-DOPENMP)

# flags
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS
  " ${SSE_FLAGS}  -fPIC -march=native"
)

# Sources files
set(CVO_LIB_SOURCE
  ${PROJECT_SOURCE_DIR}/src/CvoPointCloud.cpp
  ${PROJECT_SOURCE_DIR}/src/Frame.cpp
  ${PROJECT_SOURCE_DIR}/src/KittiHandler.cpp
  ${PROJECT_SOURCE_DIR}/src/PoseGraph.cpp
)

set(MAPPING_LIB_SOURCE
  ${PROJECT_SOURCE_DIR}/src/mapping/bkiblock.cpp
  ${PROJECT_SOURCE_DIR}/src/mapping/bkioctomap.cpp
  ${PROJECT_SOURCE_DIR}/src/mapping/bkioctree_node.cpp
  ${PROJECT_SOURCE_DIR}/src/mapping/bkioctree.cpp
  ${PROJECT_SOURCE_DIR}/src/mapping/point3f.cpp
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
)


# compile main library.
add_library(cvo ${CVO_LIB_SOURCE})
target_link_libraries(cvo ${TBB_LIBRARIES}  ${Sophus_LIBRARIES} ${PCL_LIBRARY_DIRS}  ${OpenCV_LIBS} gtsam)

add_library(mapping ${MAPPING_LIB_SOURCE})
target_link_libraries(mapping)

add_executable(cvo_kitti_color_stereo ${PROJECT_SOURCE_DIR}/src/main_kitti_stereo_color.cpp )
target_link_libraries(cvo_kitti_color_stereo cvo boost_system boost_thread cxsparse  ${OpenCV_LIBS}  ${TBB_LIBRARIES} ${PCL_LIBRARIES}  )

add_executable(cvo_align ${PROJECT_SOURCE_DIR}/src/main_cvo_align.cpp )
target_link_libraries(cvo_align cvo boost_system boost_thread cxsparse  ${OpenCV_LIBS}  ${TBB_LIBRARIES} ${PCL_LIBRARIES}  )

add_executable(local_mapping ${PROJECT_SOURCE_DIR}/src/main_local_mapping.cpp )
target_link_libraries(local_mapping cvo mapping boost_system boost_thread cxsparse  ${OpenCV_LIBS}  ${TBB_LIBRARIES} ${PCL_LIBRARIES}  )
