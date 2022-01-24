set (LIB_NAME viewer)
set (REQUIRED_LIBS gtest pthread ceres glog qhull cupointcloud
  ${PCL_LIBRARIES} ${CUDA_LIBRARIES}
  ${CUDA_cudadevrt_LIBRARY})

file (GLOB HEADER_FILES *.h **/*.hpp *.cuh)

set(SRC
    viewer.cc
    viewer.h
    color_handler.cc
    color_handler.h
    )

add_library(${LIB_NAME} ${SRC})
target_link_libraries(${LIB_NAME} ${REQUIRED_LIBS})

install(TARGETS ${LIB_NAME} DESTINATION lib)
install(FILES ${HEADER_FILES} DESTINATION "include/viewer")
