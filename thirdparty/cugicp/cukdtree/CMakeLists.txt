set (LIB_NAME cukdtree)
set (REQUIRED_LIBS gtest pthread ceres glog qhull cupointcloud ${CUDA_LIBRARIES}
  ${CUDA_cudadevrt_LIBRARY})

file (GLOB HEADER_FILES *.h **/*.hpp *.cuh)

set(SRC
    cukdtree.cuh
    cukdtree.h
    )

cuda_add_library(${LIB_NAME} ${SRC} STATIC)
target_link_libraries(${LIB_NAME} ${REQUIRED_LIBS})

install(TARGETS ${LIB_NAME} DESTINATION lib)
install(FILES ${HEADER_FILES} DESTINATION "include/cukdtree")
