set (LIB_NAME cupointcloud)
set (REQUIRED_LIBS gtest pthread ceres glog qhull)

file (GLOB HEADER_FILES *.h **/*.hpp)
set(SRC
    cupointcloud.h
    )

cuda_add_library(${LIB_NAME} ${SRC} STATIC)
target_link_libraries(${LIB_NAME} ${REQUIRED_LIBS})

install(TARGETS ${LIB_NAME} DESTINATION lib)
install(FILES ${HEADER_FILES} DESTINATION "include/cupointcloud/")
