#get_filename_component(SELF_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

include(CMakeFindDependencyMacro)
# Capturing values from configure (optional)
#set(my-config-var @my-config-var@)

# Same syntax as find_package
#find_dependency(LAPACK REQUIRED)

# Any extra setup

# Add the targets file. include() just loads and executes the CMake code in the
# file passed to it. Note that the file loaded here is the same one generated in
# the second install() command in the root-level CMakeLists.txt
include("${CMAKE_CURRENT_LIST_DIR}/UnifiedCvoTargets.cmake")
