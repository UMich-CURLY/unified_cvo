# - Try to find the UnifiedCvo library
# Once done this will define
#
#  UnifiedCvo_FOUND - system has UnifiedCvo
#  UnifiedCvo_INCLUDE_DIR - UnifiedCvo include directory
#  UnifiedCvo_LIB - UnifiedCvo library directory
#  UnifiedCvo_LIBRARIES - UnifiedCvo libraries to link

if(UnifiedCvo_FOUND)
    return()
endif()

# We prioritize libraries installed in /usr/local with the prefix .../UnifiedCvo-*, 
# so we make a list of them here
file(GLOB lib_glob "/usr/local/lib/UnifiedCvo-*")
file(GLOB inc_glob "/usr/local/include/UnifiedCvo-*")

# Find the library with the name "UnifiedCvo" on the system. Store the final path
# in the variable UnifiedCvo_LIB
find_library(UnifiedCvo_LIB 
    # The library is named "UnifiedCvo", but can have various library forms, like
    # libUnifiedCvo.a, libUnifiedCvo.so, libUnifiedCvo.so.1.x, etc. This should
    # search for any of these.
    NAMES UnifiedCvo
    # Provide a list of places to look based on prior knowledge about the system.
    # We want the user to override /usr/local with environment variables, so
    # this is included here.
    HINTS
        ${UnifiedCvo_DIR}
        ${UNIFIEDCVO_DIR}
        $ENV{UnifiedCvo_DIR}
        $ENV{UNIFIEDCVO_DIR}
        ENV UNIFIEDCVO_DIR
    # Provide a list of places to look as defaults. /usr/local shows up because
    # that's the default install location for most libs. The globbed paths also
    # are placed here as well.
    PATHS
        /usr
        /usr/local
        /usr/local/lib
        ${lib_glob}
    # Constrain the end of the full path to the detected library, not including
    # the name of library itself.
    PATH_SUFFIXES 
        lib
)

# Find the path to the file "source_file.hpp" on the system. Store the final
# path in the variables UnifiedCvo_INCLUDE_DIR. The HINTS, PATHS, and
# PATH_SUFFIXES, arguments have the same meaning as in find_library().
#find_path(UnifiedCvo_INCLUDE_DIR source_file.hpp
#    HINTS
#        ${UnifiedCvo_DIR}
#        ${UNIFIEDCVO_DIR}
#        $ENV{UnifiedCvo_DIR}
#        $ENV{UNIFIEDCVO_DIR}
#        ENV UNIFIEDCVO_DIR
#    PATHS
#        /usr
#        /usr/local
#        /usr/local/include
#        ${inc_glob}
#    PATH_SUFFIXES 
#        include
#)


# Check that both the paths to the include and library directory were found.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(UnifiedCvo
    "\nUnifiedCvo not found --- You can download it using:\n\tgit clone 
    https://github.com/mmorse1217/cmake-project-template\n and setting the 
    CMAKEDEMO_DIR environment variable accordingly"
    UnifiedCvo_LIB UnifiedCvo_INCLUDE_DIR)

# These variables don't show up in the GUI version of CMake. Not required but
# people seem to do this...
mark_as_advanced(UnifiedCvo_INCLUDE_DIR UnifiedCvo_LIB)

# Finish defining the variables specified above. Variables names here follow
# CMake convention.
set(UnifiedCvo_INCLUDE_DIRS ${UnifiedCvo_INCLUDE_DIR})
set(UnifiedCvo_LIBRARIES ${UnifiedCvo_LIB})

# If the above CMake code was successful and we found the library, and there is
# no target defined, lets make one.
#if(UnifiedCvo_FOUND AND NOT TARGET UnifiedCvo::UnifiedCvo)
#    add_library(UnifiedCvo::UnifiedCvo UNKNOWN IMPORTED)
    # Set location of interface include directory, i.e., the directory
    # containing the header files for the installed library
#    set_target_properties(UnifiedCvo::UnifiedCvo PROPERTIES
#        INTERFACE_INCLUDE_DIRECTORIES "${UnifiedCvo_INCLUDE_DIRS}"
#        )

    # Set location of the installed library
#    set_target_properties(UnifiedCvo::UnifiedCvo PROPERTIES
#        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
#        IMPORTED_LOCATION "${UnifiedCvo_LIBRARIES}"
#        )
#endif()
