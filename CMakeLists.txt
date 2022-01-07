cmake_minimum_required (VERSION 3.18.4)
include(CheckCSourceCompiles)
include(CheckCXXSourceCompiles)
include(CheckFunctionExists)

project (REPROFLOP)
enable_language (CXX)

# The version number.
set (REPROFLOP_VERSION_MAJOR 1)
set (REPROFLOP_VERSION_MINOR 0)
set (REPROFLOP_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set (REPROFLOP_BINARY_DIR ${PROJECT_BINARY_DIR})

# Configure a header file to pass some of the CMake settings to the source code
#
configure_file (
  "${PROJECT_SOURCE_DIR}/cmake/config.h.in"
  "${PROJECT_BINARY_DIR}/include/config.h"
)

# Include dir with config.h.in
#
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")

# Enable testing
#
enable_testing ()

# Recurse into the "src" subdirectory. This does not actually
# cause another cmake executable to run. The same process will walk through
# the project's entire directory structure.
#
add_subdirectory (src)

# Build a CPack driven installer package
#
include (InstallRequiredSystemLibraries)
set (CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set (CPACK_PACKAGE_VERSION_MAJOR "${REPROFLOP_VERSION_MAJOR}")
set (CPACK_PACKAGE_VERSION_MINOR "${REPROFLOP_VERSION_MINOR}")
include (CPack)
