cmake_minimum_required (VERSION 3.18.4)

include(CheckCSourceCompiles)
include(CheckCXXSourceCompiles)
include(CheckFunctionExists)

project (REPROFLOP)
enable_language (CXX)

# The version number.
#
set (REPROFLOP_VERSION_MAJOR 1)
set (REPROFLOP_VERSION_MINOR 0)

# TODO: Remove if not needed for finding OpenCL kernel files...
#
# Source and binary directories.
#
# set (REPROFLOP_SOURCE_DIR ${PROJECT_SOURCE_DIR})
# set (REPROFLOP_BINARY_DIR ${PROJECT_BINARY_DIR})

# Enable testing.
#
enable_testing ()

# Add CMake options for target architecture.
#
option (REPROFLOP_CPU "Create a CPU specific library" OFF)
option (REPROFLOP_GPU "Create a GPU specific library" OFF)

# If specified, enable timing code (slows down execution).
#
option (REPROFLOP_TIMING "Enable/disable timing of library routines (uses cpu cycles)" OFF)
if (REPROFLOP_TIMING)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DREPROFLOP_TIMING")
endif (REPROFLOP_TIMING)

# TODO: Change default value to OFF...
#
# Add CMake option to enable default execution of examples with CTest.
#
option (REPROFLOP_TEST_EXAMPLES "Execute examples with default parameters during CTest" ON)

# Recurse into the "src" and "test" subdirectories to build library, examples and tests.
#
add_subdirectory (src)
add_subdirectory (test)
