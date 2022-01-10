# Generate CPU specific library.

# Include ReproFLOP library.
#
include_directories ("${CMAKE_CURRENT_SOURCE_DIR}/common")
include_directories ("${CMAKE_CURRENT_SOURCE_DIR}/ReproFLOP")
include_directories ("${PROJECT_BINARY_DIR}/include")
set (EXTRA_LIBS ${EXTRA_LIBS} reproflop_gpu)

# Set default compiler flags.
#
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -march=native -fabi-version=0 -O1 -Wall -Wno-comment -Wno-maybe-uninitialized")

# Link OpenCL library (required for building and executing GPU code).
#
set (EXTRA_LIBS ${EXTRA_LIBS} OpenCL)

# Architecture specific flags.
#
if (REPROFLOP_GPU_AMD)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -masm=att -DAMD")
    include_directories( "/opt/sdks/amd/current/include/" )
    link_directories ( "/opt/sdks/amd/current/lib/x86_64/" )
elseif (REPROFLOP_GPU_NVIDIA)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -masm=intel")
    include_directories( "$ENV{CUDA_HOME}/include/" )
    link_directories ( "$ENV{CUDA_HOME}/lib64/" )
endif ()

# Traverse library and examples.
#
add_subdirectory(ReproFLOP)
add_subdirectory(examples)
