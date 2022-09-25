# Generate CPU specific tests.

# Include ReproFLOP library.
#
include_directories ("${PROJECT_SOURCE_DIR}/src/gpu/common")
include_directories ("${PROJECT_SOURCE_DIR}/src/gpu/ReproFLOP")
include_directories ("${PROJECT_BINARY_DIR}/include")
set (EXTRA_LIBS ${EXTRA_LIBS} reproflop_gpu)

# Set default compiler flags.
#
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -march=native -fabi-version=0 -g -Wall -Wno-comment -Wno-maybe-uninitialized")

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

# Add target test executables.
#
add_executable (TestBaseFunctionality TestBaseFunctionality.cpp ${REPROFLOP_INCLUDE})

# Set install target directory.
#
set (TEST_TARGET_DIR ${PROJECT_BINARY_DIR}/tests/gpu)

# Link libraries.
#
target_link_libraries (TestBaseFunctionality ${EXTRA_LIBS})

# Add install targets.
#
install (TARGETS TestBaseFunctionality DESTINATION ${TEST_TARGET_DIR})

# Add CMake tests.
#
add_test (TestBaseFunctionality ${TEST_TARGET_DIR}/TestBaseFunctionality)
set_tests_properties (TestBaseFunctionality PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; All OK!")
