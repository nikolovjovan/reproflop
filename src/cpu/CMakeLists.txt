# Generate CPU specific library.

# Include ReproFLOP library.
#
include_directories ("${CMAKE_CURRENT_SOURCE_DIR}/ReproFLOP")
include_directories ("${PROJECT_BINARY_DIR}/include")
set (EXTRA_LIBS ${EXTRA_LIBS} reproflop_cpu)

# Set default compiler flags.
#
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -march=native -masm=intel -fabi-version=0 -O1 -Wall")

# Traverse library and examples.
#
add_subdirectory(ReproFLOP)
add_subdirectory(examples)
