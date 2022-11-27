# Update compiler flags.
#
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# Set install target directory.
#
set (EXAMPLE_TARGET_DIR ${PROJECT_BINARY_DIR}/examples/gpu/RNGExample)

# Traverse example versions.
#
add_subdirectory (Base)
add_subdirectory (StrongReproducibility)
