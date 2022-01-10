# Update compiler flags.
#
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# Link math library.
#
set (EXTRA_LIBS ${EXTRA_LIBS} m)

# Set install target directory.
#
set (EXAMPLE_TARGET_DIR ${PROJECT_BINARY_DIR}/examples/cpu/mri-gridding)

# Traverse example versions.
#
add_subdirectory (Base_WeakReproducibility)
add_subdirectory (StrongReproducibility)

# Copy input data files during install.
#
install (DIRECTORY data DESTINATION ${EXAMPLE_TARGET_DIR})
