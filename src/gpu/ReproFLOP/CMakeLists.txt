# Grab the .c and .cpp files.
#
file (GLOB_RECURSE REPROFLOP_C_CPP_SOURCE "*.c" "*.cpp" "${PROJECT_SOURCE_DIR}/src/gpu/common/*.c" "${PROJECT_SOURCE_DIR}/src/gpu/common/*.cpp")

# Grab the C/C++ headers.
#
file (GLOB_RECURSE REPROFLOP_C_CPP_HEADERS "*.h" "${PROJECT_SOURCE_DIR}/src/gpu/common/*.h")

# Grab the .cl files.
#
file (GLOB_RECURSE REPROFLOP_CL_SOURCE "*.cl")

set (REPROFLOP_FILES "${REPROFLOP_C_CPP_SOURCE};${REPROFLOP_C_CPP_HEADERS};${REPROFLOP_CL_SOURCE}")

# Add the main library.
#
add_library (reproflop_gpu ${REPROFLOP_FILES})

# Add library install target.
#
set (REPROFLOP_LIB "${PROJECT_BINARY_DIR}/lib")
install (TARGETS reproflop_gpu DESTINATION ${REPROFLOP_LIB})

# Add headers install target.
#
set (REPROFLOP_INCLUDE "${PROJECT_BINARY_DIR}/include")
install (FILES ${REPROFLOP_C_CPP_HEADERS} DESTINATION ${REPROFLOP_INCLUDE})
install (FILES ${REPROFLOP_CL_SOURCE} DESTINATION "${REPROFLOP_INCLUDE}/cl")
