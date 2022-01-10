# Grab the .c and .cpp files.
#
file (GLOB_RECURSE REPROFLOP_C_CPP_SOURCE "*.c" "*.cpp")

# Grab the C/C++ headers.
#
file (GLOB_RECURSE REPROFLOP_C_CPP_HEADERS "*.h")

set (REPROFLOP_FILES "${REPROFLOP_C_CPP_SOURCE};${REPROFLOP_C_CPP_HEADERS}")

# Add the main library.
#
add_library (reproflop_cpu ${REPROFLOP_FILES})

# Add library install target.
#
set (REPROFLOP_LIB "${PROJECT_BINARY_DIR}/lib")
install (TARGETS reproflop_cpu DESTINATION ${REPROFLOP_LIB})

# Add headers install target.
#
set (REPROFLOP_INCLUDE "${PROJECT_BINARY_DIR}/include")
install (FILES ${REPROFLOP_C_CPP_HEADERS} DESTINATION ${REPROFLOP_INCLUDE})
