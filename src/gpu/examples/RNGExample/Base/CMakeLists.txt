# Set target name.
#
set (EXAMPLE_TARGET_NAME RNGExample_Base)

# Grab the .c and .cpp files.
#
file (GLOB_RECURSE EXAMPLE_C_CPP_SOURCE "*.c" "*.cpp")

# Grab the C/C++ headers.
#
file (GLOB_RECURSE EXAMPLE_C_CPP_HEADERS "*.h")

# Grab the .cl files.
#
file (GLOB_RECURSE EXAMPLE_CL_SOURCE "*.cl")

set (EXAMPLE_FILES "${EXAMPLE_C_CPP_SOURCE};${EXAMPLE_C_CPP_HEADERS};${EXAMPLE_CL_SOURCE}")

# Add target example executable.
#
add_executable (${EXAMPLE_TARGET_NAME} ${EXAMPLE_FILES})

# Link libraries.
#
target_link_libraries (${EXAMPLE_TARGET_NAME} ${EXTRA_LIBS})

# Add install targets.
#
install (TARGETS ${EXAMPLE_TARGET_NAME} DESTINATION ${EXAMPLE_TARGET_DIR})

# Add headers install target.
#
set (REPROFLOP_INCLUDE "${PROJECT_BINARY_DIR}/include")
install (FILES ${EXAMPLE_CL_SOURCE} DESTINATION "${REPROFLOP_INCLUDE}/cl")

# Add CMake tests to execute examples to verify default execution does not crash if option is specified.
#
if (TEST_EXAMPLES)
    add_test (${EXAMPLE_TARGET_NAME} ${EXAMPLE_TARGET_DIR}/${EXAMPLE_TARGET_NAME})
endif ()
