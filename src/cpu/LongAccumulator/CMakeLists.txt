# Testing
#
add_executable (test_base_fn test_base_fn.cpp)
target_link_libraries (test_base_fn ${EXTRA_LIBS})

# Add the install targets
#
install (TARGETS test_base_fn DESTINATION ${PROJECT_BINARY_DIR}/tests/cpu)

# Grab the .c and .cpp files
#
file (GLOB_RECURSE REPROFLOP_C_CPP_SOURCE "*.c" "*.cpp" ${CMAKE_CURRENT_SOURCE_DIR})

# Grab the C/C++ headers
#
file (GLOB_RECURSE REPROFLOP_C_CPP_HEADERS "LongAccumulator.h" ${CMAKE_CURRENT_SOURCE_DIR})
set (REPROFLOP_C_CPP_FILES "${REPROFLOP_C_CPP_SOURCE};${REPROFLOP_C_CPP_HEADERS}")

# Add the main library
#
add_library (reproflop_cpu ${REPROFLOP_C_CPP_FILES})
set (REPROFLOP_LIB "${PROJECT_BINARY_DIR}/lib")
install (TARGETS reproflop_cpu DESTINATION ${REPROFLOP_LIB})

set (REPROFLOP_INCLUDE "${PROJECT_BINARY_DIR}/include")
install (FILES ${REPROFLOP_C_CPP_HEADERS} DESTINATION ${REPROFLOP_INCLUDE})

add_test (TestSimpleAddition ${PROJECT_BINARY_DIR}/tests/cpu/test_base_fn)
set_tests_properties (TestSimpleAddition PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; All OK!")
