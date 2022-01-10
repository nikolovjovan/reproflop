# Generate CPU specific tests.

# Include ReproFLOP library.
#
include_directories ("${PROJECT_SOURCE_DIR}/src/cpu/ReproFLOP")
include_directories ("${PROJECT_BINARY_DIR}/include")
set (EXTRA_LIBS ${EXTRA_LIBS} reproflop_cpu)

# Set default compiler flags.
#
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -march=native -masm=intel -fabi-version=0 -O1 -Wall")

# Add target test executables.
#
add_executable (TestBaseFunctionality TestBaseFunctionality.cpp ${REPROFLOP_INCLUDE})

# Set install target directory.
#
set (TEST_TARGET_DIR ${PROJECT_BINARY_DIR}/tests/cpu)

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
