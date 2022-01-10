# Generate CPU specific tests.

# Include ReproFLOP library.
#
include_directories ("${PROJECT_SOURCE_DIR}/src/cpu/ReproFLOP")
include_directories ("${PROJECT_BINARY_DIR}/include")
set (EXTRA_LIBS ${EXTRA_LIBS} reproflop_cpu)

# Set default compiler flags.
#
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -march=native -masm=intel -fabi-version=0 -O1 -Wall")

# If specified, enable timing code (slows down execution).
#
option (REPROFLOP_TIMING "Enable/disable timing of library routines (uses cpu cycles)" OFF)
if (REPROFLOP_TIMING)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DREPROFLOP_TIMING")
endif (REPROFLOP_TIMING)

# Add target test executables.
#
add_executable (TestBaseFunctionality TestBaseFunctionality.cpp ${REPROFLOP_INCLUDE})

# Link libraries.
#
target_link_libraries (TestBaseFunctionality ${EXTRA_LIBS})

# Add install targets.
#
install (TARGETS TestBaseFunctionality DESTINATION ${PROJECT_BINARY_DIR}/tests/cpu)

# Add CMake tests.
#
add_test (TestBaseFunctionality ${PROJECT_BINARY_DIR}/tests/cpu/TestBaseFunctionality)
set_tests_properties (TestBaseFunctionality PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; All OK!")
