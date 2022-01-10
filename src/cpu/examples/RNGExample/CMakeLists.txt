# Update compiler flags.
#
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# Add rounding-math for strong reproducibility example.
#
set_source_files_properties (RNGExample_StrongReproducibility.cpp PROPERTIES COMPILE_FLAGS -frounding-math)

# Add target example executables.
#
add_executable (RNGExample_Base RNGExample_Base.cpp)
add_executable (RNGExample_WeakReproducibility RNGExample_WeakReproducibility.cpp)
add_executable (RNGExample_StrongReproducibility RNGExample_StrongReproducibility.cpp)

# Link libraries.
#
target_link_libraries (RNGExample_Base ${EXTRA_LIBS})
target_link_libraries (RNGExample_WeakReproducibility ${EXTRA_LIBS})
target_link_libraries (RNGExample_StrongReproducibility ${EXTRA_LIBS})

# Set install target directory.
#
set (EXAMPLE_TARGET_DIR ${PROJECT_BINARY_DIR}/examples/cpu/RNGExample)

# Add install targets.
#
install (TARGETS RNGExample_Base DESTINATION ${EXAMPLE_TARGET_DIR})
install (TARGETS RNGExample_WeakReproducibility DESTINATION ${EXAMPLE_TARGET_DIR})
install (TARGETS RNGExample_StrongReproducibility DESTINATION ${EXAMPLE_TARGET_DIR})

# Add CMake tests to execute examples to verify default execution does not crash if option is specified.
#
if (REPROFLOP_TEST_EXAMPLES)
    add_test (RNGExample_Base ${EXAMPLE_TARGET_DIR}/RNGExample_Base)
    add_test (RNGExample_WeakReproducibility ${EXAMPLE_TARGET_DIR}/RNGExample_WeakReproducibility)
    add_test (RNGExample_StrongReproducibility ${EXAMPLE_TARGET_DIR}/RNGExample_StrongReproducibility)
endif (REPROFLOP_TEST_EXAMPLES)