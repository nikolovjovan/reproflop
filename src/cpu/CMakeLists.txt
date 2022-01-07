# Generate CPU specific library

# Add the ReproFlop library?
#
option (USE_REPROFLOP "Use ReproFlop super accumulator implementation for reproducible and accurate summation" ON)
if (USE_REPROFLOP)
  include_directories ("${CMAKE_CURRENT_SOURCE_DIR}/LongAccumulator")
  include_directories ("${PROJECT_BINARY_DIR}/include")
  set (EXTRA_LIBS ${EXTRA_LIBS} reproflop_cpu)
endif (USE_REPROFLOP)

# Compiler flags
#
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -march=native -masm=intel -fabi-version=0 -O1 -Wall")

# Enabling timing
#
option (REPROFLOP_TIMING "Enable/disable timing of library routines (uses cpu cycles)" OFF)
if (REPROFLOP_TIMING)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DREPROFLOP_TIMING")
endif (REPROFLOP_TIMING)

add_subdirectory(LongAccumulator)
# add_subdirectory(RNGExample)
# add_subdirectory("kmeans (Rodinia)")
# add_subdirectory("mri-gridding (Parboil)")