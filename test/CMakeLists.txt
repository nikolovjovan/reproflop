# Recurse into the "cpu" and "gpu" subdirectories to build architecture specific tests.
#
if (REPROFLOP_GPU)
   add_subdirectory (gpu)
else (REPROFLOP_CPU)
   add_subdirectory (cpu)
endif ()
