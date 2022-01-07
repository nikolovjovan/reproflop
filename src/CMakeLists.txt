# Recurse into the "cpu" and "gpu" subdirectories to build architecture specific library

# Compile for CPUs
option (REPROFLOP_CPU "Create a CPU specific library" OFF)
# Compile for GPUs
option (REPROFLOP_GPU "Create a GPU specific library" OFF)

if (REPROFLOP_GPU)
   add_subdirectory (gpu)
else (REPROFLOP_CPU)
   add_subdirectory (cpu)
endif (REPROFLOP_GPU)
