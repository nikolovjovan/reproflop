#include "LongAccumulator.h"
// #include "LongAccumulatorCPU.h"

// #include <iostream>
#include <limits>

using namespace std;

#define OPENCL_PROGRAM_FILENAME "LongAccumulator.cl"
#define ACCUMULATE_KERNEL_NAME  "LongAccumulatorAccumulate"
#define MERGE_KERNEL_NAME       "LongAccumulatorMerge"
#define ROUND_KERNEL_NAME       "LongAccumulatorRound"

#ifdef AMD
constexpr char compileOptionsFormat[256] = "-DACCUMULATOR_SIZE=%u -DACCUMULATOR_COUNT=%u -DWARP_SIZE=%u -DACCUMULATE_WARP_COUNT=%u -DMERGE_WARP_COUNT=%u -DMERGE_ACCUMULATOR_COUNT=%u -DUSE_KNUTH";
#else
constexpr char compileOptionsFormat[256] = "-DACCUMULATOR_SIZE=%u -DACCUMULATOR_COUNT=%u -DWARP_SIZE=%u -DACCUMULATE_WARP_COUNT=%u -DMERGE_WARP_COUNT=%u -DMERGE_ACCUMULATOR_COUNT=%u -DUSE_KNUTH -DNVIDIA -cl-mad-enable -cl-fast-relaxed-math";
#endif

bool LongAccumulator::s_initializedOpenCL = false;
bool LongAccumulator::s_initializedLngAcc = false;

cl_platform_id LongAccumulator::s_platform = nullptr;
cl_device_id LongAccumulator::s_device = nullptr;
cl_context LongAccumulator::s_context = nullptr;
cl_command_queue LongAccumulator::s_commandQueue = nullptr;

cl_program LongAccumulator::s_program = nullptr;
cl_kernel LongAccumulator::s_clkAccumulate = nullptr;
cl_kernel LongAccumulator::s_clkMerge = nullptr;
cl_kernel LongAccumulator::s_clkRound = nullptr;

cl_mem LongAccumulator::s_data_arr = nullptr;
cl_mem LongAccumulator::s_data_res = nullptr;

cl_mem LongAccumulator::s_accumulators = nullptr;

int LongAccumulator::InitializeOpenCL()
{
    if (s_initializedOpenCL) {
        return 1;
    }

    s_initializedOpenCL = true;

    cl_int ciErrNum;

    char platform_name[64];

#ifdef AMD
    strcpy(platform_name, "AMD Accelerated Parallel Processing");
#else
    strcpy(platform_name, "NVIDIA CUDA");
#endif

    s_platform = GetOCLPlatform(platform_name);
    if (s_platform == NULL) {
        cerr << "ERROR: Failed to find the platform '" << platform_name << "' ...\n";
        return -1;
    }

    // Get a GPU device
    //
    s_device = GetOCLDevice(s_platform);
    if (s_device == NULL) {
        cerr << "Error in clGetDeviceIDs, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        return -2;
    }

    // Create the context
    //
    s_context = clCreateContext(0, 1, &s_device, NULL, NULL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        cerr << "Error in clCreateContext, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        return -3;
    }

    // Create a command-queue
    //
    s_commandQueue = clCreateCommandQueue(s_context, s_device, CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        cerr << "Error = " << ciErrNum << "\n";
        cerr << "Error in clCreateCommandQueue, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        return -4;
    }

    ciErrNum = InitializeAcc(s_context, s_commandQueue, s_device);
    if (ciErrNum != CL_SUCCESS) {
        cerr << "Error = " << ciErrNum << "\n";
        cerr << "Error in InitializeAcc, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        return -5;
    }

    return 0;
}

int LongAccumulator::CleanupOpenCL()
{
    cl_int ciErrNum;

    if (!s_initializedOpenCL) {
        return 0;
    }

    // Release kernels and program
    //
    ciErrNum = CleanupAcc();

    // Shutting down and freeing memory...
    //
    if (s_commandQueue) {
        clReleaseCommandQueue(s_commandQueue);
    }
    if (s_context) {
        clReleaseContext(s_context);
    }

    s_initializedOpenCL = false;

    return ciErrNum;
}

cl_int LongAccumulator::InitializeAcc(
    cl_context context,
    cl_command_queue commandQueue,
    cl_device_id device)
{
    cl_int ciErrNum;

    size_t sourceCodeLength = 0;
    char* sourceCode = nullptr;

    char path[256];
    strcpy(path, REPROFLOP_BINARY_DIR);
    strcat(path, "/include/cl/");
    strcat(path, OPENCL_PROGRAM_FILENAME);

    // Read the OpenCL kernel in from source file
    //
    FILE *pProgramHandle = fopen(path, "r");
    if (!pProgramHandle) {
        cerr << "Failed to load kernel.\n";
        return -1;
    }
    fseek(pProgramHandle, 0, SEEK_END);
    sourceCodeLength = ftell(pProgramHandle);
    rewind(pProgramHandle);
    sourceCode = (char *) malloc(sourceCodeLength + 1);
    sourceCode[sourceCodeLength] = '\0';
    if (fread(sourceCode, sizeof(char), sourceCodeLength, pProgramHandle) != sourceCodeLength)
    {
        cerr << "Failed to read source code.\n";
        return -2;
    }
    fclose(pProgramHandle);

    s_initializedLngAcc = true;

    // Create OpenCL program from source
    //
    s_program = clCreateProgramWithSource(context, 1, (const char **)&sourceCode, &sourceCodeLength, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        cerr << "Error = " << ciErrNum << "\n";
        cerr << "Error in clCreateProgramWithSource, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        return -3;
    }

    // Building LongAccumulator program
    //
    char compileOptions[256];
    snprintf(compileOptions, 256, compileOptionsFormat, ACCUMULATOR_SIZE, ACCUMULATOR_COUNT, WARP_SIZE, ACCUMULATE_WARP_COUNT, MERGE_WARP_COUNT, MERGE_ACCUMULATOR_COUNT);
    ciErrNum = clBuildProgram(s_program, 0, NULL, compileOptions, NULL, NULL);
    if (ciErrNum != CL_SUCCESS) {
        cerr << "Error = " << ciErrNum << "\n";
        cerr << "Error in clBuildProgram, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";

        // Determine the reason for the error
        //
        char buildLog[16384];
        clGetProgramBuildInfo(s_program, s_device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), &buildLog, NULL);

        cout << buildLog << "\n";
        return -4;
    }

    // Create kernels
    //
    s_clkAccumulate = clCreateKernel(s_program, ACCUMULATE_KERNEL_NAME, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        cerr << "Error = " << ciErrNum << "\n";
        cerr << "Error in clCreateKernel: " ACCUMULATE_KERNEL_NAME ", Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        return -5;
    }
    s_clkMerge = clCreateKernel(s_program, MERGE_KERNEL_NAME, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        cerr << "Error = " << ciErrNum << "\n";
        cerr << "Error in clCreateKernel: " MERGE_KERNEL_NAME ", Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        return -6;
    }
    s_clkRound = clCreateKernel(s_program, ROUND_KERNEL_NAME, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        cerr << "Error = " << ciErrNum << "\n";
        cerr << "Error in clCreateKernel: " ROUND_KERNEL_NAME ", Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        return -7;
    }

    // Allocate internal buffers
    //
    uint32_t size = ACCUMULATOR_COUNT * ACCUMULATOR_SIZE * sizeof(cl_uint);
    s_accumulators = clCreateBuffer(s_context, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        printf("Error in clCreateBuffer for s_accumulators, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        return -8;
    }

    // Deallocate temp storage
    //
    free(sourceCode);

    return 0;
}

cl_int LongAccumulator::CleanupAcc()
{
    cl_int ciErrNum;
    cl_int res = 0;

    if (!s_initializedLngAcc) {
        return 0;
    }

    // Release memory...
    //
    if (s_accumulators) {
        ciErrNum = clReleaseMemObject(s_accumulators);
        if (ciErrNum != CL_SUCCESS) {
            cerr << "Error = " << ciErrNum << "\n";
            cerr << "Error in clReleaseMemObject, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
            res |= 1 << 0;
        }
    }

    if (s_clkAccumulate) {
        ciErrNum = clReleaseKernel(s_clkAccumulate);
        if (ciErrNum != CL_SUCCESS) {
            cerr << "Error = " << ciErrNum << "\n";
            cerr << "Error in clReleaseKernel: " ACCUMULATE_KERNEL_NAME ", Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
            res |= 1 << 1;
        }
    }

    if (s_clkMerge) {
        ciErrNum = clReleaseKernel(s_clkMerge);
        if (ciErrNum != CL_SUCCESS) {
            cerr << "Error = " << ciErrNum << "\n";
            cerr << "Error in clReleaseKernel: " MERGE_KERNEL_NAME ", Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
            res |= 1 << 2;
        }
    }

    if (s_clkRound) {
        ciErrNum = clReleaseKernel(s_clkRound);
        if (ciErrNum != CL_SUCCESS) {
            cerr << "Error = " << ciErrNum << "\n";
            cerr << "Error in clReleaseKernel: " ROUND_KERNEL_NAME ", Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
            res |= 1 << 4;
        }
    }

    if (s_program) {
        ciErrNum = clReleaseProgram(s_program);
        if (ciErrNum != CL_SUCCESS) {
            cerr << "Error = " << ciErrNum << "\n";
            cerr << "Error in clReleaseProgram, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
            res |= 1 << 8;
        }
    }

    return -res;
}

float LongAccumulator::Sum(const int N, float *arr, int *err)
{
    cl_int ciErrNum;
    float result = 0;

    // Allocating OpenCL memory...
    //
    s_data_arr = clCreateBuffer(s_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_float), arr, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        cerr << "Error in clCreateBuffer for s_data_arr, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        if (err != nullptr) {
            *err = ciErrNum;
        }
        return numeric_limits<float>::signaling_NaN();
    }
    s_data_res = clCreateBuffer(s_context, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        cerr <<"Error in clCreateBuffer for s_data_res, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        if (err != nullptr) {
            *err = ciErrNum;
        }
        return numeric_limits<float>::signaling_NaN();
    }

    size_t global_work_size, local_work_size;

    {
        local_work_size = ACCUMULATE_WORKGROUP_SIZE;
        global_work_size = local_work_size * ACCUMULATOR_COUNT;

        cl_uint i = 0;
        ciErrNum  = clSetKernelArg(s_clkAccumulate, i++, sizeof(cl_uint), (void *)&N);
        ciErrNum |= clSetKernelArg(s_clkAccumulate, i++, sizeof(cl_mem),  (void *)&s_data_arr);
        ciErrNum |= clSetKernelArg(s_clkAccumulate, i++, sizeof(cl_mem),  (void *)&s_accumulators);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            if (err != nullptr) {
                *err = ciErrNum;
            }
            return numeric_limits<float>::signaling_NaN();
        }

        ciErrNum = clEnqueueNDRangeKernel(
            s_commandQueue,
            s_clkAccumulate,
            /* work_dim */ 1,
            /* global_work_offset */ NULL,
            &global_work_size,
            &local_work_size,
            /* num_events_in_wait_list */ 0,
            /* event_wait_list* */ NULL,
            /* event */ NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            if (err != nullptr) {
                *err = ciErrNum;
            }
            return numeric_limits<float>::signaling_NaN();
        }
    }

    // // Allocate internal buffers
    // //
    // uint32_t size = ACCUMULATOR_COUNT * ACCUMULATOR_SIZE * sizeof(cl_uint);
    // uint32_t *accumulators = (uint32_t*) malloc(size * sizeof(uint32_t));

    // // Retrieve internal buffers.
    // //
    // ciErrNum = clEnqueueReadBuffer(s_commandQueue, s_accumulators, CL_TRUE, 0, size, accumulators, 0, NULL, NULL);
    // if (ciErrNum != CL_SUCCESS) {
    //     printf("ciErrNum = %d\n", ciErrNum);
    //     printf("Error in clEnqueueReadBuffer Line %u in file %s !!!\n\n", __LINE__, __FILE__);
    //     exit(EXIT_FAILURE);
    // }

    // cout << "\naccumulators (before global merge):\n\n";

    // bool allzero = false;
    // for (int step = 0; step < 2; ++step) {
    //     for (int i = 0; i < ACCUMULATOR_COUNT; ++i) {
    //         allzero = true;
    //         for (int j = ACCUMULATOR_SIZE - 1; j >= 0 && allzero; --j) {
    //             if (accumulators[i * ACCUMULATOR_SIZE + j]) {
    //                 allzero = false;
    //             }
    //         }
    //         if (!allzero) {
    //             printf("[%05u]: ", i);
    //             if (step == 0) {
    //                 LongAccumulatorCPU cpuLacc (&accumulators[i * ACCUMULATOR_SIZE]);
    //                 cout << cpuLacc() << '\n';
    //             } else if (step == 1) {
    //                 for (int j = ACCUMULATOR_SIZE - 1; j >= 0; --j) {
    //                     printf("%u%c", accumulators[i * ACCUMULATOR_SIZE + j], j > 0 ? '\t' : '\n');
    //                 }
    //             }
    //         }
    //     }

    //     if (step == 0) {
    //         cout << "\n(LongAccumulator) accumulators:\n";
    //     }
    // }

    {
        local_work_size = MERGE_WORKGROUP_SIZE;
        global_work_size = local_work_size * (ACCUMULATOR_COUNT / MERGE_ACCUMULATOR_COUNT);

        cl_uint i = 0;
        ciErrNum = clSetKernelArg(s_clkMerge, i++, sizeof(cl_mem),  (void *)&s_accumulators);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            if (err != nullptr) {
                *err = ciErrNum;
            }
            return numeric_limits<float>::signaling_NaN();
        }

        ciErrNum = clEnqueueNDRangeKernel(
            s_commandQueue,
            s_clkMerge,
            /* work_dim */ 1,
            /* global_work_offset */ NULL,
            &global_work_size,
            &local_work_size,
            /* num_events_in_wait_list */ 0,
            /* event_wait_list* */ NULL,
            /* event */ NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("ciErrNum = %d\n", ciErrNum);
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            if (err != nullptr) {
                *err = ciErrNum;
            }
            return numeric_limits<float>::signaling_NaN();
        }
    }

    // // Retrieve internal buffers.
    // //
    // ciErrNum = clEnqueueReadBuffer(s_commandQueue, s_accumulators, CL_TRUE, 0, size, accumulators, 0, NULL, NULL);
    // if (ciErrNum != CL_SUCCESS) {
    //     printf("ciErrNum = %d\n", ciErrNum);
    //     printf("Error in clEnqueueReadBuffer Line %u in file %s !!!\n\n", __LINE__, __FILE__);
    //     exit(EXIT_FAILURE);
    // }

    // cout << "\naccumulators (after global merge part 1):\n\n";

    // allzero = false;
    // for (int step = 0; step < 2; ++step) {
    //     for (int i = 0; i < ACCUMULATOR_COUNT; ++i) {
    //         allzero = true;
    //         for (int j = ACCUMULATOR_SIZE - 1; j >= 0 && allzero; --j) {
    //             if (accumulators[i * ACCUMULATOR_SIZE + j]) {
    //                 allzero = false;
    //             }
    //         }
    //         if (!allzero) {
    //             printf("[%05u]: ", i);
    //             if (step == 0) {
    //                 LongAccumulatorCPU cpuLacc (&accumulators[i * ACCUMULATOR_SIZE]);
    //                 cout << cpuLacc() << '\n';
    //             } else if (step == 1) {
    //                 for (int j = ACCUMULATOR_SIZE - 1; j >= 0; --j) {
    //                     printf("%u%c", accumulators[i * ACCUMULATOR_SIZE + j], j > 0 ? '\t' : '\n');
    //                 }
    //             }
    //         }
    //     }

    //     if (step == 0) {
    //         cout << "\n(LongAccumulator) accumulators:\n";
    //     }
    // }

    {
        local_work_size = MERGE_WORKGROUP_SIZE;
        global_work_size = local_work_size; // only a single workgroup is needed

        cl_uint i = 0;
        ciErrNum  = clSetKernelArg(s_clkRound, i++, sizeof(cl_mem),  (void *)&s_accumulators);
        ciErrNum |= clSetKernelArg(s_clkRound, i++, sizeof(cl_mem),  (void *)&s_data_res);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            if (err != nullptr) {
                *err = ciErrNum;
            }
            return numeric_limits<float>::signaling_NaN();
        }

        ciErrNum = clEnqueueNDRangeKernel(
            s_commandQueue,
            s_clkRound,
            /* work_dim */ 1,
            /* global_work_offset */ NULL,
            &global_work_size,
            &local_work_size,
            /* num_events_in_wait_list */ 0,
            /* event_wait_list* */ NULL,
            /* event */ NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("ciErrNum = %d\n", ciErrNum);
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            if (err != nullptr) {
                *err = ciErrNum;
            }
            return numeric_limits<float>::signaling_NaN();
        }
    }

    // Retrieve result.
    //
    ciErrNum = clEnqueueReadBuffer(s_commandQueue, s_data_res, CL_TRUE, 0, sizeof(cl_float), &result, 0, NULL, NULL);
    if (ciErrNum != CL_SUCCESS) {
        printf("ciErrNum = %d\n", ciErrNum);
        printf("Error in clEnqueueReadBuffer Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        exit(EXIT_FAILURE);
    }

    // // Retrieve internal buffers.
    // //
    // ciErrNum = clEnqueueReadBuffer(s_commandQueue, s_accumulators, CL_TRUE, 0, size, accumulators, 0, NULL, NULL);
    // if (ciErrNum != CL_SUCCESS) {
    //     printf("ciErrNum = %d\n", ciErrNum);
    //     printf("Error in clEnqueueReadBuffer Line %u in file %s !!!\n\n", __LINE__, __FILE__);
    //     exit(EXIT_FAILURE);
    // }

    // cout << "\naccumulators (after global merge part 2):\n\n";

    // allzero = false;
    // for (int step = 0; step < 2; ++step) {
    //     for (int i = 0; i < ACCUMULATOR_COUNT; ++i) {
    //         allzero = true;
    //         for (int j = ACCUMULATOR_SIZE - 1; j >= 0 && allzero; --j) {
    //             if (accumulators[i * ACCUMULATOR_SIZE + j]) {
    //                 allzero = false;
    //             }
    //         }
    //         if (!allzero) {
    //             printf("[%05u]: ", i);
    //             if (step == 0) {
    //                 LongAccumulatorCPU cpuLacc (&accumulators[i * ACCUMULATOR_SIZE]);
    //                 cout << cpuLacc() << '\n';
    //             } else if (step == 1) {
    //                 for (int j = ACCUMULATOR_SIZE - 1; j >= 0; --j) {
    //                     printf("%u%c", accumulators[i * ACCUMULATOR_SIZE + j], j > 0 ? '\t' : '\n');
    //                 }
    //             }
    //         }
    //     }

    //     if (step == 0) {
    //         cout << "\n(LongAccumulator) accumulators:\n";
    //     }
    // }

    // free(accumulators);

    // Release memory...
    //
    if (s_data_arr) {
        ciErrNum = clReleaseMemObject(s_data_arr);
        if (ciErrNum != CL_SUCCESS) {
            cerr << "Error = " << ciErrNum << "\n";
            cerr << "Error in clReleaseMemObject, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        }
    }

    if (s_data_res) {
        ciErrNum = clReleaseMemObject(s_data_res);
        if (ciErrNum != CL_SUCCESS) {
            cerr << "Error = " << ciErrNum << "\n";
            cerr << "Error in clReleaseMemObject, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        }
    }

    if (err != nullptr) {
        *err = ciErrNum;
    }

    return result;
}