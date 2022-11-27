#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <pthread.h>
#include <random>
#include <thread>
#include <vector>

#include "common.h"

using namespace std;

// Floating-point specification constants

constexpr int EXPONENT_MIN_VALUE = -126;
constexpr int EXPONENT_MAX_VALUE = 127;
constexpr int EXPONENT_BIAS = 127;

// Default values

constexpr int DEFAULT_THREAD_COUNT = 8;
constexpr int DEFAULT_ELEMENT_COUNT = 1000;
constexpr uint32_t DEFAULT_SEED = 1549813198;
constexpr int DEFAULT_EXPONENT_MIN_VALUE = -10;
constexpr int DEFAULT_EXPONENT_MAX_VALUE = 10;
constexpr int DEFAULT_REPEAT_COUNT = 100;

// GPU kernel constants

// Size of a single work unit (warp).
//
constexpr uint32_t WARP_SIZE = 16;

// For summation we use 16 warps of 16 threads for a total of 256 threads in a workgroup.
//
constexpr uint32_t WARP_COUNT = 16;
constexpr uint32_t WORKGROUP_SIZE = WARP_SIZE * WARP_COUNT;
constexpr uint32_t WORKGROUP_COUNT = 16;

#define OPENCL_PROGRAM_FILENAME "Sum.cl"
#define SUM_KERNEL_NAME "sum"

#ifdef AMD
constexpr char compileOptionsFormat[256] = "-DWARP_SIZE=%u -DWARP_COUNT=%u -DUSE_KNUTH";
#else
constexpr char compileOptionsFormat[256] = "-DWARP_SIZE=%u -DWARP_COUNT=%u -DUSE_KNUTH -DNVIDIA -cl-mad-enable -cl-fast-relaxed-math";
#endif

// Program parameters

int thread_count = DEFAULT_THREAD_COUNT;
int element_count = DEFAULT_ELEMENT_COUNT;
uint32_t seed = DEFAULT_SEED;
int exponent_min = DEFAULT_EXPONENT_MIN_VALUE;
int exponent_max = DEFAULT_EXPONENT_MAX_VALUE;
int repeat_count = DEFAULT_REPEAT_COUNT;
bool print_elements = false;
bool perf_test = false;

// Shared variables

vector<float> *elements;
default_random_engine *shuffle_engine;

bool is_opencl_initialized = false;

cl_platform_id platform = nullptr;
cl_device_id device = nullptr;
cl_context context = nullptr;
cl_command_queue command_queue = nullptr;

cl_program program = nullptr;
cl_kernel kernel = nullptr;

cl_mem data_arr = nullptr;
cl_mem data_res = nullptr;

// Results

float sum_sequential, sum_gpu;
uint64_t time_sequential, time_gpu;

void print_usage(char program_name[])
{
    cout << "Usage: " << program_name << " <options>" << endl;
    cout << "  -t <num>: Uses <num> threads for parallel execution (default value: 8)" << endl;
    cout << "  -n <num>: Generates <num> random floating-point numbers for analysis (default value: 1000)" << endl;
    cout << "  -s <num>: Generates the numbers using <num> as seed (default value: 1549813198)" << endl;
    cout << "  -l <num>: Uses <num> as exponent minimum value (default value: -10)" << endl;
    cout << "  -h <num>: Uses <num> as exponent maximum value (inclusive) (default value: 10)" << endl;
    cout << "  -r <num>: Repeats the execution of both implementations <num> times for reproducibility study (default value: 100)" << endl;
    cout << "  -p: Print generated numbers before analysis" << endl;
    cout << "  -!: Split elements evenly between threads for performance testing" << endl;
    cout << "  -?: Print this message" << endl;
}

void parse_parameters(int argc, char *argv[])
{
    for (int i = 1; i < argc; ++i) {
        int arglen = strlen(argv[i]);
        if (arglen < 2 || argv[i][0] != '-') {
            cout << "Invalid argument: \"" << argv[i] << '"' << endl;
            print_usage(argv[0]);
            exit(EXIT_FAILURE);
        }
        int n;
        switch (argv[i][1]) {
        case 't':
            if (i + 1 == argc) {
                cout << "Thread count not specified!" << endl;
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
            }
            n = atoi(argv[++i]);
            if (n < 1) {
                cout << "Invalid thread count: " << n << "! Using default value: " << DEFAULT_THREAD_COUNT << endl;
            } else {
                thread_count = n;
            }
            break;
        case 'n':
            if (i + 1 == argc) {
                cout << "Element count not specified!" << endl;
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
            }
            n = atoi(argv[++i]);
            if (n < 1) {
                cout << "Invalid element count: " << n << "! Using default value: " << DEFAULT_ELEMENT_COUNT << endl;
            } else {
                element_count = n;
            }
            break;
        case 's':
            if (i + 1 == argc) {
                cout << "Seed not specified!" << endl;
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
            }
            n = atoi(argv[++i]);
            if (n == 0) {
                random_device rd;
                seed = rd();
                cout << "Special seed value: 0. Generated random seed: " << seed << endl;
            } else {
                seed = n;
            }
            break;
        case 'l':
            if (i + 1 == argc) {
                cout << "Exponent minimum value not specified!" << endl;
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
            }
            n = atoi(argv[++i]);
            if (n < EXPONENT_MIN_VALUE) {
                cout << "Invalid exponent minimum value: " << n
                     << "! Using default value: " << DEFAULT_EXPONENT_MIN_VALUE << endl;
            } else {
                exponent_min = n;
            }
            break;
        case 'h':
            if (i + 1 == argc) {
                cout << "Exponent maximum value not specified!" << endl;
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
            }
            n = atoi(argv[++i]);
            if (n > EXPONENT_MAX_VALUE) {
                cout << "Invalid exponent maximum value: " << n
                     << "! Using default value: " << DEFAULT_EXPONENT_MAX_VALUE << endl;
            } else {
                exponent_max = n;
            }
            break;
        case 'r':
            if (i + 1 == argc) {
                cout << "Repeat count not specified!" << endl;
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
            }
            n = atoi(argv[++i]);
            if (n < 0) {
                cout << "Invalid repeat count: " << n << "! Using default value: " << DEFAULT_REPEAT_COUNT << endl;
            } else {
                repeat_count = n;
            }
            break;
        case 'p':
            print_elements = true;
            break;
        case '!':
            perf_test = true;
            break;
        case '?':
            print_usage(argv[0]);
            exit(EXIT_SUCCESS);
        default:
            cout << "Invalid option: \"" << argv[i] << "\"!" << endl;
            exit(EXIT_FAILURE);
        }
    }
    if (thread_count > element_count) {
        cout << "There are more threads available than there are elements to process. Reducing thread count to element count." << endl;
        thread_count = element_count;
    }
}

void print_parameters()
{
    cout << "Parameters:" << endl;
    cout << "  Thread count:             " << thread_count << endl;
    cout << "  Element count:            " << element_count << endl;
    cout << "  Seed:                     " << seed << endl;
    cout << "  Exponent minimum value:   " << exponent_min << endl;
    cout << "  Exponent maximum value:   " << exponent_max << endl;
    cout << "  Repeat count:             " << repeat_count << endl;
    cout << "  Print elements:           " << (print_elements ? "YES" : "NO") << endl;
    cout << "  Performance test:         " << (perf_test ? "YES" : "NO") << endl;
}

void generate_elements()
{
    mt19937 gen(seed);
    uniform_int_distribution<uint32_t> sign_dist(0, 1);
    // uniform_int_distribution<uint32_t> exponent_dist(0, (1 << 8) - 1);
    uniform_int_distribution<uint32_t> exponent_dist(exponent_min + EXPONENT_BIAS, exponent_max + EXPONENT_BIAS);
    uniform_int_distribution<uint32_t> mantisa_dist(0, (1 << 23) - 1);

    elements = new vector<float>(element_count);

    int positive_count = 0, negative_count = 0;
    for (int i = 0; i < element_count; ++i) {
        uint32_t bits = (sign_dist(gen) << 31) | (exponent_dist(gen) << 23) | (mantisa_dist(gen) << 0);
        static_assert(sizeof(uint32_t) == sizeof(float));
        float number;
        memcpy(&number, &bits, sizeof(uint32_t));
        (*elements)[i] = number;
        if (print_elements) {
            cout << i + 1 << ". element: " << fixed << setprecision(10) << (*elements)[i] << " (" << scientific
                 << setprecision(10) << (*elements)[i] << ')' << endl;
            if ((*elements)[i] > 0) {
                positive_count++;
            } else {
                negative_count++;
            }
        }
    }

    if (print_elements) {
        cout << "Number of positive elements: " << positive_count << endl;
        cout << "Number of negative elements: " << negative_count << endl;
    }

    cout << "Successfully generated " << element_count << " random floating-point numbers." << endl;
}

void run_sequential()
{
    chrono::steady_clock::time_point start;
    float sum;
    for (int run_idx = 0; run_idx <= repeat_count; ++run_idx) {
        if (run_idx == 0) {
            start = chrono::steady_clock::now();
        }
        sum = 0;
        for (int i = 0; i < element_count; ++i) {
            // NOTE: This operation results in non-reproducible results. The reason is that
            //       element order is shuffled after every repetition hence rounding errors
            //       and other inherent errors with floating-point arithmetic occur.
            sum += (*elements)[i];
        }
        if (run_idx == 0) {
            sum_sequential = sum;
            time_sequential = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();
            cout << "Sequential sum: " << fixed << setprecision(10) << sum_sequential << " (" << scientific
                 << setprecision(10) << sum_sequential << ')' << endl;
        } else if (sum != sum_sequential) {
            cout << "Sequential sum not reproducible after " << run_idx << " runs!" << endl;
            break;
        }
        if (run_idx < repeat_count) {
            // Shuffle element vector to incur variability
            shuffle(elements->begin(), elements->end(), *shuffle_engine);
        }
    }
}

void initialize_opencl()
{
    if (is_opencl_initialized) {
        return;
    }

    cl_int ciErrNum;

    char platform_name[64];

#ifdef AMD
    strcpy(platform_name, "AMD Accelerated Parallel Processing");
#else
    strcpy(platform_name, "NVIDIA CUDA");
#endif

    platform = GetOCLPlatform(platform_name);
    if (platform == NULL) {
        cerr << "ERROR: Failed to find the platform '" << platform_name << "' ...\n";
        return;
    }

    // Get a GPU device
    //
    device = GetOCLDevice(platform);
    if (device == NULL) {
        cerr << "Error in clGetDeviceIDs, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        return;
    }

    // Create the context
    //
    context = clCreateContext(0, 1, &device, NULL, NULL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        cerr << "Error in clCreateContext, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        return;
    }

    // Create a command-queue
    //
    command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        cerr << "Error = " << ciErrNum << "\n";
        cerr << "Error in clCreateCommandQueue, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        return;
    }

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
        return;
    }
    fseek(pProgramHandle, 0, SEEK_END);
    sourceCodeLength = ftell(pProgramHandle);
    rewind(pProgramHandle);
    sourceCode = (char *) malloc(sourceCodeLength + 1);
    sourceCode[sourceCodeLength] = '\0';
    if (fread(sourceCode, sizeof(char), sourceCodeLength, pProgramHandle) != sourceCodeLength)
    {
        cerr << "Failed to read source code.\n";
        return;
    }
    fclose(pProgramHandle);

    // Create OpenCL program from source
    //
    program = clCreateProgramWithSource(context, 1, (const char **)&sourceCode, &sourceCodeLength, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        cerr << "Error = " << ciErrNum << "\n";
        cerr << "Error in clCreateProgramWithSource, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        return;
    }

    // Building Sum program
    //
    char compileOptions[256];
    snprintf(compileOptions, 256, compileOptionsFormat, WARP_SIZE, WARP_COUNT);
    ciErrNum = clBuildProgram(program, 0, NULL, compileOptions, NULL, NULL);
    if (ciErrNum != CL_SUCCESS) {
        cerr << "Error = " << ciErrNum << "\n";
        cerr << "Error in clBuildProgram, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";

        // Determine the reason for the error
        //
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), &buildLog, NULL);

        cout << buildLog << "\n";
        return;
    }

    // Create kernels
    //
    kernel = clCreateKernel(program, SUM_KERNEL_NAME, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        cerr << "Error = " << ciErrNum << "\n";
        cerr << "Error in clCreateKernel: " SUM_KERNEL_NAME ", Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        return;
    }

    // Allocating global OpenCL memory
    //
    data_arr = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, element_count * sizeof(cl_float), elements->data(), &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        cerr << "Error in clCreateBuffer for data_arr, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        return;
    }
    data_res = clCreateBuffer(context, CL_MEM_READ_WRITE, WORKGROUP_COUNT * sizeof(cl_float), NULL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        cerr <<"Error in clCreateBuffer for data_res, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        return;
    }

    // Deallocate temp storage
    //
    free(sourceCode);

    is_opencl_initialized = true;
}

void cleanup_opencl()
{
    if (!is_opencl_initialized) {
        return;
    }

    cl_int ciErrNum;

    // Release memory...
    //
    if (data_arr) {
        ciErrNum = clReleaseMemObject(data_arr);
        if (ciErrNum != CL_SUCCESS) {
            cerr << "Error = " << ciErrNum << "\n";
            cerr << "Error in clReleaseMemObject, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        }
    }

    if (data_res) {
        ciErrNum = clReleaseMemObject(data_res);
        if (ciErrNum != CL_SUCCESS) {
            cerr << "Error = " << ciErrNum << "\n";
            cerr << "Error in clReleaseMemObject, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        }
    }

    if (kernel) {
        ciErrNum = clReleaseKernel(kernel);
        if (ciErrNum != CL_SUCCESS) {
            cerr << "Error = " << ciErrNum << "\n";
            cerr << "Error in clReleaseKernel: " SUM_KERNEL_NAME ", Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        }
    }

    if (program) {
        ciErrNum = clReleaseProgram(program);
        if (ciErrNum != CL_SUCCESS) {
            cerr << "Error = " << ciErrNum << "\n";
            cerr << "Error in clReleaseProgram, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        }
    }

    // Shutting down and freeing memory...
    //
    if (command_queue) {
        clReleaseCommandQueue(command_queue);
    }
    if (context) {
        clReleaseContext(context);
    }

    is_opencl_initialized = false;
}

float compute_sum_gpu()
{
    cl_int ciErrNum;
    size_t global_work_size, local_work_size;

    local_work_size = WORKGROUP_SIZE;
    global_work_size = local_work_size * WORKGROUP_COUNT;

    cl_uint i = 0;
    ciErrNum  = clSetKernelArg(kernel, i++, sizeof(cl_uint), (void *)&element_count);
    ciErrNum |= clSetKernelArg(kernel, i++, sizeof(cl_mem),  (void *)&data_arr);
    ciErrNum |= clSetKernelArg(kernel, i++, sizeof(cl_mem),  (void *)&data_res);
    if (ciErrNum != CL_SUCCESS) {
        printf("ciErrNum = %d\n", ciErrNum);
        printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        return numeric_limits<float>::signaling_NaN();
    }

    ciErrNum = clEnqueueNDRangeKernel(
        command_queue,
        kernel,
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
        return numeric_limits<float>::signaling_NaN();
    }

    float partial_sums[WORKGROUP_COUNT];

    // Retrieve result.
    //
    ciErrNum = clEnqueueReadBuffer(command_queue, data_res, CL_TRUE, 0, WORKGROUP_COUNT * sizeof(cl_float), &partial_sums, 0, NULL, NULL);
    if (ciErrNum != CL_SUCCESS) {
        printf("ciErrNum = %d\n", ciErrNum);
        printf("Error in clEnqueueReadBuffer Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        exit(EXIT_FAILURE);
    }

    float sum = 0;
    for (int i = 0; i < WORKGROUP_COUNT; ++i) {
        // printf("%d: %f\n", i, partial_sums[i]);
        sum += partial_sums[i];
    }

    return sum;
}

void run_gpu()
{
    initialize_opencl();

    chrono::steady_clock::time_point start;
    float sum;
    for (int run_idx = 0; run_idx <= repeat_count; ++run_idx) {
        if (run_idx == 0) {
            start = chrono::steady_clock::now();
        }
        sum = compute_sum_gpu();
        if (run_idx == 0) {
            sum_gpu = sum;
            time_gpu = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();
            cout << "GPU sum: " << fixed << setprecision(10) << sum_gpu << " (" << scientific
                 << setprecision(10) << sum_gpu << ')' << endl;
        } else if (sum != sum_gpu) {
            cout << "GPU sum not reproducible after " << run_idx << " runs!" << endl;
            break;
        }
        if (run_idx < repeat_count) {
            // Shuffle element vector to incur variability
            shuffle(elements->begin(), elements->end(), *shuffle_engine);
        }
    }

    cleanup_opencl();
}

void cleanup()
{
    delete shuffle_engine;
    delete elements;
}

int main(int argc, char *argv[])
{
    parse_parameters(argc, argv);
    print_parameters();

    generate_elements();

    shuffle_engine = new default_random_engine(seed);

    run_sequential();
    run_gpu();

    cout << "Sequential execution time: " << time_sequential << " [us] (" << fixed << setprecision(10)
         << (float) time_sequential / 1000.0 << " [ms])" << endl;
    cout << "GPU execution time: " << time_gpu << " [us] (" << fixed << setprecision(10)
         << (float) time_gpu / 1000.0 << " [ms])" << endl;
    cout << "Speedup: " << fixed << setprecision(10) << ((float) time_sequential) / ((float) time_gpu) << endl;

    cleanup();

    return EXIT_SUCCESS;
}