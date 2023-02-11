/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>

#include "parboil.h"
#include "file.h"
#include "gpu_info.h"
#include "ocl.h"
#include "convert_dataset.h"

#include "LongAccumulator.h"

using namespace std;

constexpr uint32_t DEFAULT_SEED = 1549813198;
constexpr uint32_t DEFAULT_NUMBER_OF_RUNS = 50;

constexpr char* input_files[] = { "bcsstk32.mtx", "fidapm05.mtx", "jgl009.mtx" };

int len = 1;
int depth = 1;
int dim = 1;
int pad = 32;
int nzcnt_len = 1;

const char *clSource[1];

// OpenCL specific
cl_int clStatus;
cl_platform_id clPlatform;
cl_device_id clDevice;
OpenCLDeviceProp clDeviceProp;
cl_context clContext;
cl_command_queue clCommandQueue;
cl_program clProgram;
cl_kernel clKernel;

// device memory allocation
// matrix
cl_mem d_data;
cl_mem d_indices;
cl_mem d_ptr;
cl_mem d_perm;

// vector
cl_mem d_Ax_vector;
cl_mem d_x_vector;

cl_mem jds_ptr_int;
cl_mem sh_zcnt_int;

bool generate_vector(float *x_vector, int dim, uint32_t seed)
{
    if (nullptr == x_vector)
    {
        return false;
    }

    mt19937 gen(seed);
    uniform_real_distribution<float> float_dist(0.f, 1.f);

    for (int i = 0; i < dim; i++)
    {
        x_vector[i] = float_dist(gen);
    }

    return true;
}

bool diff(int dim, float *h_Ax_vector_1, float *h_Ax_vector_2)
{
    for (int i = 0; i < dim; i++)
        if (h_Ax_vector_1[i] != h_Ax_vector_2[i])
            return true;
    return false;
}

void init_ocl()
{
    clStatus = clGetPlatformIDs(1, &clPlatform, NULL);
    CHECK_ERROR("clGetPlatformIDs")

    clStatus = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, 1, &clDevice, NULL);
    CHECK_ERROR("clGetDeviceIDs")

    clStatus = clGetDeviceInfo(clDevice, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &(clDeviceProp.multiProcessorCount), NULL);
    CHECK_ERROR("clGetDeviceInfo")

    cl_context_properties clCps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)clPlatform, 0};
    clContext = clCreateContextFromType(clCps, CL_DEVICE_TYPE_GPU, NULL, NULL, &clStatus);
    CHECK_ERROR("clCreateContextFromType")

    clCommandQueue = clCreateCommandQueue(clContext, clDevice, CL_QUEUE_PROFILING_ENABLE, &clStatus);
    CHECK_ERROR("clCreateCommandQueue")

    pb_SetOpenCL(&clContext, &clCommandQueue);

    clSource[0] = readFile("./kernel.cl");
    clProgram = clCreateProgramWithSource(clContext, 1, clSource, NULL, &clStatus);
    CHECK_ERROR("clCreateProgramWithSource")

    char clOptions[50];
    sprintf(clOptions, "");
    clStatus = clBuildProgram(clProgram, 1, &clDevice, clOptions, NULL, NULL);
    CHECK_ERROR("clBuildProgram")

    clKernel = clCreateKernel(clProgram, "spmv_jds_naive", &clStatus);
    CHECK_ERROR("clCreateKernel")

    // memory allocation
    d_data = clCreateBuffer(clContext, CL_MEM_READ_ONLY, len * sizeof(float), NULL, &clStatus);
    CHECK_ERROR("clCreateBuffer")
    d_indices = clCreateBuffer(clContext, CL_MEM_READ_ONLY, len * sizeof(int), NULL, &clStatus);
    CHECK_ERROR("clCreateBuffer")
    d_perm = clCreateBuffer(clContext, CL_MEM_READ_ONLY, dim * sizeof(int), NULL, &clStatus);
    CHECK_ERROR("clCreateBuffer")
    d_x_vector = clCreateBuffer(clContext, CL_MEM_READ_ONLY, dim * sizeof(float), NULL, &clStatus);
    CHECK_ERROR("clCreateBuffer")
    d_Ax_vector = clCreateBuffer(clContext, CL_MEM_WRITE_ONLY, dim * sizeof(float), NULL, &clStatus);
    CHECK_ERROR("clCreateBuffer")

    jds_ptr_int = clCreateBuffer(clContext, CL_MEM_READ_ONLY, 5000 * sizeof(int), NULL, &clStatus);
    CHECK_ERROR("clCreateBuffer")
    sh_zcnt_int = clCreateBuffer(clContext, CL_MEM_READ_ONLY, 5000 * sizeof(int), NULL, &clStatus);
    CHECK_ERROR("clCreateBuffer")
}

void cleanup_ocl()
{
    clStatus = clReleaseKernel(clKernel);
    clStatus |= clReleaseProgram(clProgram);

    clStatus |= clReleaseMemObject(d_data);
    clStatus |= clReleaseMemObject(d_indices);
    clStatus |= clReleaseMemObject(d_perm);
    clStatus |= clReleaseMemObject(d_x_vector);
    clStatus |= clReleaseMemObject(d_Ax_vector);
    CHECK_ERROR("clReleaseMemObject")

    clStatus |= clReleaseCommandQueue(clCommandQueue);
    clStatus |= clReleaseContext(clContext);

    free((void *)clSource[0]);

    CHECK_ERROR("Failed to release OpenCL resources!\n")
}

void spmv_ocl(int dim, int *h_nzcnt, int *h_ptr, int *h_indices, float *h_data,
              float *h_x_vector, int *h_perm, float *h_Ax_vector)
{
    // cleanup memory
    clMemSet(clCommandQueue, d_Ax_vector, 0, dim * sizeof(float));

    // memory copy
    clStatus = clEnqueueWriteBuffer(clCommandQueue, d_data, CL_FALSE, 0, len * sizeof(float), h_data, 0, NULL, NULL);
    CHECK_ERROR("clEnqueueWriteBuffer")
    clStatus = clEnqueueWriteBuffer(clCommandQueue, d_indices, CL_FALSE, 0, len * sizeof(int), h_indices, 0, NULL, NULL);
    CHECK_ERROR("clEnqueueWriteBuffer")
    clStatus = clEnqueueWriteBuffer(clCommandQueue, d_perm, CL_FALSE, 0, dim * sizeof(int), h_perm, 0, NULL, NULL);
    CHECK_ERROR("clEnqueueWriteBuffer")
    clStatus = clEnqueueWriteBuffer(clCommandQueue, d_x_vector, CL_FALSE, 0, dim * sizeof(int), h_x_vector, 0, NULL, NULL);
    CHECK_ERROR("clEnqueueWriteBuffer")

    clStatus = clEnqueueWriteBuffer(clCommandQueue, jds_ptr_int, CL_FALSE, 0, depth * sizeof(int), h_ptr, 0, NULL, NULL);
    CHECK_ERROR("clEnqueueWriteBuffer")
    clStatus = clEnqueueWriteBuffer(clCommandQueue, sh_zcnt_int, CL_TRUE, 0, nzcnt_len * sizeof(int), h_nzcnt, 0, NULL, NULL);
    CHECK_ERROR("clEnqueueWriteBuffer")

    size_t grid;
    size_t block;

    compute_active_thread(&block, &grid, nzcnt_len, pad, clDeviceProp.major, clDeviceProp.minor, clDeviceProp.multiProcessorCount);
    //  printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!grid is %d and block is %d=\n",grid,block);
    //  printf("!!! dim is %d\n",dim);

    clStatus = clSetKernelArg(clKernel, 0, sizeof(cl_mem), &d_Ax_vector);
    CHECK_ERROR("clSetKernelArg")
    clStatus = clSetKernelArg(clKernel, 1, sizeof(cl_mem), &d_data);
    CHECK_ERROR("clSetKernelArg")
    clStatus = clSetKernelArg(clKernel, 2, sizeof(cl_mem), &d_indices);
    CHECK_ERROR("clSetKernelArg")
    clStatus = clSetKernelArg(clKernel, 3, sizeof(cl_mem), &d_perm);
    CHECK_ERROR("clSetKernelArg")
    clStatus = clSetKernelArg(clKernel, 4, sizeof(cl_mem), &d_x_vector);
    CHECK_ERROR("clSetKernelArg")
    clStatus = clSetKernelArg(clKernel, 5, sizeof(int), &dim);
    CHECK_ERROR("clSetKernelArg")

    clStatus = clSetKernelArg(clKernel, 6, sizeof(cl_mem), &jds_ptr_int);
    CHECK_ERROR("clSetKernelArg")
    clStatus = clSetKernelArg(clKernel, 7, sizeof(cl_mem), &sh_zcnt_int);
    CHECK_ERROR("clSetKernelArg")

    // main execution
    clStatus = clEnqueueNDRangeKernel(clCommandQueue, clKernel, 1, NULL, &grid, &block, 0, NULL, NULL);
    CHECK_ERROR("clEnqueueNDRangeKernel")

    clStatus = clFinish(clCommandQueue);
    CHECK_ERROR("clFinish")

    // HtoD memory copy
    clStatus = clEnqueueReadBuffer(clCommandQueue, d_Ax_vector, CL_TRUE, 0, dim * sizeof(float), h_Ax_vector, 0, NULL, NULL);
    CHECK_ERROR("clEnqueueReadBuffer")
}

void spmv_reproducible_sum(int dim, int *h_nzcnt, int *h_ptr, int *h_indices, float *h_data,
                       float *h_x_vector, int *h_perm, float *h_Ax_vector)
{
    float* products = (float*) malloc(dim * sizeof(float));

    // Consider creating a random map by creating an array 0..dim - 1 and randomly shuffling it
    // for each execution. This should provide required randomness given the order of operations
    // is sequential at the moment.
    //
    for (int i = 0; i < dim; i++) {
        int bound = h_nzcnt[i];

        for (int k = 0; k < bound; k++) {
            int j = h_ptr[k] + i;
            int in = h_indices[j];

            float d = h_data[j];
            float t = h_x_vector[in];

            products[k] = d * t;
        }

        if (bound > 0)
        {
            h_Ax_vector[h_perm[i]] = LongAccumulator::Sum(bound, products);
        }
    }

    free(products);
}

void spmv_reproducible_dot(int dim, int *csr_ptr, int *csr_indices, float *csr_data,
                       float *h_x_vector, float *h_Ax_vector)
{
    float* sel_x_vector = (float*) malloc(dim * sizeof(float));

    for (int i = 0; i < dim; i++) {
        int nz_cnt_in_row = csr_ptr[i + 1] - csr_ptr[i];

        if (nz_cnt_in_row <= 0)
        {
            continue;
        }

        for (int j = 0; j < nz_cnt_in_row; ++j) {
            sel_x_vector[j] = h_x_vector[csr_indices[j + csr_ptr[i]]];
        }

        h_Ax_vector[i] = LongAccumulator::DotProduct(nz_cnt_in_row, csr_data + csr_ptr[i], sel_x_vector);
    }

    free(sel_x_vector);
}

void spmv_reproducible_spmv(int dim, int *csr_ptr, int *csr_indices, float *csr_data,
                       float *h_x_vector, float *h_Ax_vector)
{
    LongAccumulator::SparseMatrixDenseVectorProduct(dim, csr_data, csr_indices, csr_ptr, h_x_vector, h_Ax_vector);
}

void execute(bool reproducible, int dim, int *h_nzcnt, int *h_ptr, int *h_indices, float *h_data,
             float *h_x_vector, int *h_perm, float *h_Ax_vector,
             uint64_t &time_setup, uint64_t &time_run)
{
    chrono::steady_clock::time_point start;

    start = chrono::steady_clock::now();
    if (reproducible) {
        LongAccumulator::InitializeOpenCL();
    } else {
        init_ocl();
    }
    time_setup = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();

    start = chrono::steady_clock::now();
    if (reproducible) {
        spmv_reproducible_spmv(dim, h_ptr, h_indices, h_data, h_x_vector, h_Ax_vector);
    } else {
        spmv_ocl(dim, h_nzcnt, h_ptr, h_indices, h_data, h_x_vector, h_perm, h_Ax_vector);
    }
    time_run = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();

    start = chrono::steady_clock::now();
    if (reproducible) {
        LongAccumulator::CleanupOpenCL();
    } else {
        cleanup_ocl();
    }
    time_setup += chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();
}

int main(int argc, char **argv)
{
    int rows, cols, nz;
    mat_entry* entries;

    // host memory allocation
    // matrix
    float *h_data;
    int *h_indices;
    int *h_ptr;
    int *h_perm;
    int *h_nzcnt;
    float *csr_data;
    int *csr_ptr;
    int *csr_indices;
    // vector
    float *h_Ax_vector, *h_Ax_vector_rep;
    float *h_x_vector;

    const int exe_path_len = strrchr(argv[0], '/') - argv[0] + 1;
    char exe_path[256];
    strncpy(exe_path, argv[0], exe_path_len);
    exe_path[exe_path_len] = '\0';

    char input_file_path[256];

    cout << "unit: [ms]\n\n";

    int col_count;

    uint64_t time_setup[3], time_run[3];
    uint64_t time_setup_rep[3], time_run_rep[3];

    chrono::steady_clock::time_point start;
    uint64_t time_first_setup = 0;

    start = chrono::steady_clock::now();
    init_ocl();
    time_first_setup =
        chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();

    start = chrono::steady_clock::now();
    cleanup_ocl();
    time_first_setup +=
        chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();

    cout << "OCL first (dummy) setup time : " << fixed << setprecision(10) << (float) time_first_setup / 1000.0 << endl << endl;
    
    cout << "The following numbers represent setup and run times for each run:" << endl;
    cout << "The first line contains setup times for each run." << endl;
    cout << "The second line contains run times for each run." << endl;

    for (int i = 0; i < 3; ++i)
    {
        strncpy(input_file_path, exe_path, exe_path_len + 1);
        strcat(input_file_path, "data/");
        strcat(input_file_path, input_files[i]);

        cout << '\n' << input_files[i] << "\n\n";

        // Load matrix from file
        //
        read_coo_from_mtx_file(
            input_file_path, // bcsstk32.mtx, fidapm05.mtx, jgl009.mtx
            &dim,
            &rows,
            &cols,
            &nz,
            &entries);

        coo_to_jds(
            rows,
            cols,
            nz,
            entries,
            1,          // row padding
            pad,        // warp size
            1,          // pack size
            0,          // debug level [0:2]
            &h_data, &h_ptr, &h_nzcnt, &h_indices, &h_perm,
            &col_count, &len, &nzcnt_len, &depth);

        coo_to_csr(
            rows,
            cols,
            nz,
            entries,
            0,          // debug level [0:2]
            &csr_data, &csr_ptr, &csr_indices);

        h_x_vector = new float[dim];

        if (!generate_vector(h_x_vector, dim, DEFAULT_SEED)) {
            fprintf(stderr, "Failed to generate dense vector.\n");
            exit(-1);
        }

        h_Ax_vector = (float*) calloc(dim, sizeof(float));
        h_Ax_vector_rep = (float*) calloc(dim, sizeof(float));

        cout << "\nnon-reproducible\n\n";

        for (int run = 0; run < 3; ++run) execute(false, dim, h_nzcnt, h_ptr, h_indices, h_data, h_x_vector, h_perm, h_Ax_vector, time_setup[run], time_run[run]);

        for (int run = 0; run < 3; ++run) {
            cout << fixed << setprecision(10) << (float) time_setup[run] / 1000.0 << (run < 2 ? '\t' : '\n');
        }
        for (int run = 0; run < 3; ++run) {
            cout << fixed << setprecision(10) << (float) time_run[run] / 1000.0 << (run < 2 ? '\t' : '\n');
        }

        cout << "\nreproducible\n\n";

        for (int run = 0; run < 3; ++run) execute(true, dim, nullptr, csr_ptr, csr_indices, csr_data, h_x_vector, nullptr, h_Ax_vector_rep, time_setup_rep[run], time_run_rep[run]);

        for (int run = 0; run < 3; ++run) {
            cout << fixed << setprecision(10) << (float) time_setup_rep[run] / 1000.0 << (run < 2 ? '\t' : '\n');
        }
        for (int run = 0; run < 3; ++run) {
            cout << fixed << setprecision(10) << (float) time_run_rep[run] / 1000.0 << (run < 2 ? '\t' : '\n');
        }

        delete[] h_data;
        delete[] h_indices;
        delete[] h_ptr;
        delete[] h_perm;
        delete[] h_nzcnt;
        
        delete[] csr_data;
        delete[] csr_ptr;
        delete[] csr_indices;

        delete[] h_x_vector;

        free(h_Ax_vector);
        free(h_Ax_vector_rep);

        delete[] entries;
    }

    return 0;
}