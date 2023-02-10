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

struct pb_TimerSet timers;

int len;
int depth;
int dim;
int pad = 32;
int nzcnt_len;

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
        if (h_Ax_vector_1[i] != h_Ax_vector_2[i]) {
            cout << "Difference found on " << i + 1 << ". element: " << h_Ax_vector_1[i] << " : " << h_Ax_vector_2[i] << '\n';
            return true;
        }
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
    pb_SwitchToTimer(&timers, pb_TimerID_COPY);

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

    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

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
    pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);

    clStatus = clEnqueueNDRangeKernel(clCommandQueue, clKernel, 1, NULL, &grid, &block, 0, NULL, NULL);
    CHECK_ERROR("clEnqueueNDRangeKernel")

    clStatus = clFinish(clCommandQueue);
    CHECK_ERROR("clFinish")

    pb_SwitchToTimer(&timers, pb_TimerID_COPY);
    // HtoD memory copy
    clStatus = clEnqueueReadBuffer(clCommandQueue, d_Ax_vector, CL_TRUE, 0, dim * sizeof(float), h_Ax_vector, 0, NULL, NULL);
    CHECK_ERROR("clEnqueueReadBuffer")

    pb_SwitchToTimer(&timers, pb_TimerID_NONE);
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

        if (bound <= 0)
        {
            h_Ax_vector[h_perm[i]] = 0.0f;
        }
        else
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
            h_Ax_vector[i] = 0.0f;
            continue;
        }

        for (int j = 0; j < nz_cnt_in_row; ++j) {
            sel_x_vector[j] = h_x_vector[csr_indices[j + csr_ptr[i]]];
        }

        cout << i << ": nz_cnt_in_row = " << nz_cnt_in_row << '\n';

        h_Ax_vector[i] = LongAccumulator::DotProduct(nz_cnt_in_row, csr_data + csr_ptr[i], sel_x_vector);
    }

    free(sel_x_vector);
}

void spmv_reproducible_spmv(int dim, int *csr_ptr, int *csr_indices, float *csr_data,
                       float *h_x_vector, float *h_Ax_vector)
{
    LongAccumulator::SparseMatrixDenseVectorProduct(dim, csr_data, csr_indices, csr_ptr, h_x_vector, h_Ax_vector);
}

void execute(uint32_t nruns, bool reproducible,
             int dim, int *h_nzcnt, int *h_ptr, int *h_indices, float *h_data,
             float *h_x_vector, int *h_perm, float *h_Ax_vector, uint64_t &time)
{
    if (!reproducible)
    {
        cout << "Running non-reproducible implementation...\n";
        init_ocl();
    }
    else
    {
        cout << "Running reproducible implementation...\n";
        LongAccumulator::InitializeOpenCL();
    }

    chrono::steady_clock::time_point start;

    float *tmp_h_Ax_vector = new float[dim];

    for (int i = 0; i < nruns; ++i)
    {
        if (i == 0)
        {
            start = chrono::steady_clock::now();
        }
        
        if (reproducible)
        {
            spmv_reproducible_spmv(dim, h_ptr, h_indices, h_data, h_x_vector, tmp_h_Ax_vector);
        }
        else
        {
            spmv_ocl(dim, h_nzcnt, h_ptr, h_indices, h_data, h_x_vector, h_perm, tmp_h_Ax_vector);
        }

        if (i == 0)
        {
            time = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();
            memcpy(h_Ax_vector, tmp_h_Ax_vector, dim * sizeof(float));
        }
        else if (diff(dim, h_Ax_vector, tmp_h_Ax_vector))
        {
            printf("%seproducible implementation not reproducible after %d runs!\n",
                   reproducible ? "R" : "Non-r", i);
            break;
        }
    }

    delete[] tmp_h_Ax_vector;

    if (!reproducible)
    {
        cleanup_ocl();
    }
    else
    {
        LongAccumulator::CleanupOpenCL();
    }
}

int main(int argc, char **argv)
{
    struct pb_Parameters *parameters;

    printf("OpenCL accelerated sparse matrix vector multiplication****\n");
    printf("Original version by Li-Wen Chang <lchang20@illinois.edu> and Shengzhao Wu<wu14@illinois.edu>\n");
    printf("This version maintained by Chris Rodrigues  ***********\n");

    parameters = pb_ReadParameters(&argc, argv);
    if (parameters->inpFiles[0] == NULL)
    {
        fprintf(stderr, "Expecting one input filename\n");
        exit(-1);
    }

    pb_InitializeTimerSet(&timers);
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

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

    // Load matrix from file
    //
    pb_SwitchToTimer(&timers, pb_TimerID_IO);

    read_coo_from_mtx_file(
        parameters->inpFiles[0], // bcsstk32.mtx, fidapm05.mtx, jgl009.mtx
        &dim,
        &rows,
        &cols,
        &nz,
        &entries);

    int col_count;
    
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    coo_to_jds(
        rows,
        cols,
        nz,
        entries,
        1,          // row padding
        pad,        // warp size
        1,          // pack size
        1,          // debug level [0:2]
        &h_data, &h_ptr, &h_nzcnt, &h_indices, &h_perm,
        &col_count, &len, &nzcnt_len, &depth);

    // cout << "col_count = " << col_count << " len = " << len << " nzcnt_len = " << nzcnt_len << " depth = " << depth << "\n\n";

    // cout << "h_data:\t\t";
    // for (int i = 0; i < nz; ++i) {
    //     cout << h_data[i] << '\t';
    // }
    // cout << '\n';

    // cout << "h_indices:\t";
    // for (int i = 0; i < nz; ++i) {
    //     cout << h_indices[i] << '\t';
    // }
    // cout << '\n';

    // cout << "h_nzcnt:\t";
    // for (int i = 0; i < dim; ++i) {
    //     cout << h_nzcnt[i] << '\t';
    // }
    // cout << '\n';

    // cout << "h_ptr:\t\t";
    // for (int i = 0; i < depth; ++i) {
    //     cout << h_ptr[i] << '\t';
    // }
    // cout << '\n';

    // cout << "h_perm:\t\t";
    // for (int i = 0; i < dim; ++i) {
    //     cout << h_perm[i] << '\t';
    // }
    // cout << '\n';

    coo_to_csr(
        rows,
        cols,
        nz,
        entries,
        1,
        &csr_data, &csr_ptr, &csr_indices);

    // cout << "csr_data:\t";
    // for (int i = 0; i < nz; ++i) {
    //     cout << csr_data[i] << '\t';
    // }
    // cout << '\n';

    // cout << "csr_indices:\t";
    // for (int i = 0; i < nz; ++i) {
    //     cout << csr_indices[i] << '\t';
    // }
    // cout << '\n';

    // cout << "csr_ptr:\t";
    // for (int i = 0; i <= rows; ++i) {
    //     cout << csr_ptr[i] << '\t';
    // }
    // cout << '\n';

    h_Ax_vector = new float[dim];
    h_Ax_vector_rep = new float[dim];

    h_x_vector = new float[dim];

    uint32_t seed = parameters->seed == 0 ? DEFAULT_SEED : parameters->seed;
    uint32_t nruns = parameters->nruns == 0 ? DEFAULT_NUMBER_OF_RUNS : parameters->nruns;

    if (!generate_vector(h_x_vector, dim, seed))
    {
        fprintf(stderr, "Failed to generate dense vector.\n");
        exit(-1);
    }

    // cout << '\n';
    // for (int i = 0, entries_idx = 0, h_idx = 0, csr_idx = 0, csr_row = 0; i < rows; ++i)
    // {
    //     for (int j = 0; j < cols; ++j)
    //     {
    //         if (entries[entries_idx].row == i && entries[entries_idx].col == j)
    //         {
    //             cout << entries[entries_idx].val << '\t';
    //             entries_idx++;
    //         }
    //         else
    //         {
    //             cout << 0 << '\t';
    //         }
    //     }

    //     cout << '\t';

    //     for (int j = 0; j < cols; ++j)
    //     {
    //         while (csr_idx >= csr_ptr[csr_row + 1])
    //         {
    //             csr_row++;
    //         }

    //         if (csr_row == i && csr_idx < csr_ptr[csr_row + 1] && csr_indices[csr_idx] == j)
    //         {
    //             cout << csr_data[csr_idx] << '\t';
    //             csr_idx++;
    //         }
    //         else
    //         {
    //             cout << 0 << '\t';
    //         }
    //     }
        
    //     cout << '\t' << h_x_vector[i] << '\n';
    // }

    uint64_t time_non_rep, time_rep;

    execute(nruns, false, dim, h_nzcnt, h_ptr, h_indices, h_data, h_x_vector, h_perm, h_Ax_vector, time_non_rep);

    execute(nruns, true, dim, nullptr, csr_ptr, csr_indices, csr_data, h_x_vector, nullptr, h_Ax_vector_rep, time_rep);

    // for (int i = 0; i < dim; ++i)
    // {
    //     cout << fixed << setprecision(10) << h_Ax_vector[i] << '\t' << h_Ax_vector_rep[i] << '\n';
    // }

    if (diff(dim, h_Ax_vector, h_Ax_vector_rep))
    {
        printf("Non-reproducible and reproducible results do not match!\n");
    }

    printf("Non-reproducible implementation time: %ld [us] (%.3f [ms])\n", time_non_rep, (float) time_non_rep / 1000.0);
    printf("Reproducible implementation time: %ld [us] (%.3f [ms])\n", time_rep, (float) time_rep / 1000.0);

    delete[] h_data;
    delete[] h_indices;
    delete[] h_ptr;
    delete[] h_perm;
    delete[] h_nzcnt;
    
    delete[] csr_data;
    delete[] csr_ptr;
    delete[] csr_indices;

    delete[] h_Ax_vector_rep;
    delete[] h_x_vector;

    delete[] entries;

    pb_SwitchToTimer(&timers, pb_TimerID_NONE);

    pb_PrintTimerSet(&timers);
    pb_FreeParameters(parameters);

    return 0;
}