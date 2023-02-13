/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <CL/cl.h>

#include "parboil.h"
#include "UDTypes.h"
#include "OpenCL_interface.h"
#include "OpenCL_common.h"
#include "CPU_kernels.h"

#define PI 3.14159265

#include <chrono>
#include <iostream>
#include <iomanip>

using namespace std;

constexpr char* input_files[] = { "small/small.uks" };

/************************************************************
 * This function reads the parameters from the file provided
 * as a comman line argument.
 ************************************************************/
void setParameters(FILE *file, parameters *p)
{
    fscanf(file, "aquisition.numsamples=%d\n", &(p->numSamples));
    fscanf(file, "aquisition.kmax=%f %f %f\n", &(p->kMax[0]), &(p->kMax[1]), &(p->kMax[2]));
    fscanf(file, "aquisition.matrixSize=%d %d %d\n", &(p->aquisitionMatrixSize[0]), &(p->aquisitionMatrixSize[1]), &(p->aquisitionMatrixSize[2]));
    fscanf(file, "reconstruction.matrixSize=%d %d %d\n", &(p->reconstructionMatrixSize[0]), &(p->reconstructionMatrixSize[1]), &(p->reconstructionMatrixSize[2]));
    fscanf(file, "gridding.matrixSize=%d %d %d\n", &(p->gridSize[0]), &(p->gridSize[1]), &(p->gridSize[2]));
    fscanf(file, "gridding.oversampling=%f\n", &(p->oversample));
    fscanf(file, "kernel.width=%f\n", &(p->kernelWidth));
    fscanf(file, "kernel.useLUT=%d\n", &(p->useLUT));

    cl_int ciErrNum;
    cl_platform_id clPlatform;
    cl_device_type deviceType = CL_DEVICE_TYPE_ALL;
    cl_device_id clDevice;

    int deviceFound = getOpenCLDevice(&clPlatform, &clDevice, &deviceType, 0);
    if (deviceFound < 0)
    {
        fprintf(stderr, "No suitable device was found\n");
        exit(1);
    }
    cl_ulong mem_size;
    clGetDeviceInfo(clDevice, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, nullptr);

    if (p->numSamples > 10000000 && mem_size / 1024 / 1024 < 3000)
    {
        printf("  Need at least 3GB of GPU memory for large dataset\n");
        exit(1);
    }
}

/************************************************************
 * This function reads the sample point data from the kspace
 * and klocation files (and sdc file if provided) into the
 * sample array.
 * Returns the number of samples read successfully.
 ************************************************************/
unsigned int readSampleData(parameters params, FILE *uksdata_f, ReconstructionSample *samples)
{
    unsigned int i;

    for (i = 0; i < params.numSamples; i++)
    {
        if (feof(uksdata_f))
        {
            break;
        }
        fread((void *)&(samples[i]), sizeof(ReconstructionSample), 1, uksdata_f);
    }

    float kScale[3];
    kScale[0] = float(params.aquisitionMatrixSize[0]) / (float(params.reconstructionMatrixSize[0]) * float(params.kMax[0]));
    kScale[1] = float(params.aquisitionMatrixSize[1]) / (float(params.reconstructionMatrixSize[1]) * float(params.kMax[1]));
    kScale[2] = float(params.aquisitionMatrixSize[2]) / (float(params.reconstructionMatrixSize[2]) * float(params.kMax[2]));

    int size_x = params.gridSize[0];
    int size_y = params.gridSize[1];
    int size_z = params.gridSize[2];

    float ax = (kScale[0] * (size_x - 1)) / 2.0;
    float bx = (float)(size_x - 1) / 2.0;

    float ay = (kScale[1] * (size_y - 1)) / 2.0;
    float by = (float)(size_y - 1) / 2.0;

    float az = (kScale[2] * (size_z - 1)) / 2.0;
    float bz = (float)(size_z - 1) / 2.0;

    for (int n = 0; n < i; n++)
    {
        samples[n].kX = floor((samples[n].kX * ax) + bx);
        samples[n].kY = floor((samples[n].kY * ay) + by);
        samples[n].kZ = floor((samples[n].kZ * az) + bz);
    }

    return i;
}

int main(int argc, char *argv[])
{
    parameters params;
    ReconstructionSample *samples; // Input Data
    unsigned int number_of_samples;
    
    chrono::steady_clock::time_point start;

    cl_int ciErrNum;
    cl_platform_id clPlatform;
    cl_device_type deviceType = CL_DEVICE_TYPE_ALL;
    cl_device_id clDevice;
    cl_context clContext;

    int deviceFound = getOpenCLDevice(&clPlatform, &clDevice, &deviceType, 0);

    size_t max_alloc_size = 0;
    (void)clGetDeviceInfo(clDevice, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(size_t), &max_alloc_size, 0);
    size_t global_mem_size = 0;
    (void)clGetDeviceInfo(clDevice, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(size_t), &global_mem_size, 0);

    const int exe_path_len = strrchr(argv[0], '/') - argv[0] + 1;
    char exe_path[256];
    strncpy(exe_path, argv[0], exe_path_len);
    exe_path[exe_path_len] = '\0';

    char uksfile[256];
    char uksdata[256];

    float *LUT;           // use look-up table for faster execution on CPU (intermediate data)
    unsigned int sizeLUT; // set in the function calculateLUT (intermediate data)

    cmplx *gridData;      // Output Data
    float *sampleDensity; // Output Data
    int gridNumElems;

    cout << "unit: [ms]\n\n";

    int nfiles = sizeof(input_files) / sizeof(input_files[0]);

    for (int file_idx = 0; file_idx < nfiles; ++file_idx)
    {
        strncpy(uksfile, exe_path, exe_path_len + 1);
        strcat(uksfile, "data/");
        strcat(uksfile, input_files[file_idx]);

        cout << input_files[file_idx] << "\n";

        {
            FILE *uksfile_f = nullptr;
            FILE *uksdata_f = nullptr;

            strcpy(uksdata, uksfile);
            strcat(uksdata, ".data");

            uksfile_f = fopen(uksfile, "r");
            if (uksfile_f == nullptr)
            {
                printf("ERROR: Could not open %s\n", uksfile);
                exit(1);
            }

            setParameters(uksfile_f, &params);
            params.binsize = 32; // default binsize

            gridNumElems = params.gridSize[0] * params.gridSize[1] * params.gridSize[2];

            size_t samples_size = params.numSamples * sizeof(ReconstructionSample);
            size_t output_size = gridNumElems * sizeof(cmplx);

            // Check max device memory size.
            //
            if ((deviceFound < 0) ||
                ((samples_size + output_size) > global_mem_size) ||
                (samples_size > max_alloc_size) ||
                (output_size > max_alloc_size))
            {
                fprintf(stderr, "No suitable device was found\n");
                if (deviceFound >= 0)
                {
                    fprintf(stderr, "Memory requirements for this dataset exceed device capabilities\n");
                }
                exit(1);
            }

            samples = (ReconstructionSample *)malloc(params.numSamples * sizeof(ReconstructionSample));

            if (samples == nullptr)
            {
                printf("ERROR: Unable to allocate and map memory for input data\n");
                exit(1);
            }

            uksdata_f = fopen(uksdata, "rb");

            if (uksdata_f == nullptr)
            {
                printf("ERROR: Could not open data file\n");
                exit(1);
            }

            number_of_samples = readSampleData(params, uksdata_f, samples);
            fclose(uksdata_f);
        }

        for (int run = 0; run < 3; ++run)
        {
            start = chrono::steady_clock::now();
            if (params.useLUT)
            {
                float beta = PI * sqrt(4 * params.kernelWidth * params.kernelWidth / (params.oversample * params.oversample) * (params.oversample - .5) * (params.oversample - .5) - .8);
                calculateLUT(beta, params.kernelWidth, &LUT, &sizeLUT);
            }

            gridData = (cmplx *)malloc(gridNumElems * sizeof(cmplx));
            if (gridData == nullptr)
            {
                fprintf(stderr, "Could not allocate memory on host! (%s: %d)\n", __FILE__, __LINE__);
                exit(1);
            }

            sampleDensity = (float *)malloc(gridNumElems * sizeof(float));
            if (sampleDensity == nullptr)
            {
                fprintf(stderr, "Could not allocate memory on host! (%s: %d)\n", __FILE__, __LINE__);
                exit(1);
            }

            cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)clPlatform, 0};
            clContext = clCreateContextFromType(cps, deviceType, nullptr, nullptr, &ciErrNum);
            OCL_ERRCK_VAR(ciErrNum);

            cl_command_queue clCommandQueue = clCreateCommandQueue(clContext, clDevice, CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
            OCL_ERRCK_VAR(ciErrNum);

            cl_uint workItemDimensions;
            OCL_ERRCK_RETVAL(clGetDeviceInfo(clDevice, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &workItemDimensions, nullptr));

            size_t workItemSizes[workItemDimensions];
            OCL_ERRCK_RETVAL(clGetDeviceInfo(clDevice, CL_DEVICE_MAX_WORK_ITEM_SIZES, workItemDimensions * sizeof(size_t), workItemSizes, nullptr));

            // Interface function to GPU implementation of gridding
            OpenCL_interface(number_of_samples, params, samples, LUT, sizeLUT, gridData, sampleDensity, clContext, clCommandQueue, clDevice, workItemSizes);

            if (params.useLUT)
            {
                free(LUT);
            }

            free(gridData);
            free(sampleDensity);

            double time = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();

            cout << fixed << setprecision(10) << (float) time / 1000.0 << '\t'; // ms
        }
        cout << '\n';

        free(samples);
    }

    return 0;
}