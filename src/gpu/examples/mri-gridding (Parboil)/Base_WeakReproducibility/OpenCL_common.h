
#ifndef __OPENCL_COMMON_H_
#define __OPENCL_COMMON_H_

#include <cstdio>
#include <cstdarg>

#include <CL/cl.h>

int getOpenCLDevice(cl_platform_id *platform, cl_device_id *device, cl_device_type *reqDeviceType, int numRequests, ...);
const char *oclErrorString(cl_int error);
const char *oclDebugErrString(cl_int error, cl_device_id device);

#define OCL_ERRCK_VAR(var)                                                                           \
    {                                                                                                \
        if (var != CL_SUCCESS)                                                                       \
            fprintf(stderr, "OpenCL Error (%s: %d): %s\n", __FILE__, __LINE__, oclErrorString(var)); \
    }

#define OCL_ERRCK_RETVAL(s)                                                                                         \
    {                                                                                                               \
        cl_int clerr = (s);                                                                                         \
        if (clerr != CL_SUCCESS)                                                                                    \
            fprintf(stderr, "OpenCL Error (%s: %d): %s\n", __FILE__, __LINE__, oclDebugErrString(clerr, clDevice)); \
    }

#define OCL_SIMPLE_ERRCK_RETVAL(s)                                                                     \
    {                                                                                                  \
        cl_int clerr = (s);                                                                            \
        if (clerr != CL_SUCCESS)                                                                       \
            fprintf(stderr, "OpenCL Error (%s: %d): %s\n", __FILE__, __LINE__, oclErrorString(clerr)); \
    }

char *oclLoadProgSource(const char *cFilename, const char *cPreamble, size_t *szFinalLength);

#endif