#ifndef COMMON_H
#define COMMON_H

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include <iostream>

#include <CL/opencl.h>

/**
 * The following two functions are copied from exblas library:
 * 
 * https://github.com/riakymch/exblas
 * 
 * and are under the license found in the root of the repository:
 * 
 * LICENSE (exblas)
 * 
 */

/**
 * @brief Function to obtain platform.
 * 
 * @param name Platform Name
 * @return cl_platform_id Platform ID
 */
cl_platform_id GetOCLPlatform(
    char name[]);

/**
 * @brief Function to obtain device.
 * 
 * @param pPlatform Platform ID
 * @return cl_device_id Device ID
 */
cl_device_id GetOCLDevice(
    cl_platform_id pPlatform);

#endif  // LONGACCUMULATOR_H