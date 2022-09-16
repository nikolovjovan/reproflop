#include "common.h"

using namespace std;

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

cl_platform_id GetOCLPlatform(
    char name[])
{
    cl_platform_id pPlatforms[10] = { 0 };
    char pPlatformName[128] = { 0 };

    cl_uint uiPlatformsCount = 0;
    cl_int err = clGetPlatformIDs(10, pPlatforms, &uiPlatformsCount);
    cl_int ui_res = -1;

    for (cl_int ui = 0; ui < (cl_int) uiPlatformsCount; ++ui) {
        err = clGetPlatformInfo(pPlatforms[ui], CL_PLATFORM_NAME, 128 * sizeof(char), pPlatformName, NULL);
        if ( err != CL_SUCCESS ) {
            cout << "ERROR: Failed to retreive platform vendor name.\n";
            return NULL;
        }

        // cout << "### Platform[" << ui << "] : " << pPlatformName << "\n";

        if (!strcmp(pPlatformName, name))
            ui_res = ui; //return pPlatforms[ui];
    }

    // cout << "### Using Platform : " << name << "\n";

    if (ui_res > -1)
        return pPlatforms[ui_res];
    else
        return NULL;
}

cl_device_id GetOCLDevice(
    cl_platform_id pPlatform)
{
    cl_device_id dDevices[10] = { 0 };
    char name[128] = { 0 };
    char dDeviceName[128] = { 0 };

    cl_uint uiNumDevices = 0;
    cl_int err = clGetDeviceIDs(pPlatform, CL_DEVICE_TYPE_GPU, 10, dDevices, &uiNumDevices);
    if (err != CL_SUCCESS) {
        cout << "Error in clGetDeviceIDs, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        exit(0);
    }

    for (cl_int ui = 0; ui < (cl_int) uiNumDevices; ++ui) {
        err = clGetDeviceInfo(dDevices[ui], CL_DEVICE_NAME, 128 * sizeof(char), dDeviceName, NULL);
        if (err != CL_SUCCESS) {
            cout << "ERROR: Failed to retreive platform vendor name.\n";
            return NULL;
        }

        // cout << "### Device[" << ui << "] : " << dDeviceName << "\n";
        if (ui == 0) {
            strcpy(name, dDeviceName);
        }
    }

    // cout << "### Using Device : " << name << "\n";

    return dDevices[0];
}