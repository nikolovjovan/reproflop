#ifndef LONGACCUMULATOR_H
#define LONGACCUMULATOR_H

#include "common.h"

// binary32 (float) component widths
//
constexpr uint32_t EXPONENT_WIDTH = 8;
constexpr uint32_t MANTISSA_WIDTH = 23;

// Total width of the long accumulator required to store a binary32 number.
//
constexpr uint32_t TOTAL_WIDTH = (1 << EXPONENT_WIDTH) + MANTISSA_WIDTH;

// Size of one long accumulator (number of words) and total number of long accumulators.
//
constexpr uint32_t ACCUMULATOR_SIZE = (TOTAL_WIDTH + 8 * sizeof(uint32_t) - 1) / (8 * sizeof(uint32_t));
constexpr uint32_t ACCUMULATOR_COUNT = 512;

// Size of a single work unit (warp).
//
constexpr uint32_t WARP_SIZE = 16;

// For the first step (accumulation) we use 16 warps of 16 threads for a total of 256 threads in a workgroup.
//
constexpr uint32_t ACCUMULATE_WARP_COUNT = 16;
constexpr uint32_t ACCUMULATE_WORKGROUP_SIZE = WARP_SIZE * ACCUMULATE_WARP_COUNT;

constexpr uint32_t nextPow2(uint32_t v)
{
    if (v == 0)
        return 1;

    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;

    return v;
}

// For the second step (merging and rounding) we use 1 warp of 16 threads for a total of 16 threads in a workgroup.
// This is because ACCUMULATOR_SIZE is 9 and we only need 9 threads 
//
constexpr uint32_t MERGE_WARP_COUNT = nextPow2(ACCUMULATOR_SIZE) / WARP_SIZE;
constexpr uint32_t MERGE_WORKGROUP_SIZE = WARP_SIZE * MERGE_WARP_COUNT;

// Number of accumulators that will get merged in the first step.
//
constexpr uint32_t MERGE_ACCUMULATOR_COUNT = ACCUMULATE_WARP_COUNT;

class LongAccumulator
{
public:
    static int InitializeOpenCL();
    static int CleanupOpenCL();

    static float Sum(const int N, float *arr, int *err = nullptr);

    static float DotProduct(const int N, float *arrA, float *arrB, int *err = nullptr);

private:
    static cl_int InitializeAcc(
        cl_context context,
        cl_command_queue commandQueue,
        cl_device_id device);
    static cl_int CleanupAcc();
private:
    static bool s_initializedOpenCL;
    static bool s_initializedLngAcc;

    static cl_platform_id s_platform;
    static cl_device_id s_device;
    static cl_context s_context;
    static cl_command_queue s_commandQueue;

    static cl_program s_program;
    static cl_kernel s_clkAccumulate;
    static cl_kernel s_clkDotProduct;
    static cl_kernel s_clkMerge;
    static cl_kernel s_clkRound;

    static cl_mem s_data_arr;
    static cl_mem s_data_arrA;
    static cl_mem s_data_arrB;
    static cl_mem s_data_res;

    static cl_mem s_accumulators;
};

#endif  // LONGACCUMULATOR_H