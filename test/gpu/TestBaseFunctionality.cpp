#include <iomanip>
#include <iostream>
#include <random>

#include "LongAccumulator.h"
#include "LongAccumulatorCPU.h"

using namespace std;

// Floating-point specification constants

constexpr uint32_t SEED = 1549813198;
constexpr int EXPONENT_MIN_VALUE = -10;
constexpr int EXPONENT_MAX_VALUE = 10;
constexpr int EXPONENT_BIAS = 127;

typedef enum {
    AllOnes     = 0,
    AllNatural  = 1,
    Random      = 2
} Mode;

constexpr int DEFAULT_ELEMENT_COUNT = 1000;
constexpr int DEFAULT_REPEAT_COUNT = 100;

Mode mode = AllOnes;
int N = DEFAULT_ELEMENT_COUNT;
int R = DEFAULT_REPEAT_COUNT;

// -- CPU testing --

// -- patches

#define __local
#define __global

uint as_uint(float f)
{
    uint tmp;
    memcpy(&tmp, &f, sizeof(float));
    return tmp;
}

float as_float(uint frep)
{
    float tmp;
    memcpy(&tmp, &frep, sizeof(uint));
    return tmp;
}

uint atomic_add(volatile uint *addr, uint val)
{
    uint old = *addr;
    *addr += val;
    return old;
}

uint atomic_sub(volatile uint *addr, uint val)
{
    uint old = *addr;
    *addr -= val;
    return old;
}

uint sim_local_id = 0;
uint sim_global_id = 0;
uint sim_group_id = 0;

uint sim_local_size = 0;
uint sim_global_size = 0;

uint get_local_id(uint idx)
{
    return sim_local_id;
}

uint get_global_id(uint idx)
{
    return sim_global_id;
}

uint get_group_id(uint idx)
{
    return sim_group_id;
}

uint get_local_size(uint idx)
{
    return sim_local_size;
}

uint get_global_size(uint idx)
{
    return sim_global_size;
}

uint wlacc[ACCUMULATOR_COUNT][ACCUMULATE_WARP_COUNT * ACCUMULATOR_SIZE];
uint accumulators[ACCUMULATOR_COUNT * ACCUMULATOR_SIZE];

// -- END patches

// Only one of these defines can be enabled at one time.
//
// #define LOCAL_ACCUMULATION_TESTING
// #define LOCAL_MERGE_TESTING
// #define GLOBAL_MERGE_TESTING

#define WORKGROUP_SIZE          (ACCUMULATE_WARP_COUNT * WARP_SIZE)
#define MERGE_WORKGROUP_SIZE    (MERGE_WARP_COUNT * WARP_SIZE)
#define barrier(val)

// typedef struct {
//     bool negative;
//     uint exponent;
//     uint mantissa;
// } float_components;

float_components ExtractFloatComponents(
    float f)
{
    float_components components;
    uint frep = as_uint(f);
    components.negative = frep & 0x80000000;
    components.exponent = (frep >> 23) & 0xFF;
    components.mantissa = frep & 0x7FFFFF;
    if (components.exponent != 0x00 && components.exponent != 0xFF) {
        components.mantissa |= 0x800000; // hidden bit is equal to 1
    }
    return components;
}

float PackComponentsToFloat(
    float_components components)
{
    uint frep = (components.negative ? 0x80000000 : 0x00000000) |
               (components.exponent << 23) |
               (components.mantissa & 0x7FFFFF);
    return as_float(frep);
}

// Local long accumulators are intertwined:
// 0_0 0_1 0_2 ... 0_(ACCUMULATE_WARP_COUNT - 1) 1_0 1_1 ... /* comment this out when not testing */
//
void AddLocal(
    __local volatile uint *tlacc,
    uint idx,
    uint val,
    bool negative)
{
    uint old = 0;

    while (idx < ACCUMULATOR_SIZE) {
        // Use atomic operation to avoid race conditions and to get the old value.
        //
        old = negative ?
            atomic_sub(&tlacc[idx * ACCUMULATE_WARP_COUNT], val) :
            atomic_add(&tlacc[idx * ACCUMULATE_WARP_COUNT], val);

        // Check for underflow/overflow.
        //
        if (negative ?
                /* underflow */ old - val > old :
                /* overflow */  old + val < old) {
            ++idx;
            val = 1;
        } else break;
    }
}

// Global long accumulators are NOT intertwined:
// 0_0 1_0 2_0 ... (ACCUMULATOR_SIZE - 1)_0 0_1 1_1 2_1 ...
//
void AddGlobal(
    __global volatile uint *acc,
    uint idx,
    uint val,
    bool negative)
{
    uint old = 0;

    while (idx < ACCUMULATOR_SIZE) {
        // Use atomic operation to avoid race conditions and to get the old value.
        //
        if (!negative) {
            old = atomic_add(&acc[idx], val);
        } else {
            old = atomic_sub(&acc[idx], val);
        }

        // Check for underflow/overflow.
        //
        if (negative ?
                /* underflow */ old - val > old :
                /* overflow */  old + val < old) {
            ++idx;
            val = 1;
        } else break;
    }
}

// Round mantissa to nearest.
//
void RoundMantissa(
    __global uint *acc,
    float_components *components,
    int word_idx,
    int bit_idx)
{
    uint bits = acc[word_idx];
    if (!(bits & (1 << bit_idx))) {
        return; // case 1
    }

    bool allzero = true;
    for (int i = bit_idx - 1; allzero && i >= 0; --i) {
        if (bits & (1 << i)) {
            allzero = false;
        }
    }

    if (allzero) {
        for (int i = word_idx - 1; allzero && i >= 0; --i) {
            if (acc[i] > 0) {
                allzero = false;
            }
        }
    }

    if (!allzero) {
        components->mantissa++; // case 2, need to check for overflow
    } else if ((components->mantissa & 0x1) == 0) {
        return; // case 3a
    } else {
        components->mantissa++; // case 3b, need to check for overflow
    }

    // Check if rounding caused overflow...
    //
    if (components->mantissa > 0xFFFFFF) {
        components->mantissa >>= 1;
        components->exponent++;
        if (components->exponent > 0xFE) {
            components->exponent = 0xFF;
            components->mantissa = 0;
        }
    }
}

// __kernel __attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
void LongAccumulatorAccumulateInitialize ()
{
    // Workgroup-level local long accumulators.
    // There is one long accumulator for each warp in a workgroup.
    // Therefore threads from different warps can share the same long accumulator.
    // WORKGROUP_SIZE = ACCUMULATE_WARP_COUNT * WARP_SIZE
    //
    // __local uint lacc[ACCUMULATE_WARP_COUNT * ACCUMULATOR_SIZE] __attribute__((aligned(4)));
    uint *lacc = wlacc[get_group_id(0)];

    // Thread-level shared long accumulator.
    // Local long accumulators are intertwined for better memory access:
    // 0_0 0_1 0_2 ... 0_(ACCUMULATE_WARP_COUNT - 1) 1_0 1_1 ...
    // The first number indicates a segment of a single long accumulator (0 .. (ACCUMULATOR_SIZE - 1)).
    // The second number indicates the accumulator which the segment belongs to.
    //
    __local uint *tlacc = lacc + (!(ACCUMULATE_WARP_COUNT & (ACCUMULATE_WARP_COUNT - 1)) ?
        (get_local_id(0) & (ACCUMULATE_WARP_COUNT - 1)) /* power of 2 modulo optimizaation */ :
        (get_local_id(0) % ACCUMULATE_WARP_COUNT));

    // Initialize thread-level shared long accumulator.
    //
    for (uint i = 0; i < ACCUMULATOR_SIZE; ++i)
        tlacc[i * ACCUMULATE_WARP_COUNT] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
}

// __kernel __attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
void LongAccumulatorAccumulate (
    const uint N,
    __global float *arr,
    __global uint *accumulators)
{
    // Workgroup-level local long accumulators.
    // There is one long accumulator for each warp in a workgroup.
    // Therefore threads from different warps can share the same long accumulator.
    // WORKGROUP_SIZE = ACCUMULATE_WARP_COUNT * WARP_SIZE
    //
    // __local uint lacc[ACCUMULATE_WARP_COUNT * ACCUMULATOR_SIZE] __attribute__((aligned(4)));
    uint *lacc = wlacc[get_group_id(0)];

    // Thread-level shared long accumulator.
    // Local long accumulators are intertwined for better memory access:
    // 0_0 0_1 0_2 ... 0_(ACCUMULATE_WARP_COUNT - 1) 1_0 1_1 ...
    // The first number indicates a segment of a single long accumulator (0 .. (ACCUMULATOR_SIZE - 1)).
    // The second number indicates the accumulator which the segment belongs to.
    //
    __local uint *tlacc = lacc + (!(ACCUMULATE_WARP_COUNT & (ACCUMULATE_WARP_COUNT - 1)) ?
        (get_local_id(0) & (ACCUMULATE_WARP_COUNT - 1)) /* power of 2 modulo optimizaation */ :
        (get_local_id(0) % ACCUMULATE_WARP_COUNT));

    // Initialize thread-level shared long accumulator.
    //
    // for (uint i = 0; i < ACCUMULATOR_SIZE; ++i)
    //     tlacc[i * ACCUMULATE_WARP_COUNT] = 0;
    // barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = get_global_id(0); i < N; i += get_global_size(0))
    {
        // Fetch element.
        //
        float f = arr[i];

        // Extract floating-point components - sign, exponent and mantissa.
        //
        float_components components = ExtractFloatComponents(f);

        // Calculate the leftmost word that will be affected.
        //
        uint k = (components.exponent + 22) >> 5;

        // Check if more than one word is affected (split mantissa).
        //
        bool split = ((components.exponent - 1) >> 5) < k;

        // Calculate number of bits in the higher part of the mantissa.
        //
        uint hi_width = (components.exponent - 9) & 0x1F;

        if (!split)
        {
            AddLocal(tlacc, k, components.mantissa << (hi_width - 24), components.negative);
        }
        else
        {
            AddLocal(tlacc, k - 1, components.mantissa << (8 + hi_width), components.negative);
            AddLocal(tlacc, k, components.mantissa >> (24 - hi_width), components.negative);
        }

    }
    barrier(CLK_LOCAL_MEM_FENCE);
/*
#ifndef LOCAL_ACCUMULATION_TESTING
#if defined(LOCAL_MERGE_TESTING)
    if (get_global_id(0) == 0)
        for (uint i = 0; i < ACCUMULATE_WARP_COUNT; ++i)
            for (uint j = 0; j < ACCUMULATOR_SIZE; ++j)
                accumulators[i * ACCUMULATOR_SIZE + j] = lacc[j * ACCUMULATE_WARP_COUNT + i];
#elif defined(GLOBAL_MERGE_TESTING)
    if (get_global_id(0) == 0)
        for (uint i = 0; i < N; ++i)
            for (uint j = 0; j < ACCUMULATOR_SIZE; ++j)
                accumulators[i * ACCUMULATOR_SIZE + j] = j + 1;
#else
    // Merge local long accumulators into a single global workgroup-level long accumulator.
    //
    if (get_local_id(0) < ACCUMULATOR_SIZE) {
        accumulators[get_group_id(0) * ACCUMULATOR_SIZE + get_local_id(0)] = 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint i = 0; i < ACCUMULATE_WARP_COUNT; ++i)
            AddGlobal((accumulators + get_group_id(0) * ACCUMULATOR_SIZE), get_local_id(0), lacc[get_local_id(0) * ACCUMULATE_WARP_COUNT + i], false);
    }
#endif
#endif
*/
}

void LongAccumulatorAccumulateMerge (
    const uint N,
    __global float *arr,
    __global uint *accumulators)
{
    uint *lacc = wlacc[get_group_id(0)];

// HACK

    if (get_global_id(0) == 0) {
        for (uint i = 0; i < ACCUMULATOR_COUNT; ++i) {
            for (uint j = 0; j < ACCUMULATOR_SIZE; ++j) {
                accumulators[i * ACCUMULATOR_SIZE + j] = 0;
            }
        }
    }

//

    // Merge local long accumulators into a single global workgroup-level long accumulator.
    //
    if (get_local_id(0) < ACCUMULATOR_SIZE) {
        // accumulators[get_group_id(0) * ACCUMULATOR_SIZE + get_local_id(0)] = 0;
        // barrier(CLK_LOCAL_MEM_FENCE);
        for (uint i = 0; i < ACCUMULATE_WARP_COUNT; ++i)
            AddGlobal((accumulators + get_group_id(0) * ACCUMULATOR_SIZE), get_local_id(0), lacc[get_local_id(0) * ACCUMULATE_WARP_COUNT + i], false);
    }
}

// __kernel __attribute__((reqd_work_group_size(MERGE_WORKGROUP_SIZE, 1, 1)))
void LongAccumulatorMerge1 (
    __global uint *accumulators)
{
#if !defined(LOCAL_ACCUMULATION_TESTING) && !defined(LOCAL_MERGE_TESTING)
    if (get_local_id(0) < ACCUMULATOR_SIZE)
    {
        // Merge warp-level accumulators into group-level long accumulators.
        //
        for (uint i = 1; i < MERGE_ACCUMULATOR_COUNT; ++i) {
            AddGlobal((accumulators + get_group_id(0) * MERGE_ACCUMULATOR_COUNT * ACCUMULATOR_SIZE), get_local_id(0), accumulators[(get_group_id(0) * MERGE_ACCUMULATOR_COUNT + i) * ACCUMULATOR_SIZE + get_local_id(0)], false);

// #ifdef GLOBAL_MERGE_TESTING
            accumulators[(get_group_id(0) * MERGE_ACCUMULATOR_COUNT + i) * ACCUMULATOR_SIZE + get_local_id(0)] = 0;
// #endif
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

// HACK
}

// __kernel __attribute__((reqd_work_group_size(MERGE_WORKGROUP_SIZE, 1, 1)))
void LongAccumulatorMerge2 (
    __global uint *accumulators)
{
// END HACK

    if ((get_local_id(0) < ACCUMULATOR_SIZE) && (get_group_id(0) == 0))
    {
#ifndef GLOBAL_MERGE_TESTING
        // Merge group-level long accumulators into a final long accumulator.
        //
        for (uint i = 1; i < ACCUMULATOR_COUNT / MERGE_ACCUMULATOR_COUNT; ++i) {
            AddGlobal(accumulators, get_local_id(0), accumulators[i * MERGE_ACCUMULATOR_COUNT * ACCUMULATOR_SIZE + get_local_id(0)], false);

            accumulators[i * MERGE_ACCUMULATOR_COUNT * ACCUMULATOR_SIZE + get_local_id(0)] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
#endif

// HACK
    }
#endif
}

// __kernel __attribute__((reqd_work_group_size(MERGE_WORKGROUP_SIZE, 1, 1)))
void LongAccumulatorRound (
    __global uint *accumulators,
    __global float *result)
{
// END HACK

        // if (get_global_id(0) == 0)
        // {
            float_components components;

            components.negative = accumulators[ACCUMULATOR_SIZE - 1] & 0x80000000;
            components.exponent = 0;
            components.mantissa = 0;

#ifndef GLOBAL_MERGE_TESTING
            // Reset additional accumulator to 0 to prepare it for storing absolute result value.
            //
            for (uint i = 0; i < ACCUMULATOR_SIZE; ++i) {
                accumulators[ACCUMULATOR_SIZE + i] = accumulators[i];
                accumulators[i] = 0;
            }

            for (uint i = 0; i < ACCUMULATOR_SIZE; ++i) {
                AddGlobal(accumulators, i, accumulators[ACCUMULATOR_SIZE + i], components.negative);
            }
#endif

            // Calculate the position of the most significant non-zero word.
            //
            int word_idx = ACCUMULATOR_SIZE - 1;
            while (word_idx >= 0 && accumulators[word_idx] == 0) {
                word_idx--;
            }

            // Check if zero.
            //
            if (word_idx < 0) {
                *result = PackComponentsToFloat(components);
                return;
            }

            // Calculate the position of the most significant bit.
            //
            uint mask = 0x80000000;
            int bit_idx = 31;
            while (bit_idx >= 0 && !(accumulators[word_idx] & mask)) {
                bit_idx--;
                mask >>= 1;
            }

            // Check if subnormal.
            //
            if (word_idx == 0 && bit_idx < 23) {
                components.mantissa = accumulators[0];
                *result = PackComponentsToFloat(components);
                return;
            }

            // Calculate the exponent using the inverse of the formula used for "sget_local_id(0)ing" the mantissa into the accumulator.
            //
            components.exponent = (word_idx << 5) + bit_idx - 22;

            // Check if infinity.
            //
            if (components.exponent > 0xFE) {
                components.exponent = 0xFF;
                *result = PackComponentsToFloat(components);
                return;
            }

            // Extract bits of mantissa from current word.
            //
            mask = 0xFFFFFFFF;
            if (bit_idx < 31) {
                mask = (1 << (bit_idx + 1)) - 1;
            }
            components.mantissa = accumulators[word_idx] & mask;

            // Extract mantissa and round to nearest.
            //
            if (bit_idx > 23) {
                components.mantissa >>= bit_idx - 23;
                RoundMantissa(accumulators, &components, word_idx, bit_idx - 24);
            } else if (bit_idx == 23) {
                RoundMantissa(accumulators, &components, word_idx - 1, 31);
            } else {
                components.mantissa <<= 23 - bit_idx;
                components.mantissa |= accumulators[word_idx - 1] >> (bit_idx + 9);
                RoundMantissa(accumulators, &components, word_idx - 1, bit_idx + 8);
            }

            *result = PackComponentsToFloat(components);
            return;
//        }
//     }
// #endif
}

std::ostream &print_acc(std::ostream &out, LongAccumulatorCPU &acc)
{
    for (int i = ACC_SIZE - 1; i >= 0; --i) {
        out << acc.getAcc()[i] << (i > 0 ? '\t' : '\n');
    }
    return out;
}

float SimulateLongAccumulator(uint N, float *arr)
{
    // memset((void *) wlacc, 0, ACCUMULATOR_COUNT * ACCUMULATE_WARP_COUNT * ACCUMULATOR_SIZE * sizeof(uint));
    // memset((void *) accumulators, 0, ACCUMULATOR_COUNT * ACCUMULATOR_SIZE * sizeof(uint));

    sim_local_size = ACCUMULATE_WORKGROUP_SIZE;
    sim_global_size = sim_local_size * ACCUMULATOR_COUNT;

    for (sim_global_id = 0; sim_global_id < sim_global_size; ++sim_global_id) {
        sim_local_id = sim_global_id % ACCUMULATE_WORKGROUP_SIZE;
        sim_group_id = sim_global_id / ACCUMULATE_WORKGROUP_SIZE;
        LongAccumulatorAccumulateInitialize();
    }

    for (sim_global_id = 0; sim_global_id < sim_global_size; ++sim_global_id) {
        sim_local_id = sim_global_id % ACCUMULATE_WORKGROUP_SIZE;
        sim_group_id = sim_global_id / ACCUMULATE_WORKGROUP_SIZE;
        LongAccumulatorAccumulate(N, arr, accumulators);
    }

    if (N < 500) {
        cout << "wlacc:\n\n";

        bool allzero = false;
        for (int step = 0; step < 2; ++step) {
            for (int i = 0; i < ACCUMULATOR_COUNT; ++i) {
                for (int j = 0; j < ACCUMULATE_WARP_COUNT && !allzero; ++j) {
                    allzero = true;
                    for (int k = 0; k < ACCUMULATOR_SIZE; ++k) {
                        if (wlacc[i][j + k * ACCUMULATE_WARP_COUNT]) {
                            allzero = false;
                        }
                    }
                    if (!allzero) {
                        printf("[%05u][%05u]: ", i, j);
                        LongAccumulatorCPU cpuLacc;
                        for (int k = ACCUMULATOR_SIZE - 1; k >= 0; --k) {
                            if (step == 0) {
                                cpuLacc.getAcc()[k] = wlacc[i][j + k * ACCUMULATE_WARP_COUNT];
                            } else if (step == 1) {
                                printf("%u%c", wlacc[i][j + k * ACCUMULATE_WARP_COUNT], k > 0 ? '\t' : '\n');
                            }
                        }
                        if (step == 0) {
                            cout << cpuLacc() << '\n';
                        }
                    }
                }
            }

            if (step == 0) {
                cout << "\n(LongAccumulator) wlacc:\n";
                allzero = false;
            }
        }
    }

    for (sim_global_id = 0; sim_global_id < sim_global_size; ++sim_global_id) {
        sim_local_id = sim_global_id % ACCUMULATE_WORKGROUP_SIZE;
        sim_group_id = sim_global_id / ACCUMULATE_WORKGROUP_SIZE;
        LongAccumulatorAccumulateMerge(N, arr, accumulators);
    }

    bool allzero = false;

    if (N < 1000000) {
        cout << "\naccumulators (before global merge):\n\n";

        allzero = false;
        for (int step = 0; step < 2; ++step) {
            for (int i = 0; i < ACCUMULATOR_COUNT; ++i) {
                allzero = true;
                for (int j = ACCUMULATOR_SIZE - 1; j >= 0 && allzero; --j) {
                    if (accumulators[i * ACCUMULATOR_SIZE + j]) {
                        allzero = false;
                    }
                }
                if (!allzero) {
                    printf("[%05u]: ", i);
                    if (step == 0) {
                        LongAccumulatorCPU cpuLacc (&accumulators[i * ACCUMULATOR_SIZE]);
                        cout << cpuLacc() << '\n';
                    } else if (step == 1) {
                        for (int j = ACCUMULATOR_SIZE - 1; j >= 0; --j) {
                            printf("%u%c", accumulators[i * ACCUMULATOR_SIZE + j], j > 0 ? '\t' : '\n');
                        }
                    }
                }
            }

            if (step == 0) {
                cout << "\n(LongAccumulator) accumulators:\n";
            }
        }
    }

    sim_local_size = MERGE_WORKGROUP_SIZE;
    sim_global_size = sim_local_size * (ACCUMULATOR_COUNT / MERGE_ACCUMULATOR_COUNT);

    for (int step = 0; step < 2; ++step) {
        for (sim_global_id = 0; sim_global_id < sim_global_size; ++sim_global_id) {
            sim_local_id = sim_global_id % MERGE_WORKGROUP_SIZE;
            sim_group_id = sim_global_id / MERGE_WORKGROUP_SIZE;
            if (step == 0) {
                LongAccumulatorMerge1(accumulators);
            } else {
                LongAccumulatorMerge2(accumulators);
            }
        }
        if (step == 0) {
            if (N < 1000000) {
                cout << "\naccumulators (after global merge part 1):\n\n";

                bool allzero = false;
                for (int step = 0; step < 2; ++step) {
                    for (int i = 0; i < ACCUMULATOR_COUNT; ++i) {
                        allzero = true;
                        for (int j = ACCUMULATOR_SIZE - 1; j >= 0 && allzero; --j) {
                            if (accumulators[i * ACCUMULATOR_SIZE + j]) {
                                allzero = false;
                            }
                        }
                        if (!allzero) {
                            printf("[%05u]: ", i);
                            if (step == 0) {
                                LongAccumulatorCPU cpuLacc (&accumulators[i * ACCUMULATOR_SIZE]);
                                cout << cpuLacc() << '\n';
                            } else if (step == 1) {
                                for (int j = ACCUMULATOR_SIZE - 1; j >= 0; --j) {
                                    printf("%u%c", accumulators[i * ACCUMULATOR_SIZE + j], j > 0 ? '\t' : '\n');
                                }
                            }
                        }
                    }

                    if (step == 0) {
                        cout << "\n(LongAccumulator) accumulators:\n";
                    }
                }
            }
        }
    }

    if (N < 1000000) {
        cout << "\naccumulators (after global merge part 2):\n\n";

        allzero = false;
        for (int step = 0; step < 2; ++step) {
            for (int i = 0; i < ACCUMULATOR_COUNT; ++i) {
                allzero = true;
                for (int j = ACCUMULATOR_SIZE - 1; j >= 0 && allzero; --j) {
                    if (accumulators[i * ACCUMULATOR_SIZE + j]) {
                        allzero = false;
                    }
                }
                if (!allzero) {
                    printf("[%05u]: ", i);
                    if (step == 0) {
                        LongAccumulatorCPU cpuLacc (&accumulators[i * ACCUMULATOR_SIZE]);
                        cout << cpuLacc() << '\n';
                    } else if (step == 1) {
                        for (int j = ACCUMULATOR_SIZE - 1; j >= 0; --j) {
                            printf("%u%c", accumulators[i * ACCUMULATOR_SIZE + j], j > 0 ? '\t' : '\n');
                        }
                    }
                }
            }

            if (step == 0) {
                cout << "\n(LongAccumulator) accumulators:\n";
            }
        }
    }

    float sum = 0;

    LongAccumulatorRound(accumulators, &sum);

    return sum;
}

// -- END CPU testing --

int main(int argc, char *argv[])
{
    mt19937 gen(SEED);
    uniform_int_distribution<uint32_t> sign_dist(0, 1);
    uniform_int_distribution<uint32_t> exponent_dist(EXPONENT_MIN_VALUE + EXPONENT_BIAS, EXPONENT_MAX_VALUE + EXPONENT_BIAS);
    uniform_int_distribution<uint32_t> mantisa_dist(0, (1 << 23) - 1);

    if (argc > 1) {
        N = atoi(argv[1]);
        if (N < 0) {
            N = DEFAULT_ELEMENT_COUNT;
        }
        cout << "Using N = " << N << ".\n";
    }

    if (argc > 2) {
        R = atoi(argv[2]);
        if (R < 0) {
            R = DEFAULT_REPEAT_COUNT;
        }
        cout << "Using R = " << R << ".\n";
    }

    if (argc > 3) {
        int mode_param = atoi(argv[3]);
        mode = mode_param < (int) AllOnes ? AllOnes : (mode_param > (int) Random ? Random : (Mode) mode_param);
        cout << "Using mode = " << mode << ".\n";
    }

    float *arr = new float[N];

    float expected_sum = mode == AllOnes ? N : (mode == AllNatural ? (float) N * ((float) N + 1.0) / 2.0 : 0);

    cout << "\nNumbers:\n\n";

    LongAccumulatorCPU acc;

    for (int i = 0; i < N; ++i) {
        if (mode == AllOnes) {
            arr[i] = 1.0;
        } else if (mode == AllNatural) {
            arr[i] = (float) (i + 1);
        } else if (mode == Random) {
            uint32_t bits = (sign_dist(gen) << 31) | (exponent_dist(gen) << 23) | (mantisa_dist(gen) << 0);
            static_assert(sizeof(uint32_t) == sizeof(float));
            float number;
            memcpy(&number, &bits, sizeof(uint32_t));
            arr[i] = number;
            expected_sum += number;
        }
        acc += arr[i];
        if (N < 100) {
            printf("[%05u]: ", i);
            cout << arr[i]<< '\n';
        }
    }

    float expected_sum_reproducible = acc();

    cout << "\nExpected sum: " << expected_sum << '\n';
    cout << "Expected sum (reproducible): " << expected_sum_reproducible << '\n';
    print_acc(cout, acc) << acc << "\n\n";

    cout << "(LongAccumulator) numbers:\n";

    if (N < 100) {
        for (int i = 0; i < N; ++i) {
            acc = arr[i];
            printf("[%05u]: ", i);
            print_acc(cout, acc);
        }

        cout << "\nExpected (LongAccumulator) partial sums:\n";

        for (int i = 0; i < WARP_SIZE; ++i) {
            acc = 0;
            for (int j = 0; i + j < N; j += WARP_SIZE) {
                acc += arr[i + j];
            }
            printf("[%05u]: ", i);
            print_acc(cout, acc);
        }

        cout << '\n';
    }

    if (expected_sum != expected_sum_reproducible) {
        cout << "Non-reproducible and reproducible expected sums do not match!\n";
    }

    float baseline_sim_sum = SimulateLongAccumulator(N, arr), baseline_sum;

    cout << "\nBaseline sum (gpu simulated reproducible): " << baseline_sim_sum << '\n';

    if (baseline_sim_sum != expected_sum_reproducible) {
        cout << "Baseline simulated gpu and reproducible expected sums do not match!\n";
        cout << "\nTestFailed!\n";
        goto cleanup;
    }

    // for (int i = 1; i <= R; ++i) {
    //     // cout << "\nRun " << i << ":\n\n";
    //     float sum = SimulateLongAccumulator(N, arr);
    //     if (sum != baseline_sim_sum) {
    //         cout << "Simulated GPU reproducible sum is not reproducible after " << i << " runs. Current sum: " << sum << ".\n";
    //         cout << "\nTestFailed!\n";
    //         goto cleanup;
    //     }
    // }

    LongAccumulator::InitializeOpenCL();

    baseline_sum = LongAccumulator::Sum(N, arr);
    cout << "\nBaseline sum: " << baseline_sum << '\n';

    if (baseline_sum != expected_sum_reproducible) {
        cout << "Baseline gpu and reproducible expected sums do not match!\n";
        cout << "\nTestFailed!\n";
        goto cleanup;
    }

    for (int i = 1; i <= R; ++i) {
        // cout << "\nRun " << i << ":\n\n";
        float sum = LongAccumulator::Sum(N, arr);
        if (sum != baseline_sum) {
            cout << "GPU reproducible sum is not reproducible after " << i << " runs. Current sum: " << sum << ".\n";
            cout << "\nTestFailed!\n";
            goto cleanup;
        }
    }

    LongAccumulator::CleanupOpenCL();

    cout << "\nTestPassed; ALL OK!\n";

cleanup:
    delete[] arr;

    return 0;
}