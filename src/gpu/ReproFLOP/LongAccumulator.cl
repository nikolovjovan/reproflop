// Only one of these defines can be enabled at one time.
//
// #define LOCAL_ACCUMULATION_TESTING
// #define LOCAL_MERGE_TESTING
// #define GLOBAL_MERGE_TESTING
// #define FINAL_MERGE_TESTING

#define WORKGROUP_SIZE          (ACCUMULATE_WARP_COUNT * WARP_SIZE)
#define MERGE_WORKGROUP_SIZE    (MERGE_WARP_COUNT * WARP_SIZE)

typedef struct {
    bool negative;
    uint exponent;
    uint mantissa;
} float_components;

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

__kernel __attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
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
    __local uint lacc[ACCUMULATE_WARP_COUNT * ACCUMULATOR_SIZE] __attribute__((aligned(4)));

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

#ifdef LOCAL_ACCUMULATION_TESTING
        accumulators[i * ACCUMULATOR_SIZE + 0] = components.negative;
        accumulators[i * ACCUMULATOR_SIZE + 1] = components.exponent;
        accumulators[i * ACCUMULATOR_SIZE + 2] = components.mantissa;
        accumulators[i * ACCUMULATOR_SIZE + 3] = k;
        accumulators[i * ACCUMULATOR_SIZE + 4] = split;
        accumulators[i * ACCUMULATOR_SIZE + 5] = hi_width;
        accumulators[i * ACCUMULATOR_SIZE + 6] = components.mantissa << (split ? (24 - hi_width) : (hi_width - 24));
        accumulators[i * ACCUMULATOR_SIZE + 7] = split ? components.mantissa >> (8 + hi_width) : 0;
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);

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
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (get_local_id(0) < ACCUMULATOR_SIZE) {
        for (uint i = 0; i < ACCUMULATE_WARP_COUNT; ++i)
            AddGlobal((accumulators + get_group_id(0) * ACCUMULATOR_SIZE), get_local_id(0), lacc[get_local_id(0) * ACCUMULATE_WARP_COUNT + i], false);
    }
#endif
#endif
}

__kernel __attribute__((reqd_work_group_size(MERGE_WORKGROUP_SIZE, 1, 1)))
void LongAccumulatorMerge (
    __global uint *accumulators)
{
#if !defined(LOCAL_ACCUMULATION_TESTING) && !defined(LOCAL_MERGE_TESTING)
    if (get_local_id(0) < ACCUMULATOR_SIZE)
    {
        // Merge warp-level accumulators into group-level long accumulators.
        //
        for (uint i = 1; i < MERGE_ACCUMULATOR_COUNT; ++i) {
            AddGlobal((accumulators + get_group_id(0) * MERGE_ACCUMULATOR_COUNT * ACCUMULATOR_SIZE), get_local_id(0), accumulators[(get_group_id(0) * MERGE_ACCUMULATOR_COUNT + i) * ACCUMULATOR_SIZE + get_local_id(0)], false);

#if defined(GLOBAL_MERGE_TESTING) || defined(FINAL_MERGE_TESTING)
            accumulators[(get_group_id(0) * MERGE_ACCUMULATOR_COUNT + i) * ACCUMULATOR_SIZE + get_local_id(0)] = 0;
#endif
        }
    }
#endif
}

__kernel __attribute__((reqd_work_group_size(MERGE_WORKGROUP_SIZE, 1, 1)))
void LongAccumulatorRound (
    __global uint *accumulators,
    __global float *result)
{
    // Only one workgroup is required for this operation.
    //
    // if (get_group_id(0) != 0) {
    //     return;
    // }

#if !defined(LOCAL_ACCUMULATION_TESTING) && !defined(LOCAL_MERGE_TESTING)
    if (get_local_id(0) < ACCUMULATOR_SIZE)
    {
#ifndef GLOBAL_MERGE_TESTING
        // Merge group-level long accumulators into a final long accumulator.
        //
        for (uint i = MERGE_ACCUMULATOR_COUNT; i < ACCUMULATOR_COUNT; i += MERGE_ACCUMULATOR_COUNT) {
            AddGlobal(accumulators, get_local_id(0), accumulators[i * ACCUMULATOR_SIZE + get_local_id(0)], false);

#ifdef FINAL_MERGE_TESTING
            accumulators[i * ACCUMULATOR_SIZE + get_local_id(0)] = 0;
#endif
        }
#endif

        if (get_local_id(0) == 0)
        {
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
        }
    }
#endif
}