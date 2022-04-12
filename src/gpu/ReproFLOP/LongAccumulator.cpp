#include "LongAccumulator.h"

#define LNGACC_KERNEL_FILENAME  "LongAccumulator.cl"

#define LNGACC_KERNEL           "LongAccumulator"
#define LNGACC_COMPLETE_KERNEL  "LongAccumulatorComplete"
#define LNGACC_ROUND_KERNEL     "LongAccumulatorRound"

using namespace std;

bool LongAccumulator::s_initializedOpenCL = false;
bool LongAccumulator::s_initializedLngAcc = false;
cl_platform_id LongAccumulator::s_platform = nullptr;
cl_device_id LongAccumulator::s_device = nullptr;
cl_context LongAccumulator::s_context = nullptr;
cl_command_queue LongAccumulator::s_commandQueue = nullptr;

cl_program LongAccumulator::s_program = nullptr;
cl_kernel LongAccumulator::s_kernel = nullptr;
cl_kernel LongAccumulator::s_complete = nullptr;
cl_kernel LongAccumulator::s_round = nullptr;

cl_mem LongAccumulator::s_data_acc = nullptr;

cl_mem LongAccumulator::s_data_res = nullptr;

#ifdef AMD
static const uint PARTIAL_SUPERACCS_COUNT = 1024;
#else
static const uint PARTIAL_SUPERACCS_COUNT = 512;
#endif
static const uint WORKGROUP_SIZE          = 256;
static const uint MERGE_WORKGROUP_SIZE    = 64;
static const uint MERGE_SUPERACCS_SIZE    = 128;
static uint NbElements;

#ifdef AMD
static char compileOptions[256] = "-DWARP_COUNT=16 -DWARP_SIZE=16 -DMERGE_WORKGROUP_SIZE=64 -DMERGE_SUPERACCS_SIZE=128 -DUSE_KNUTH";
#else
static char compileOptions[256] = "-DWARP_COUNT=16 -DWARP_SIZE=16 -DMERGE_WORKGROUP_SIZE=64 -DMERGE_SUPERACCS_SIZE=128 -DUSE_KNUTH -DNVIDIA -cl-mad-enable -cl-fast-relaxed-math"; // -cl-nv-verbose";
#endif

int LongAccumulator::InitializeOpenCL()
{
    if (s_initializedOpenCL) {
        return 1;
    }

    s_initializedOpenCL = true;

    cl_int ciErrNum;

    char platform_name[64];

#ifdef AMD
    strcpy(platform_name, "AMD Accelerated Parallel Processing");
#else
    strcpy(platform_name, "NVIDIA CUDA");
#endif

    s_platform = GetOCLPlatform(platform_name);
    if (s_platform == NULL) {
        cerr << "ERROR: Failed to find the platform '" << platform_name << "' ...\n";
        return -1;
    }

    // Get a GPU device
    //
    s_device = GetOCLDevice(s_platform);
    if (s_device == NULL) {
        cerr << "Error in clGetDeviceIDs, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        return -2;
    }

    // Create the context
    //
    s_context = clCreateContext(0, 1, &s_device, NULL, NULL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        cerr << "Error in clCreateContext, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        return -3;
    }

    // Create a command-queue
    //
    s_commandQueue = clCreateCommandQueue(s_context, s_device, CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        cerr << "Error = " << ciErrNum << "\n";
        cerr << "Error in clCreateCommandQueue, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        return -4;
    }

    ciErrNum = InitializeAcc(s_context, s_commandQueue, s_device);
    if (ciErrNum != CL_SUCCESS) {
        cerr << "Error = " << ciErrNum << "\n";
        cerr << "Error in InitializeAcc, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        return -5;
    }

    // Allocating OpenCL memory...
    //
    // s_data_acc = clCreateBuffer(s_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_double), h_a, &ciErrNum);
    // if (ciErrNum != CL_SUCCESS) {
    //     cerr << "Error in clCreateBuffer for s_data_acc, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
    //     exit(EXIT_FAILURE);
    // }
    // s_data_res = clCreateBuffer(s_context, CL_MEM_READ_WRITE, sizeof(cl_double), NULL, &ciErrNum);
    // if (ciErrNum != CL_SUCCESS) {
    //     cerr <<"Error in clCreateBuffer for s_data_res, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
    //     exit(EXIT_FAILURE);
    // }

    return 0;
}

int LongAccumulator::CleanupOpenCL()
{
    cl_int ciErrNum;

    //Retrieving results...
    // ciErrNum = clEnqueueReadBuffer(cqCommandQueue, s_data_res, CL_TRUE, 0, sizeof(cl_double), &h_Res, 0, NULL, NULL);
    // if (ciErrNum != CL_SUCCESS) {
    //     cerr << "ciErrNum = << ciErrNum << "\n";
    //     cerr << "Error in clEnqueueReadBuffer Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
    //     return -1;
    // }

    if (!s_initializedOpenCL) {
        return 0;
    }

    // Release kernels and program
    //
    CleanupAcc();

    // Shutting down and freeing memory...
    //
    if (s_data_acc) {
        clReleaseMemObject(s_data_acc);
    }
    if (s_data_res) {
        clReleaseMemObject(s_data_res);
    }
    if (s_commandQueue) {
        clReleaseCommandQueue(s_commandQueue);
    }
    if (s_context) {
        clReleaseContext(s_context);
    }

    s_initializedOpenCL = false;

    return 0;
}

cl_int LongAccumulator::InitializeAcc(
    cl_context context,
    cl_command_queue commandQueue,
    cl_device_id device)
{
    cl_int ciErrNum;

    size_t sourceCodeLength = 0;
    char* sourceCode = nullptr;

    char path[256];
    strcpy(path, REPROFLOP_BINARY_DIR);
    strcat(path, "/include/cl/");
    strcat(path, LNGACC_KERNEL_FILENAME);

    // Read the OpenCL kernel in from source file
    //
    FILE *pProgramHandle = fopen(path, "r");
    if (!pProgramHandle) {
        cerr << "Failed to load kernel.\n";
        return -1;
    }
    fseek(pProgramHandle, 0, SEEK_END);
    sourceCodeLength = ftell(pProgramHandle);
    rewind(pProgramHandle);
    sourceCode = (char *) malloc(sourceCodeLength + 1);
    sourceCode[sourceCodeLength] = '\0';
    if (fread(sourceCode, sizeof(char), sourceCodeLength, pProgramHandle) != sourceCodeLength)
    {
        cerr << "Failed to read source code.\n";
        return -2;
    }
    fclose(pProgramHandle);

    s_initializedLngAcc = true;

    // Create OpenCL program from source
    //
    s_program = clCreateProgramWithSource(context, 1, (const char **)&sourceCode, &sourceCodeLength, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        cerr << "Error = " << ciErrNum << "\n";
        cerr << "Error in clCreateProgramWithSource, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        return -3;
    }

    // Building LongAccumulator program
    //
    ciErrNum = clBuildProgram(s_program, 0, NULL, compileOptions, NULL, NULL);
    if (ciErrNum != CL_SUCCESS) {
        cerr << "Error = " << ciErrNum << "\n";
        cerr << "Error in clBuildProgram, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";

        // Determine the reason for the error
        //
        char buildLog[16384];
        clGetProgramBuildInfo(s_program, s_device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), &buildLog, NULL);

        cout << buildLog << "\n";
        return -4;
    }

    // Create kernels
    //
    s_kernel = clCreateKernel(s_program, LNGACC_KERNEL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        cerr << "Error = " << ciErrNum << "\n";
        cerr << "Error in clCreateKernel: LongAccumulator, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        return -5;
    }
    s_complete = clCreateKernel(s_program, LNGACC_COMPLETE_KERNEL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        cerr << "Error = " << ciErrNum << "\n";
        cerr << "Error in clCreateKernel: LongAccumulatorComplete, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        return -6;
    }
    s_round = clCreateKernel(s_program, LNGACC_ROUND_KERNEL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        cerr << "Error = " << ciErrNum << "\n";
        cerr << "Error in clCreateKernel: LongAccumulatorRound, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
        return -7;
    }

    // Allocate internal buffers
    //
    // uint size = PARTIAL_SUPERACCS_COUNT * bin_count * sizeof(cl_long);
    // d_PartialSuperaccs = clCreateBuffer(s_context, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
    // if (ciErrNum != CL_SUCCESS) {
    //     printf("Error in clCreateBuffer for d_PartialSuperaccs, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
    //     return EXIT_FAILURE;
    // }
    // d_Superacc = clCreateBuffer(s_context, CL_MEM_READ_WRITE, bin_count * sizeof(bintype), NULL, &ciErrNum);
    // if (ciErrNum != CL_SUCCESS) {
    //     printf("Error in clCreateBuffer for d_Superacc, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
    //     return EXIT_FAILURE;
    // }

    // Deallocate temp storage
    //
    free(sourceCode);

    return 0;
}

cl_int LongAccumulator::CleanupAcc()
{
    cl_int ciErrNum;
    cl_int res = 0;

    if (!s_initializedLngAcc) {
        return 0;
    }

    if (s_data_acc) {
        ciErrNum = clReleaseMemObject(s_data_acc);
        if (ciErrNum != CL_SUCCESS) {
            cerr << "Error = " << ciErrNum << "\n";
            cerr << "Error in clReleaseMemObject, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
            res |= 1 << 0;
        }
    }

    if (s_kernel) {
        ciErrNum = clReleaseKernel(s_kernel);
        if (ciErrNum != CL_SUCCESS) {
            cerr << "Error = " << ciErrNum << "\n";
            cerr << "Error in clReleaseKernel: LongAccumulator, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
            res |= 1 << 1;
        }
    }

    if (s_complete) {
        ciErrNum = clReleaseKernel(s_complete);
        if (ciErrNum != CL_SUCCESS) {
            cerr << "Error = " << ciErrNum << "\n";
            cerr << "Error in clReleaseKernel: LongAccumulatorComplete, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
            res |= 1 << 2;
        }
    }

    if (s_round) {
        ciErrNum = clReleaseKernel(s_round);
        if (ciErrNum != CL_SUCCESS) {
            cerr << "Error = " << ciErrNum << "\n";
            cerr << "Error in clReleaseKernel: LongAccumulatorRound, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
            res |= 1 << 3;
        }
    }

    if (s_program) {
        ciErrNum = clReleaseProgram(s_program);
        if (ciErrNum != CL_SUCCESS) {
            cerr << "Error = " << ciErrNum << "\n";
            cerr << "Error in clReleaseProgram, Line " << __LINE__ << " in file " << __FILE__ << "!!!\n\n";
            res |= 1 << 4;
        }
    }

    return -res;
}

// #include <bitset>
// #include <cfenv>
// #include <cstring>
// #include <iomanip>
// #include <iostream>

// using namespace std;

// float_components extractComponents(float f)
// {
//     float_components components;
//     uint32_t tmp;
//     static_assert(sizeof(uint32_t) == sizeof(float));
//     memcpy(&tmp, &f, sizeof(float));
//     components.negative = tmp & 0x80000000;
//     components.exponent = (tmp >> 23) & 0xFF;
//     components.mantissa = tmp & 0x7FFFFF;
//     if (components.exponent != 0x00 && components.exponent != 0xFF) {
//         components.mantissa |= 0x800000; // hidden bit is equal to 1
//     }
//     return components;
// }

// float packToFloat(float_components components)
// {
//     uint32_t tmp = (components.negative ? 0x80000000 : 0x00000000) | (components.exponent << 23) |
//                    (components.mantissa & 0x7FFFFF);
//     float f;
//     static_assert(sizeof(uint32_t) == sizeof(float));
//     memcpy(&f, &tmp, sizeof(uint32_t));
//     return f;
// }

// LongAccumulator::LongAccumulator(float f)
// {
//     *this += f;
// }

// LongAccumulator &LongAccumulator::operator+=(const LongAccumulator &other)
// {
//     for (uint32_t i = 0; i < ACC_SIZE; ++i) {
//         add(i, other.acc[i], false);
//     }
//     return *this;
// }

// LongAccumulator &LongAccumulator::operator-=(const LongAccumulator &other)
// {
//     for (uint32_t i = 0; i < ACC_SIZE; ++i) {
//         add(i, other.acc[i], true);
//     }
//     return *this;
// }

// LongAccumulator &LongAccumulator::operator+=(float f)
// {
//     // Extract floating-point components - sign, exponent and mantissa
//     float_components components = extractComponents(f);
//     // Calculate the leftmost word that will be affected
//     uint32_t k = (components.exponent + 22) >> 5;
//     // Check if more than one word is affected (split mantissa)
//     bool split = ((components.exponent - 1) >> 5) < k;
//     // Calculate number of bits in the higher part of the mantissa
//     uint32_t hi_width = (components.exponent - 9) & 0x1F;
//     if (!split) {
//         add(k, components.mantissa << (hi_width - 24), components.negative);
//     } else {
//         add(k - 1, components.mantissa << (8 + hi_width), components.negative);
//         add(k, components.mantissa >> (24 - hi_width), components.negative);
//     }
//     return *this;
// }

// LongAccumulator &LongAccumulator::operator-=(float f)
// {
//     return *this += -f;
// }

// LongAccumulator &LongAccumulator::operator=(float f)
// {
//     acc = {};
//     return f == 0 ? *this : *this += f;
// }

// LongAccumulator LongAccumulator::operator+() const
// {
//     return *this;
// }

// LongAccumulator LongAccumulator::operator-() const
// {
//     LongAccumulator res;
//     res -= *this;
//     return res;
// }

// LongAccumulator operator+(LongAccumulator acc, const LongAccumulator &other)
// {
//     return acc += other;
// }

// LongAccumulator operator-(LongAccumulator acc, const LongAccumulator &other)
// {
//     return acc -= other;
// }

// LongAccumulator operator+(LongAccumulator acc, float f)
// {
//     return acc += f;
// }

// LongAccumulator operator-(LongAccumulator acc, float f)
// {
//     return acc -= f;
// }

// bool operator==(const LongAccumulator &l, const LongAccumulator &r)
// {
//     for (int i = 0; i < ACC_SIZE; ++i) {
//         if (l.acc[i] != r.acc[i]) {
//             return false;
//         }
//     }
//     return true;
// }

// bool operator!=(const LongAccumulator &l, const LongAccumulator &r)
// {
//     return !(l == r);
// }

// bool operator<(const LongAccumulator &l, const LongAccumulator &r)
// {
//     bool sign_l = l.acc[ACC_SIZE - 1] & 0x80000000;
//     bool sign_r = r.acc[ACC_SIZE - 1] & 0x80000000;
//     if (sign_l && !sign_r) {
//         return true;
//     } else if (!sign_l && sign_r) {
//         return false;
//     }
//     LongAccumulator positive_l = sign_l ? -l : l;
//     LongAccumulator positive_r = sign_r ? -r : r;
//     int i = ACC_SIZE - 1;
//     while (i >= 0 && positive_l.acc[i] == positive_r.acc[i]) {
//         i--;
//     }
//     if (i < 0) {
//         return false; // equal
//     }
//     return sign_l ? positive_l.acc[i] > positive_r.acc[i]
//                   : positive_l.acc[i] < positive_r.acc[i]; // for negative numbers invert
// }

// bool operator>(const LongAccumulator &l, const LongAccumulator &r)
// {
//     return r < l;
// }

// bool operator<=(const LongAccumulator &l, const LongAccumulator &r)
// {
//     return !(l > r);
// }

// bool operator>=(const LongAccumulator &l, const LongAccumulator &r)
// {
//     return !(l < r);
// }

// bool operator==(const LongAccumulator &l, float r)
// {
//     return l == LongAccumulator(r);
// }

// bool operator!=(const LongAccumulator &l, float r)
// {
//     return l != LongAccumulator(r);
// }

// bool operator<(const LongAccumulator &l, float r)
// {
//     return l < LongAccumulator(r);
// }

// bool operator>(const LongAccumulator &l, float r)
// {
//     return l > LongAccumulator(r);
// }

// bool operator<=(const LongAccumulator &l, float r)
// {
//     return l <= LongAccumulator(r);
// }

// bool operator>=(const LongAccumulator &l, float r)
// {
//     return l >= LongAccumulator(r);
// }

// ostream &operator<<(ostream &out, const LongAccumulator &acc)
// {
//     bool sign = acc.acc[ACC_SIZE - 1] & 0x80000000;
//     LongAccumulator positive_acc = sign ? -acc : acc;
//     int startIdx = ACC_SIZE - 1, endIdx = 0;
//     while (startIdx >= 0 && positive_acc.acc[startIdx] == 0) {
//         startIdx--;
//     }
//     if (startIdx < 4) {
//         startIdx = 4;
//     }
//     while (endIdx < ACC_SIZE && positive_acc.acc[endIdx] == 0) {
//         endIdx++;
//     }
//     if (endIdx > 4) {
//         endIdx = 4;
//     }
//     out << (sign ? "- " : "+ ");
//     for (int idx = startIdx; idx >= endIdx; --idx) {
//         bitset<32> bits(positive_acc.acc[idx]);
//         int startBit = 31, endBit = 0;
//         if (idx == startIdx) {
//             while (startBit >= 0 && !bits.test(startBit)) {
//                 startBit--;
//             }
//         }
//         if (idx == endIdx) {
//             while (endBit < 32 && !bits.test(endBit)) {
//                 endBit++;
//             }
//         }
//         if (idx == 4) {
//             if (startBit < 21) {
//                 startBit = 21; // include first bit before .
//             }
//             if (endBit > 20) {
//                 endBit = 20; // include first bit after .
//             }
//             for (int i = startBit; i > 20; --i) {
//                 out << bits.test(i);
//             }
//             out << " . ";
//             for (int i = 20; i >= endBit; --i) {
//                 out << bits.test(i);
//             }
//         } else {
//             for (int i = startBit; i >= endBit; --i) {
//                 out << bits.test(i);
//             }
//         }
//         if (idx > endIdx) {
//             out << ' ';
//         }
//     }
//     return out;
// }

// float LongAccumulator::operator()()
// {
//     float_components components;

//     components.negative = acc[ACC_SIZE - 1] & 0x80000000;
//     components.exponent = 0;
//     components.mantissa = 0;

//     LongAccumulator absolute = components.negative ? -*this : *this;

//     // Calculate the position of the most significant non-zero word...
//     int word_idx = ACC_SIZE - 1;
//     while (word_idx >= 0 && absolute.acc[word_idx] == 0) {
//         word_idx--;
//     }

//     // Check if zero...
//     if (word_idx < 0) {
//         return packToFloat(components);
//     }

//     // Calculate the position of the most significant bit...
//     bitset<32> bits(absolute.acc[word_idx]);
//     int bit_idx = 31;
//     while (bit_idx >= 0 && !bits.test(bit_idx)) {
//         bit_idx--;
//     }

//     // Check if subnormal...
//     if (word_idx == 0 && bit_idx < 23) {
//         components.mantissa = absolute.acc[0];
//         return packToFloat(components);
//     }

//     // Calculate the exponent using the inverse of the formula used for "sliding" the mantissa into the accumulator.
//     components.exponent = (word_idx << 5) + bit_idx - 22;

//     // Check if infinity...
//     if (components.exponent > 0xFE) {
//         components.exponent = 0xFF;
//         return packToFloat(components);
//     }

//     // Extract bits of mantissa from current word...
//     uint32_t mask = ((uint64_t) 1 << (bit_idx + 1)) - 1;
//     components.mantissa = absolute.acc[word_idx] & mask;

//     // Extract mantissa and round according to currently selected rounding mode...
//     if (bit_idx > 23) {
//         components.mantissa >>= bit_idx - 23;
//         round(absolute, components, word_idx, bit_idx - 24);
//     } else if (bit_idx == 23) {
//         round(absolute, components, word_idx - 1, 31);
//     } else {
//         components.mantissa <<= 23 - bit_idx;
//         components.mantissa |= absolute.acc[word_idx - 1] >> (bit_idx + 9);
//         round(absolute, components, word_idx - 1, bit_idx + 8);
//     }

//     return packToFloat(components);
// }

// void LongAccumulator::add(uint32_t idx, uint32_t val, bool negative)
// {
//     while (idx < ACC_SIZE) {
//         uint32_t old = acc[idx];
//         if (!negative) {
//             acc[idx] += val;
//             if (acc[idx] < old) { // overflow
//                 ++idx;
//                 val = 1;
//             } else break;
//         } else {
//             acc[idx] -= val;
//             if (acc[idx] > old) { // underflow
//                 ++idx;
//                 val = 1;
//             } else break;
//         }
//     }
// }

// void LongAccumulator::round(const LongAccumulator &acc, float_components &components, int word_idx, int bit_idx)
// {
//     int rounding_mode = fegetround();
//     // negative if cannot be determined
//     if (rounding_mode < 0) {
//         rounding_mode = FE_TONEAREST;
//     }
//     bitset<32> bits(acc.acc[word_idx]);
//     if (rounding_mode == FE_TONEAREST) {
//         if (!bits.test(bit_idx)) {
//             return; // case 1
//         }
//         bool allzero = true;
//         for (int i = bit_idx - 1; allzero && i >= 0; --i) {
//             if (bits.test(i)) {
//                 allzero = false;
//             }
//         }
//         if (allzero) {
//             for (int i = word_idx - 1; allzero && i >= 0; --i) {
//                 if (acc.acc[i] > 0) {
//                     allzero = false;
//                 }
//             }
//         }
//         if (!allzero) {
//             components.mantissa++; // case 2, need to check for overflow
//         } else if ((components.mantissa & 0x1) == 0) {
//             return; // case 3a
//         } else {
//             components.mantissa++; // case 3b, need to check for overflow
//         }
//     } else if (rounding_mode == FE_UPWARD && !components.negative ||
//                rounding_mode == FE_DOWNWARD && components.negative) {
//         bool allzero = true;
//         for (int i = bit_idx; allzero && i >= 0; --i) {
//             if (bits.test(i)) {
//                 allzero = false;
//             }
//         }
//         if (allzero) {
//             for (int i = word_idx - 1; allzero && i >= 0; --i) {
//                 if (acc.acc[i] > 0) {
//                     allzero = false;
//                 }
//             }
//         }
//         if (!allzero) {
//             components.mantissa++; // case 1, need to check for overflow
//         } else {
//             return; // case 2
//         }
//     } else {
//         // FE_TOWARDSZERO - always ignores other bits
//         // FE_UPWARD while negative - since mantissa is unsigned, always returns lower value (ignores other bits)
//         // FE_DOWNWARD while positive - same as FE_TOWARDSZERO in this case
//         return;
//     }
//     // Check if rounding caused overflow...
//     if (components.mantissa > 0xFFFFFF) {
//         components.mantissa >>= 1;
//         components.exponent++;
//         if (components.exponent > 0xFE) {
//             components.exponent = 0xFF;
//             components.mantissa = 0;
//         }
//     }
// }