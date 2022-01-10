#ifndef LONGACCUMULATOR_H
#define LONGACCUMULATOR_H

#include "common.h"

class LongAccumulator
{
public:
    static int InitializeOpenCL();
    static int CleanupOpenCL();

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
    static cl_kernel s_kernel;
    static cl_kernel s_complete;
    static cl_kernel s_round;

    static cl_mem s_data_acc;

    static cl_mem s_data_res;
};

// #include <array>
// #include <cstdint>
// #include <iostream>

// constexpr uint32_t ACC_SIZE = 9;

// typedef struct {
//     bool negative;
//     uint32_t exponent;
//     uint32_t mantissa;
// } float_components;

// float_components extractComponents(float f);
// float packToFloat(float_components components);

// class LongAccumulator
// {
//   public:
//     LongAccumulator() {}
//     LongAccumulator(float f);

//     LongAccumulator &operator+=(const LongAccumulator &other);
//     LongAccumulator &operator-=(const LongAccumulator &other);

//     LongAccumulator &operator+=(float f);
//     LongAccumulator &operator-=(float f);

//     LongAccumulator &operator=(float f);

//     LongAccumulator operator+() const;
//     LongAccumulator operator-() const;

//     friend LongAccumulator operator+(LongAccumulator acc, const LongAccumulator &other);
//     friend LongAccumulator operator-(LongAccumulator acc, const LongAccumulator &other);

//     friend LongAccumulator operator+(LongAccumulator acc, float f);
//     friend LongAccumulator operator-(LongAccumulator acc, float f);

//     friend bool operator==(const LongAccumulator &l, const LongAccumulator &r);
//     friend bool operator!=(const LongAccumulator &l, const LongAccumulator &r);

//     friend bool operator< (const LongAccumulator &l, const LongAccumulator &r);
//     friend bool operator> (const LongAccumulator &l, const LongAccumulator &r);
//     friend bool operator<=(const LongAccumulator &l, const LongAccumulator &r);
//     friend bool operator>=(const LongAccumulator &l, const LongAccumulator &r);

//     friend bool operator==(const LongAccumulator &l, float r);
//     friend bool operator!=(const LongAccumulator &l, float r);

//     friend bool operator< (const LongAccumulator &l, float r);
//     friend bool operator> (const LongAccumulator &l, float r);
//     friend bool operator<=(const LongAccumulator &l, float r);
//     friend bool operator>=(const LongAccumulator &l, float r);

//     friend std::ostream &operator<<(std::ostream &out, const LongAccumulator &acc);

//     float operator()(); // returns float value of this accumulator
//   private:
//     std::array<uint32_t, ACC_SIZE> acc {};

//     // Adds the value to the word at index idx with carry/borrow
//     void add(uint32_t idx, uint32_t val, bool negative);

//     // Rounds mantissa according to currently selected rounding mode
//     void round(const LongAccumulator &acc, float_components &components, int word_idx, int bit_idx);
// };

#endif  // LONGACCUMULATOR_H