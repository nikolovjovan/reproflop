#ifndef LONGACCUMULATORCPU_H
#define LONGACCUMULATORCPU_H

#include <array>
#include <cstdint>
#include <iostream>

constexpr uint32_t ACC_SIZE = 9;

typedef struct {
    bool negative;
    uint32_t exponent;
    uint32_t mantissa;
} float_components;

float_components extractComponents(float f);
float packToFloat(float_components components);

class LongAccumulatorCPU
{
public:
    LongAccumulatorCPU() {}
    LongAccumulatorCPU(float f);
    LongAccumulatorCPU(uint32_t *gpuAcc);

    LongAccumulatorCPU &operator+=(const LongAccumulatorCPU &other);
    LongAccumulatorCPU &operator-=(const LongAccumulatorCPU &other);

    LongAccumulatorCPU &operator+=(float f);
    LongAccumulatorCPU &operator-=(float f);

    LongAccumulatorCPU &operator=(float f);

    LongAccumulatorCPU operator+() const;
    LongAccumulatorCPU operator-() const;

    friend LongAccumulatorCPU operator+(LongAccumulatorCPU acc, const LongAccumulatorCPU &other);
    friend LongAccumulatorCPU operator-(LongAccumulatorCPU acc, const LongAccumulatorCPU &other);

    friend LongAccumulatorCPU operator+(LongAccumulatorCPU acc, float f);
    friend LongAccumulatorCPU operator-(LongAccumulatorCPU acc, float f);

    friend bool operator==(const LongAccumulatorCPU &l, const LongAccumulatorCPU &r);
    friend bool operator!=(const LongAccumulatorCPU &l, const LongAccumulatorCPU &r);

    friend bool operator< (const LongAccumulatorCPU &l, const LongAccumulatorCPU &r);
    friend bool operator> (const LongAccumulatorCPU &l, const LongAccumulatorCPU &r);
    friend bool operator<=(const LongAccumulatorCPU &l, const LongAccumulatorCPU &r);
    friend bool operator>=(const LongAccumulatorCPU &l, const LongAccumulatorCPU &r);

    friend bool operator==(const LongAccumulatorCPU &l, float r);
    friend bool operator!=(const LongAccumulatorCPU &l, float r);

    friend bool operator< (const LongAccumulatorCPU &l, float r);
    friend bool operator> (const LongAccumulatorCPU &l, float r);
    friend bool operator<=(const LongAccumulatorCPU &l, float r);
    friend bool operator>=(const LongAccumulatorCPU &l, float r);

    friend std::ostream &operator<<(std::ostream &out, const LongAccumulatorCPU &acc);

    float operator()(); // returns float value of this accumulator

private:
    std::array<uint32_t, ACC_SIZE> acc {};

    // Adds the value to the word at index idx with carry/borrow
    void add(uint32_t idx, uint32_t val, bool negative);

    // Rounds mantissa according to currently selected rounding mode
    void round(const LongAccumulatorCPU &acc, float_components &components, int word_idx, int bit_idx);
};

#endif  // LONGACCUMULATORCPU_H