#ifndef LONGACCUMULATOR_H
#define LONGACCUMULATOR_H

#include <cstdint>

// Floating-point specification constants

constexpr int EXPONENT_MIN_VALUE = -126;
constexpr int EXPONENT_MAX_VALUE = 127;
constexpr int EXPONENT_BIAS = 127;

// Long accumulator constants

constexpr uint32_t ACC_SIZE = 9;
constexpr uint32_t RADIX_LOCATION = 20;

typedef struct
{
    bool negative;
    uint32_t exponent;
    uint32_t mantissa;
} float_components;

float_components extractComponents(float f);
float packToFloat(float_components components);

class LongAccumulator
{
public:
    LongAccumulator();
    ~LongAccumulator();

    void clear();

    void add(float f);
    void add(LongAccumulator &acc);

    float roundToFloat();

    void print();
private:
    uint32_t *acc;

    // Adds the value to the word at index idx with carry/borrow
    void add(uint32_t idx, uint32_t val, bool negative);
};

#endif