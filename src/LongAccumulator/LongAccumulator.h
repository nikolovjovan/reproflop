#ifndef LONGACCUMULATOR_H
#define LONGACCUMULATOR_H

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

class LongAccumulator
{
  public:
    LongAccumulator() {}
    LongAccumulator(float f);

    LongAccumulator &operator+=(const LongAccumulator &other);
    LongAccumulator &operator-=(const LongAccumulator &other);

    LongAccumulator &operator+=(float f);
    LongAccumulator &operator-=(float f);

    LongAccumulator &operator=(float f);

    LongAccumulator operator+() const;
    LongAccumulator operator-() const;

    friend LongAccumulator operator+(LongAccumulator acc, const LongAccumulator &other);
    friend LongAccumulator operator-(LongAccumulator acc, const LongAccumulator &other);

    friend LongAccumulator operator+(LongAccumulator acc, float f);
    friend LongAccumulator operator-(LongAccumulator acc, float f);

    friend bool operator==(const LongAccumulator &l, const LongAccumulator &r);
    friend bool operator!=(const LongAccumulator &l, const LongAccumulator &r);

    friend bool operator< (const LongAccumulator &l, const LongAccumulator &r);
    friend bool operator> (const LongAccumulator &l, const LongAccumulator &r);
    friend bool operator<=(const LongAccumulator &l, const LongAccumulator &r);
    friend bool operator>=(const LongAccumulator &l, const LongAccumulator &r);

    friend bool operator==(const LongAccumulator &l, float r);
    friend bool operator!=(const LongAccumulator &l, float r);

    friend bool operator< (const LongAccumulator &l, float r);
    friend bool operator> (const LongAccumulator &l, float r);
    friend bool operator<=(const LongAccumulator &l, float r);
    friend bool operator>=(const LongAccumulator &l, float r);

    friend std::ostream &operator<<(std::ostream &out, const LongAccumulator &acc);

    float operator()(); // returns float value of this accumulator
  private:
    std::array<uint32_t, ACC_SIZE> acc {};

    // Adds the value to the word at index idx with carry/borrow
    void add(uint32_t idx, uint32_t val, bool negative);

    // Rounds mantissa according to currently selected rounding mode
    void round(const LongAccumulator &acc, float_components &components, int word_idx, int bit_idx);
};

#endif