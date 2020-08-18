#include "LongAccumulator.h"
#include <cstring>
#include <iostream>
#include <iomanip>
#include <bitset>
#include <limits>

using namespace std;

float_components extractComponents(float f)
{
    float_components components;
    uint32_t tmp;
    static_assert(sizeof(uint32_t) == sizeof(float));
    memcpy(&tmp, &f, sizeof(float));
    components.negative = tmp & 0x80000000;
    components.exponent = (tmp >> 23) & 0xFF;
    components.mantissa = tmp & 0x7FFFFF;
    if (components.exponent != 0x00 && components.exponent != 0xFF) {
        components.mantissa |= 0x800000; // hidden bit is equal to 1
    }
    cout << "Float: " << fixed << setprecision(10) << f << " (" << bitset<32>(tmp)
        << ") Sign: " << components.negative
        << " Exponent: " << components.exponent << " (" << bitset<8>(components.exponent)
        << ") Mantissa: " << components.mantissa << " (" << bitset<24>(components.mantissa) << ')' << endl;
    return components;
}

float packToFloat(float_components components)
{
    uint32_t tmp = (components.negative ? 0x80000000 : 0x00000000) |
        (components.exponent << 23) |
        (components.mantissa & 0x7FFFFF);
    float f;
    static_assert(sizeof(uint32_t) == sizeof(float));
    memcpy(&f, &tmp, sizeof(uint32_t));
    return f;
}

LongAccumulator::LongAccumulator()
{
    acc = new uint32_t[ACC_SIZE];
    clear();
    // cout << "Constructor!" << endl;
}

LongAccumulator::~LongAccumulator()
{
    delete acc;
    // cout << "Destructor!" << endl;
}

void LongAccumulator::clear()
{
    for (uint32_t i = 0; i < ACC_SIZE; ++i) {
        acc[i] = 0;
    }
}

void LongAccumulator::add(float f)
{
    // Extract floating-point components - sign, exponent and mantissa
    float_components components = extractComponents(f);
    // Calculate the leftmost word that will be affected
    uint32_t k = (components.exponent + 22) >> 5;
    // Check if more than one word is affected (split mantissa)
    bool split = ((components.exponent - 1) >> 5) < k;
    cout << "k = " << k << " split: " << split << endl;
    if (!split) {
        add(k, components.mantissa, components.negative);
    } else {
        // Calculate number of bits in the higher part of the mantissa
        uint32_t hi_width = (components.exponent - 9) % 32;
        // Shift mantissa left for lower bits
        uint32_t lo = components.mantissa << (8 + hi_width);
        add(k - 1, lo, components.negative);
        // Shift mantissa right for higher bits
        uint32_t hi = components.mantissa >> (24 - hi_width);
        add(k, hi, components.negative);
        cout << "hi_width = " << hi_width << " hi: " << bitset<24>(hi) << " lo: ";
        bitset<32> bits_lo(lo);
        for (int i = 31; i >= 8 + hi_width; --i) {
            cout << bits_lo.test(i);
        }
        cout << endl;
    }
}

void LongAccumulator::add(LongAccumulator &acc)
{
    for (uint32_t i = 0; i < ACC_SIZE; ++i) {
        add(i, acc.acc[i], false);
    }
}

float LongAccumulator::roundToFloat()
{
    float_components components;
    components.negative = acc[ACC_SIZE - 1] & 0x80000000;
    components.exponent = 0;
    components.mantissa = 0;
    int k = ACC_SIZE - 1;
    while (k >= 0) {
        if (components.negative) {
            if (acc[k] == 0xFFFFFFFF) k--;
            else break;
        } else {
            if (acc[k] == 0x00000000) k--;
            else break;
        }
    }
    if (k < 0) {
        // +0 or -0 depending on the sign which is already set
        // exponent and mantissa have values 0
        return packToFloat(components);
    }
    int i = 31;
    bitset<32> bits(acc[k]);
    while (i >= 0) {
        // For positive numbers we look for bit 1, for negative - bit 0
        if (bits.test(i) ^ components.negative) break;
        else i--;
    }
    // TODO: Implement rounding to preserve accuracy!!!
    if (k == 0 && i < 23) {
        // Subnormal - mantissa "narrower than 24b", exponent is 0
        components.mantissa = acc[0] & 0xFFFFFF;
        // Needs to be adjusted for sign (inverted if negative)...
    } else {
        // Inverse of the formula used for "sliding" the number into the accumulator
        components.exponent = (k << 5) + i - 22;
        if (components.exponent > 0xFE) {
            // Infinity - mantissa is 0, exponent is 0xFF (255)
            components.exponent = 0xFF;
        } else {
            // cout << "k = " << k << " i = " << i << " acc[k] = " << bitset<32>(acc[k]) << endl;
            uint32_t mask = (1 << (i + 1)) - 1;
            components.mantissa = acc[k] & mask;
            // cout << "mask = " << bitset<32>(mask) << " acc[k] & mask = " << components.mantissa << endl;
            if (i > 23) {
                components.mantissa >>= i - 23;
                // Check remaining bits to round the number...
            } else {
                components.mantissa <<= 23 - i;
                // cout << "current mantissa: " << bitset<24>(components.mantissa) << endl;
                // cout << "acc[k - 1] = " << bitset<32>(acc[k - 1]) << endl;
                // cout << "acc[k - 1] >> x = " << bitset<32>(acc[k - 1] >> (i + 9)) << endl;
                components.mantissa |= acc[k - 1] >> (i + 9);
                // cout << "final mantissa: " << bitset<24>(components.mantissa) << endl;
            }
        }
    }
    // cout << "k = " << k << " i = " << i << " exponent = " << components.exponent << " mantissa = " << bitset<24>(components.mantissa) << endl;
    if (components.negative) {
        // Invert all bits in mantissa until least significant 1
        bitset<24> mb(components.mantissa);
        int i = 0;
        while (i < 24 && !mb.test(i)) i++;
        i++; // skip least significant 1
        for (; i < 24; ++i) mb.set(i, !mb.test(i));
        // cout << "Inverted: " << mb << endl;
        components.mantissa = mb.to_ulong();
    }

    return packToFloat(components);
}

void LongAccumulator::print()
{
    bool negative = acc[ACC_SIZE - 1] & 0x80000000;
    int start = ACC_SIZE - 1;
    while (start >= 0) {
        if (negative) {
            if (acc[start] == 0xFFFFFFFF) start--;
            else break;
        } else {
            if (acc[start] == 0x00000000) start--;
            else break;
        }
    }
    int end = 0;
    while (end < ACC_SIZE) {
        if (acc[end] == 0x00000000) end++;
        else break;
    }

    for (int i = start; i >= end; --i) {
        bitset<32> bits(acc[i]);
        if (i == 4) {
            for (int i = 31; i > RADIX_LOCATION; --i) {
                cout << bits.test(i);
            }
            cout << " . ";
            for (int i = RADIX_LOCATION; i >= 0; --i) {
                cout << bits.test(i);
            }
            cout << ' ';
        } else {
            cout << bits << ' '; // << endl;
        }
    }
    cout << endl;
}

void LongAccumulator::add(uint32_t idx, uint32_t val, bool negative)
{
    if (idx < 0 || idx >= ACC_SIZE) return;
    uint32_t old = acc[idx];
    if (!negative) {
        acc[idx] += val;
        if (acc[idx] < old) { // overflow
            add(idx + 1, 1, false);
        }
    } else {
        acc[idx] -= val;
        if (acc[idx] > old) { // underflow
            add(idx + 1, 1, true);
        }
    }
}