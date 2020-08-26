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
    // cout << "Float: " << fixed << setprecision(10) << f << " (" << bitset<32>(tmp)
    //     << ") Sign: " << components.negative
    //     << " Exponent: " << components.exponent << " (" << bitset<8>(components.exponent)
    //     << ") Mantissa: " << components.mantissa << " (" << bitset<24>(components.mantissa) << ')' << endl;
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

LongAccumulator::LongAccumulator(float f)
{
    *this += f;
}

LongAccumulator& LongAccumulator::operator+=(const LongAccumulator& other)
{
    for (uint32_t i = 0; i < ACC_SIZE; ++i) {
        add(i, other.acc[i], false);
    }
    return *this;
}

LongAccumulator& LongAccumulator::operator-=(const LongAccumulator& other)
{
    for (uint32_t i = 0; i < ACC_SIZE; ++i) {
        add(i, other.acc[i], true);
    }
    return *this;
}

LongAccumulator& LongAccumulator::operator+=(float f)
{
    // Extract floating-point components - sign, exponent and mantissa
    float_components components = extractComponents(f);
    // Calculate the leftmost word that will be affected
    uint32_t k = (components.exponent + 22) >> 5;
    // Check if more than one word is affected (split mantissa)
    bool split = ((components.exponent - 1) >> 5) < k;
    // Calculate number of bits in the higher part of the mantissa
    uint32_t hi_width = (components.exponent - 9) % 32;
    if (!split) {
        add(k, components.mantissa << (hi_width - 24), components.negative);
    } else {
        add(k - 1, components.mantissa << (8 + hi_width), components.negative);
        add(k, components.mantissa >> (24 - hi_width), components.negative);
    }
    return *this;
}

LongAccumulator& LongAccumulator::operator-=(float f)
{
    return *this += -f;
}

LongAccumulator& LongAccumulator::operator=(float f)
{
    acc = {};
    return f == 0 ? *this : *this += f;
}

LongAccumulator LongAccumulator::operator+() const
{
    return *this;
}

LongAccumulator LongAccumulator::operator-() const
{
    LongAccumulator res;
    res -= *this;
    return res;
}

LongAccumulator operator+(LongAccumulator acc, const LongAccumulator& other)
{
    return acc += other;
}

LongAccumulator operator-(LongAccumulator acc, const LongAccumulator& other)
{
    return acc -= other;
}

LongAccumulator operator+(LongAccumulator acc, float f)
{
    return acc += f;
}

LongAccumulator operator-(LongAccumulator acc, float f)
{
    return acc -= f;
}

bool operator==(const LongAccumulator& l, const LongAccumulator& r)
{
    for (int i = 0; i < ACC_SIZE; ++i) {
        if (l.acc[i] != r.acc[i]) {
            return false;
        }
    }
    return true;
}

bool operator!=(const LongAccumulator& l, const LongAccumulator& r)
{
    return !(l == r);
}

bool operator< (const LongAccumulator& l, const LongAccumulator& r)
{
    bool sign_l = l.acc[ACC_SIZE - 1] & 0x80000000;
    bool sign_r = r.acc[ACC_SIZE - 1] & 0x80000000;
    if (sign_l && !sign_r) return true;
    else if (!sign_l && sign_r) return false;
    LongAccumulator positive_l = sign_l ? -l : l;
    LongAccumulator positive_r = sign_r ? -r : r;
    int i = ACC_SIZE - 1;
    while (i >= 0 && positive_l.acc[i] == positive_r.acc[i]) i--;
    if (i < 0) return false; // equal
    return sign_l ? positive_l.acc[i] > positive_r.acc[i] : positive_l.acc[i] < positive_r.acc[i]; // for negative numbers invert
}

bool operator> (const LongAccumulator& l, const LongAccumulator& r)
{
    return r < l;
}

bool operator<=(const LongAccumulator& l, const LongAccumulator& r)
{
    return !(l > r);
}

bool operator>=(const LongAccumulator& l, const LongAccumulator& r)
{
    return !(l < r);
}

bool operator==(const LongAccumulator& l, float r)
{
    return l == LongAccumulator(r);
}

bool operator!=(const LongAccumulator& l, float r)
{
    return l != LongAccumulator(r);
}

bool operator< (const LongAccumulator& l, float r)
{
    return l < LongAccumulator(r);
}

bool operator> (const LongAccumulator& l, float r)
{
    return l > LongAccumulator(r);
}

bool operator<=(const LongAccumulator& l, float r)
{
    return l <= LongAccumulator(r);
}

bool operator>=(const LongAccumulator& l, float r)
{
    return l >= LongAccumulator(r);
}

ostream& operator<<(ostream& out, const LongAccumulator& acc)
{
    bool sign = acc.acc[ACC_SIZE - 1] & 0x80000000;
    LongAccumulator positive_acc = sign ? -acc : acc;
    int startIdx = ACC_SIZE - 1, endIdx = 0;
    while (startIdx >= 0 && positive_acc.acc[startIdx] == 0) startIdx--;
    if (startIdx < 4) startIdx = 4;
    while (endIdx < ACC_SIZE && positive_acc.acc[endIdx] == 0) endIdx++;
    if (endIdx > 4) endIdx = 4;
    out << (sign ? '-' : '+');
    for (int idx = startIdx; idx >= endIdx; --idx) {
        bitset<32> bits(positive_acc.acc[idx]);
        int startBit = 31, endBit = 0;
        if (idx == startIdx) while (startBit >= 0 && !bits.test(startBit)) startBit--;
        if (idx == endIdx) while (endBit < 32 && !bits.test(endBit)) endBit++;
        if (idx == 4) {
            if (startBit < 21) startBit = 21; // include first bit before .
            if (endBit > 20) endBit = 20; // include first bit after .
            for (int i = startBit; i > 20; --i) out << bits.test(i);
            out << " . ";
            for (int i = 20; i >= endBit; --i) out << bits.test(i);
            out << ' ';
        } else {
            out << bits << ' ';
        }
    }
    return out;
}

float LongAccumulator::operator()()
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
            uint32_t mask = ((uint64_t) 1 << (i + 1)) - 1;
            components.mantissa = acc[k] & mask;
            // cout << "mask = " << bitset<32>(mask) << " acc[k] & mask = " << bitset<32>(components.mantissa) << endl;
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