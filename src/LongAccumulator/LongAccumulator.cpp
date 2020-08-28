#include "LongAccumulator.h"

#include <bitset>
#include <cfenv>
#include <cstring>
#include <iomanip>
#include <iostream>

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
    out << (sign ? "- " : "+ ");
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
        } else {
            for (int i = startBit; i >= endBit; --i) out << bits.test(i);
        }
        if (idx > endIdx) out << ' ';
    }
    return out;
}

float LongAccumulator::operator()()
{
    float_components components;

    components.negative = acc[ACC_SIZE - 1] & 0x80000000;
    components.exponent = 0;
    components.mantissa = 0;

    LongAccumulator absolute = components.negative ? -*this : *this;

    // Calculate the position of the most significant non-zero word...
    int word_idx = ACC_SIZE - 1;
    while (word_idx >= 0 && absolute.acc[word_idx] == 0) word_idx--;

    // Check if zero...
    if (word_idx < 0) {
        return packToFloat(components);
    }

    // Calculate the position of the most significant bit...
    bitset<32> bits(absolute.acc[word_idx]);
    int bit_idx = 31;
    while (bit_idx >= 0 && !bits.test(bit_idx)) bit_idx--;

    // Check if subnormal...
    if (word_idx == 0 && bit_idx < 23) {
        components.mantissa = absolute.acc[0];
        return packToFloat(components);
    }

    // Calculate the exponent using the inverse of the formula used for "sliding" the mantissa into the accumulator.
    components.exponent = (word_idx << 5) + bit_idx - 22;

    // Check if infinity...
    if (components.exponent > 0xFE) {
        components.exponent = 0xFF;
        return packToFloat(components);
    }

    // Extract bits of mantissa from current word...
    uint32_t mask = ((uint64_t) 1 << (bit_idx + 1)) - 1;
    components.mantissa = absolute.acc[word_idx] & mask;

    // Extract mantissa and round according to currently selected rounding mode...
    if (bit_idx > 23) {
        components.mantissa >>= bit_idx - 23;
        round(absolute, components, word_idx, bit_idx - 24);
    } else if (bit_idx == 23) {
        round(absolute, components, word_idx - 1, 31);
    } else {
        components.mantissa <<= 23 - bit_idx;
        components.mantissa |= absolute.acc[word_idx - 1] >> (bit_idx + 9);
        round(absolute, components, word_idx - 1, bit_idx + 8);
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

void LongAccumulator::round(const LongAccumulator& acc, float_components& components, int word_idx, int bit_idx)
{
    int rounding_mode = fegetround();
    // negative if cannot be determined
    if (rounding_mode < 0) rounding_mode = FE_TONEAREST;
    bitset<32> bits(acc.acc[word_idx]);
    if (rounding_mode == FE_TONEAREST) {
        if (!bits.test(bit_idx)) return; // case 1
        bool allzero = true;
        for (int i = bit_idx - 1; allzero && i >= 0; --i) if (bits.test(i)) allzero = false;
        if (allzero) {
            for (int i = word_idx - 1; allzero && i >= 0; --i) if (acc.acc[i] > 0) allzero = false;
        }
        if (!allzero) {
            components.mantissa++; // case 2, need to check for overflow
        } else if ((components.mantissa & 0x1) == 0) {
            return; // case 3a
        } else {
            components.mantissa++; // case 3b, need to check for overflow
        }
    } else if (rounding_mode == FE_UPWARD && !components.negative ||
                rounding_mode == FE_DOWNWARD && components.negative) {
        bool allzero = true;
        for (int i = bit_idx; allzero && i >= 0; --i) if (bits.test(i)) allzero = false;
        if (allzero) {
            for (int i = word_idx - 1; allzero && i >= 0; --i) if (acc.acc[i] > 0) allzero = false;
        }
        if (!allzero) {
            components.mantissa++; // case 1, need to check for overflow
        } else {
            return; // case 2
        }
    } else {
        // FE_TOWARDSZERO - always ignores other bits
        // FE_UPWARD while negative - since mantissa is unsigned, always returns lower value (ignores other bits)
        // FE_DOWNWARD while positive - same as FE_TOWARDSZERO in this case
        return;
    }
    // Check if rounding caused overflow...
    if (components.mantissa > 0x7FFFFF) {
        components.mantissa >>= 1;
        components.exponent++;
        if (components.exponent > 0xFE) {
            components.exponent = 0xFF;
            components.mantissa = 0;
        }
    }
}