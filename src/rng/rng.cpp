#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <pthread.h>

#define DEFAULT_THREAD_NUM  (8)
#define DEFAULT_ARRAY_SIZE  (1000)
#define DEFAULT_SEED        (1549813198)
#define DEFAULT_MIN_EXP     (0)
#define DEFAULT_MAX_EXP     (127)

#define SMALLEST_SUBNORMAL          (0b0'00000000'00000000000000000000001)
#define LARGEST_SUBNORMAL           (0b0'00000000'11111111111111111111111)

#define SMALLEST_NORMAL             (0b0'00000001'00000000000000000000000)
#define LARGEST_NORMAL              (0b0'11111110'11111111111111111111111)

#define LARGEST_LESS_THAN_ONE       (0b0'01111110'11111111111111111111111)
#define ONE                         (0b0'01111111'00000000000000000000000)
#define SMALLEST_LARGER_THAN_ONE    (0b0'01111110'11111111111111111111111)

#define MINUS_TWO                   (0b1'10000000'00000000000000000000000)

#define POSITIVE_ZERO               (0b0'00000000'00000000000000000000000)
#define NEGATIVE_ZERO               (0b1'00000000'00000000000000000000000)

#define POSITIVE_INFINITY           (0b0'11111111'00000000000000000000000)
#define NEGATIVE_INFINITY           (0b1'11111111'00000000000000000000000)

#define POSITIVE_QNAN               (0b0'11111111'10000000000000000000001)
#define NEGATIVE_QNAN               (0b1'11111111'10000000000000000000001)
#define POSITIVE_SNAN               (0b0'11111111'00000000000000000000001)
#define NEGATIVE_SNAN               (0b1'11111111'00000000000000000000001)

pthread_t *threads;

void print_float_in_binary(uint32_t f) {
    uint32_t sign = (f >> 31) & 0b1;
    printf("Sign: %u (%c) ", sign, (sign == 0 ? '+' : '-'));
    printf("Exponent: ");
    uint32_t exponent = (f >> 23) & 0xFF;
    for (int i = 0; i < 8; ++i) {
        printf("%u", (exponent >> (8 - i - 1)) & 0b1);
    }
    printf(" (%u) ", exponent);
    uint32_t mantisa = f & 0x7FFF;
    for (int i = 0; i < 23; ++i) {
        printf("%u", (mantisa >> (23 - i - 1)) & 0b1);
    }
    printf(" (%u) ", ((0b1 << 24) | mantisa));
}

int main(int argc, char *argv[])
{
    uint32_t buf;

    buf = SMALLEST_SUBNORMAL;
    printf("Smallest subnormal: [%.10e]\n", reinterpret_cast<float&>(buf));
    buf = LARGEST_SUBNORMAL;
    printf("Largest subnormal: [%.10e]\n", reinterpret_cast<float&>(buf));

    buf = SMALLEST_NORMAL;
    printf("Smallest normal: [%.10e]\n", reinterpret_cast<float&>(buf));
    buf = LARGEST_NORMAL;
    printf("Largest normal: [%.10e]\n", reinterpret_cast<float&>(buf));

    buf = LARGEST_LESS_THAN_ONE;
    printf("Largest less than one (1.0): [%.10e]\n", reinterpret_cast<float&>(buf));
    buf = ONE;
    printf("One (1.0): [%.10e]\n", reinterpret_cast<float&>(buf));
    buf = SMALLEST_LARGER_THAN_ONE;
    printf("Smallest larger than one (1.0): [%.10e]\n", reinterpret_cast<float&>(buf));

    buf = MINUS_TWO;
    printf("Minus two (-2.0): [%.10e]\n", reinterpret_cast<float&>(buf));

    buf = POSITIVE_ZERO;
    printf("Zero (positive): [%.10e]\n", reinterpret_cast<float&>(buf));
    buf = NEGATIVE_ZERO;
    printf("Negative zero: [%.10e]\n", reinterpret_cast<float&>(buf));

    buf = POSITIVE_INFINITY;
    printf("Infinity (positive): [%.10e]\n", reinterpret_cast<float&>(buf));
    buf = NEGATIVE_INFINITY;
    printf("Negative infinity: [%.10e]\n", reinterpret_cast<float&>(buf));

    buf = POSITIVE_QNAN;
    printf("qNaN: [%.10e]\n", reinterpret_cast<float&>(buf));
    buf = NEGATIVE_QNAN;
    printf("'Negative' qNaN: [%.10e]\n", reinterpret_cast<float&>(buf));
    buf = POSITIVE_SNAN;
    printf("sNaN: [%.10e]\n", reinterpret_cast<float&>(buf));
    buf = NEGATIVE_SNAN;
    printf("'Negative' sNaN: [%.10e]\n", reinterpret_cast<float&>(buf));

    printf("\n10 random floating-point numbers with exponent range [10-20]:\n");

    std::random_device rd;
    // printf("%u\n", rd());
    // std::mt19937 gen(2926344227);
    std::mt19937 gen(rd());

    std::uniform_int_distribution<uint32_t> sign_dist(0, 1);
    // std::uniform_int_distribution<uint32_t> exponent_dist(0, (1 << 8) - 1);
    std::uniform_int_distribution<uint32_t> exponent_dist(10 + 127, 20 + 127);
    std::uniform_int_distribution<uint32_t> mantisa_dist(0, (1 << 23) - 1);

    uint32_t positive_cnt = 0, negative_cnt = 0;

    for (int i = 0; i < 100000; ++i) {
        uint32_t tmp = (sign_dist(gen) << 31) | (exponent_dist(gen) << 23) | (mantisa_dist(gen) << 0);
        // print_float_in_binary(tmp);
        float f;
        static_assert(sizeof(uint32_t) == sizeof(float));
        memcpy(&f, &tmp, sizeof(uint32_t));
        // printf("%.10f (%.10e)\n", f, f);
        if (f > 0) positive_cnt++;
        else negative_cnt++;
    }

    printf("Positive: %u, Negative: %u\n", positive_cnt, negative_cnt);

    return 0;
}