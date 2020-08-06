#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <unistd.h>
#include <pthread.h>

using namespace std;

#define EXPONENT_BIAS               (127)
#define EXPONENT_MIN_VALUE          (-126)
#define EXPONENT_MAX_VALUE          (127)

#define DEFAULT_THREAD_COUNT        (8)
#define DEFAULT_ARRAY_SIZE          (1000)
#define DEFAULT_SEED                (1549813198)
#define DEFAULT_EXPONENT_MIN_VALUE  (-10)
#define DEFAULT_EXPONENT_MAX_VALUE  (10)
#define DEFAULT_REPEAT_COUNT        (100)

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

void print_float_examples() {
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
}

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
    uint32_t thread_count = DEFAULT_THREAD_COUNT;
    uint32_t array_size = DEFAULT_ARRAY_SIZE;
    uint32_t seed = DEFAULT_SEED;
    int exponent_min = DEFAULT_EXPONENT_MIN_VALUE;
    int exponent_max = DEFAULT_EXPONENT_MAX_VALUE;
    uint32_t repeat_count = DEFAULT_REPEAT_COUNT;
    bool print_numbers = false;

    int opt;
    while ((opt = getopt(argc, argv, "t:n:s:l:h:r:p")) != -1) {
        switch (opt) {
            case 't':
                opt = atoi(optarg);
                // printf("Thread count: '%s' (%d)\n", optarg, opt);
                if (opt < 1) {
                    printf("Invalid thread count: %d! Using default value: %u.\n", opt, DEFAULT_THREAD_COUNT);
                } else {
                    thread_count = opt;
                }
                break;
            case 'n':
                opt = atoi(optarg);
                // printf("Array size: '%s' (%d)\n", optarg, opt);
                if (opt < 1) {
                    printf("Invalid array size: %d! Using default value: %u.\n", opt, DEFAULT_ARRAY_SIZE);
                } else {
                    array_size = opt;
                }
                break;
            case 's':
                opt = atoi(optarg);
                // printf("Seed: '%s' (%u)\n", optarg, opt);
                if (opt == 0) {
                    std::random_device rd;
                    seed = rd();
                    printf("Special seed value 0. Generated random seed: %u.\n", seed);
                } else {
                    seed = opt;
                }
                break;
            case 'l':
                opt = atoi(optarg);
                // printf("Exponent min value: '%s' (%u)\n", optarg, opt);
                if (opt < EXPONENT_MIN_VALUE) {
                    printf("Invalid exponent min value: %d! Using default value: %d.\n", opt, DEFAULT_EXPONENT_MIN_VALUE);
                } else {
                    exponent_min = opt;
                }
                break;
            case 'h':
                opt = atoi(optarg);
                // printf("Exponent max value: '%s' (%u)\n", optarg, opt);
                if (opt > EXPONENT_MAX_VALUE) {
                    printf("Invalid exponent max value: %d! Using default value: %d.\n", opt, DEFAULT_EXPONENT_MAX_VALUE);
                } else {
                    exponent_max = opt;
                }
                break;
            case 'r':
                opt = atoi(optarg);
                // printf("Repeat count: '%s' (%d)\n", optarg, opt);
                if (opt < 1) {
                    printf("Invalid repeat count: %d! Using default value: %u.\n", opt, DEFAULT_REPEAT_COUNT);
                } else {
                    repeat_count = opt;
                }
                break;
            case 'p':
                // printf("Print generated numbers: enabled.\n");
                print_numbers = true;
                break;
            default:
                printf("Invalid option: '-%c'!\n", opt);
                exit(1);
        }
    }

    // Print parameters

    printf("Thread count: %u.\n", thread_count);
    printf("Array size: %u.\n", array_size);
    printf("Seed: %u.\n", seed);
    printf("Exponent min value: %d.\n", exponent_min);
    printf("Exponent max value: %d.\n", exponent_max);
    printf("Repeat count: %d.\n", repeat_count);
    printf("Print numbers: %s.\n", (print_numbers ? "enabled" : "disabled"));

    // Generate random number array

    mt19937 gen(seed);

    uniform_int_distribution<uint32_t> sign_dist(0, 1);
    // std::uniform_int_distribution<uint32_t> exponent_dist(0, (1 << 8) - 1);
    uniform_int_distribution<uint32_t> exponent_dist(exponent_min + EXPONENT_BIAS, exponent_max + EXPONENT_BIAS);
    std::uniform_int_distribution<uint32_t> mantisa_dist(0, (1 << 23) - 1);

    uint32_t positive_cnt = 0, negative_cnt = 0;

    float *arr = new float[array_size];

    for (uint32_t i = 0; i < array_size; ++i) {
        uint32_t tmp = (sign_dist(gen) << 31) | (exponent_dist(gen) << 23) | (mantisa_dist(gen) << 0);
        // print_float_in_binary(tmp);
        static_assert(sizeof(uint32_t) == sizeof(float));
        memcpy(&(arr[i]), &tmp, sizeof(uint32_t));
        if (print_numbers) {
            printf("arr[%u] = %.10f (%.10e)\n", i, arr[i], arr[i]);
        }
        if (arr[i] > 0) positive_cnt++;
        else negative_cnt++;
    }

    printf("Positive count: %u, Negative count: %u\n", positive_cnt, negative_cnt);

    // Calculate sequential sum of the array

    float seq_sum = 0;
    bool valid = true;

    uint32_t i;
    for (i = 0; i < repeat_count; ++i) {
        float sum = 0;
        for (uint32_t j = 0; j < array_size; ++j) {
            sum += arr[j];
        }
        if (i == 0) {
            seq_sum = sum;
        } else {
            // Bitwise comparison is expected for sequential implementation, use 6-7 significant digit comparison for parallelized implementation!!!
            if (sum != seq_sum) {
                printf("Result of %u. sequential run is not the same as base result!\n", (i + 1));
                valid = false;
                break;
            }
        }
    }

    if (valid) {
        printf("Sequential sum: %.10f (%.10e)\n", seq_sum, seq_sum);
    } else {
        printf("Sequential sum not reproducible after %u runs!\n", (i + 1));
    }

    // Deallocate dynamic memory

    free(arr);

    return 0;
}