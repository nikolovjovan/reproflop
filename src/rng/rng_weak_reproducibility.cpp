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

uint32_t thread_count = DEFAULT_THREAD_COUNT;
uint32_t array_size = DEFAULT_ARRAY_SIZE;
uint32_t seed = DEFAULT_SEED;
int exponent_min = DEFAULT_EXPONENT_MIN_VALUE;
int exponent_max = DEFAULT_EXPONENT_MAX_VALUE;
uint32_t repeat_count = DEFAULT_REPEAT_COUNT;
bool print_numbers = false;

float *arr;
uint32_t blk_size;
float *partial_sums;
float par_sum;
bool par_sum_valid;
pthread_barrier_t bar;
uint64_t time_par;

uint64_t get_posix_clock_time()
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (uint64_t) (ts.tv_sec * 1000000 + ts.tv_nsec / 1000);
    } else {
        return 0;
    }
}

void print_float_examples()
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
}

void print_float_in_binary(uint32_t f)
{
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

void print_float_in_binary(float f)
{
    uint32_t bits;
    static_assert(sizeof(float) == sizeof(uint32_t));
    memcpy(&bits, &f, sizeof(float));
    print_float_in_binary(bits);
}

void* parallel_sum(void *data)
{
    uint32_t id = *((uint32_t*) data);
    uint32_t start = id * blk_size;
    uint32_t end = start + blk_size > array_size ? array_size : start + blk_size;

    for (uint32_t repeat_counter = 0; repeat_counter <= repeat_count; ++repeat_counter) {
        if (id == 0 && repeat_counter == 0) {
            time_par = get_posix_clock_time();
        }

        partial_sums[id] = 0;

        for (uint32_t i = start; i < end; ++i) {
            partial_sums[id] += arr[i];
        }

        if (repeat_counter == 0) {
            printf("Partial sum %u. calculated: %.10f\n", id, partial_sums[id]);
        }

        // Wait on barrier to synchronize all threads to start parallel reduction

        pthread_barrier_wait(&bar);

        // Reduce partial sums

        uint32_t pow2_count = 1, step_count = 0;
        while (pow2_count < thread_count) {
            pow2_count <<= 1;
            step_count++;
        }
        pow2_count >>= 1;

        if (step_count > 0) {
            // First reduction step is not based on power of two reduction since thread_count may not be a power of two...
            if (id == 0 && repeat_counter == 0) {
                // TODO: Remove this once reduction works
                printf("Step 1: pow2_count = %u, required_count = %u\n", pow2_count, thread_count - pow2_count);
            }
            if (id < thread_count - pow2_count) {
                partial_sums[id] += partial_sums[id + pow2_count];
            }
            // Wait on barrier to synchronize all threads for next reduction step
            pthread_barrier_wait(&bar);

            // The rest of the steps are simple power of two reduction. There are now pow2_count partial sums to reduce...
            for (uint32_t i = 1; i < step_count; ++i) {
                if (id == 0 && repeat_counter == 0) {
                    // TODO: Remove this once reduction works
                    printf("Step %u: pow2_count = %u, required_count = %u\n", i + 1, pow2_count, pow2_count >> 1);
                }
                pow2_count >>= 1;
                if (id < pow2_count) {
                    partial_sums[id] += partial_sums[id + pow2_count];
                }
                // Wait on barrier to synchronize all threads for next reduction step
                pthread_barrier_wait(&bar);
            }
        }

        if (id == 0) {
            if (repeat_counter == 0) {
                time_par = get_posix_clock_time() - time_par;
                par_sum = partial_sums[0];
                par_sum_valid = true;
                printf("Parallel sum: %.10f (%.10e)\n", par_sum, par_sum);
            } else {
                // printf("Parallel sum: %.10f (%.10e)\n", partial_sums[0], partial_sums[0]);
                // Bitwise comparison is expected for sequential implementation.
                // TODO: Use 6-7 significant digit comparison for parallelized implementation!!!
                if (partial_sums[0] != par_sum) {
                    printf("Parallel sum not reproducible after %u runs!\n", (repeat_counter + 1));
                    par_sum_valid = false;
                    break;
                }
            }
        }

        // Wait on barrier to synchronize all threads for next repetition
        pthread_barrier_wait(&bar);

        if (!par_sum_valid) {
            printf("Thread %u exiting!\n", id);
            break;
        }
    }

    pthread_exit(NULL);
}

int main(int argc, char *argv[])
{
    int opt;
    while ((opt = getopt(argc, argv, "t:n:s:l:h:r:p")) != -1) {
        switch (opt) {
            case 't':
                opt = atoi(optarg);
                if (opt < 1) {
                    printf("Invalid thread count: %d! Using default value: %u.\n", opt, DEFAULT_THREAD_COUNT);
                } else {
                    thread_count = opt;
                }
                break;
            case 'n':
                opt = atoi(optarg);
                if (opt < 1) {
                    printf("Invalid array size: %d! Using default value: %u.\n", opt, DEFAULT_ARRAY_SIZE);
                } else {
                    array_size = opt;
                }
                break;
            case 's':
                opt = atoi(optarg);
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
                if (opt < EXPONENT_MIN_VALUE) {
                    printf("Invalid exponent min value: %d! Using default value: %d.\n", opt, DEFAULT_EXPONENT_MIN_VALUE);
                } else {
                    exponent_min = opt;
                }
                break;
            case 'h':
                opt = atoi(optarg);
                if (opt > EXPONENT_MAX_VALUE) {
                    printf("Invalid exponent max value: %d! Using default value: %d.\n", opt, DEFAULT_EXPONENT_MAX_VALUE);
                } else {
                    exponent_max = opt;
                }
                break;
            case 'r':
                opt = atoi(optarg);
                if (opt < 0) {
                    printf("Invalid repeat count: %d! Using default value: %u.\n", opt, DEFAULT_REPEAT_COUNT);
                } else {
                    repeat_count = opt;
                }
                break;
            case 'p':
                print_numbers = true;
                break;
            default:
                printf("Invalid option: '-%c'!\n", opt);
                exit(EXIT_FAILURE);
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

    arr = new float[array_size];

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

    uint64_t time_seq = get_posix_clock_time();

    for (uint32_t i = 0; i < array_size; ++i) {
        seq_sum += arr[i];
    }

    time_seq = get_posix_clock_time() - time_seq;

    printf("Sequential sum: %.10f (%.10e)\n", seq_sum, seq_sum);

    bool valid = true;
    for (uint32_t i = 0; i < repeat_count; ++i) {
        float sum = 0;
        for (uint32_t j = 0; j < array_size; ++j) {
            sum += arr[j];
        }
        // Bitwise comparison is expected for sequential implementation.
        // Use 6-7 significant digit comparison for parallelized implementation!!!
        if (sum != seq_sum) {
            printf("Sequential sum not reproducible after %u runs!\n", (i + 1));
            valid = false;
            break;
        }
    }

    // Initialize POSIX threads

    blk_size = (array_size + thread_count - 1) / thread_count;

    pthread_t *threads = new pthread_t[thread_count];
    pthread_attr_t attr;
    uint32_t *thread_ids = new uint32_t[thread_count];
    partial_sums = new float[thread_count];

    // Create thread attribute object (in case this code is used with compilers other than G++)

    int err;
    
    err = pthread_attr_init(&attr);
    if (err) {
        fprintf(stderr, "Error - pthread_attr_init() return code: %d\n", err);
        exit(EXIT_FAILURE);
    }

    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    err = pthread_barrier_init(&bar, NULL, thread_count);
    if (err) {
        fprintf(stderr, "Error - pthread_barrier_init() return code: %d\n", err);
        exit(EXIT_FAILURE);
    }

    for (uint32_t i = 0; i < thread_count; ++i) {
        thread_ids[i] = i;
        err = pthread_create(&(threads[i]), &attr, parallel_sum, &(thread_ids[i]));
        if (err) {
            fprintf(stderr, "Error - pthread_create() return code: %d\n", err);
            exit(EXIT_FAILURE);
        }
    }

    // Wait for other threads to complete

    for (uint32_t i = 0; i < thread_count; ++i) {
        err = pthread_join(threads[i], NULL);
        if (err) {
            fprintf(stderr, "Error - pthread_join() return code: %d\n", err);
            exit(EXIT_FAILURE);
        }
    }

    printf("Sequential execution time: %llu [us] (%.10f [ms])\n", time_seq, ((float) time_seq) / 1000.0);
    printf("Parallel execution time: %llu [us] (%.10f [ms])\n", time_par, ((float) time_par) / 1000.0);
    printf("Speedup: %.10f\n", ((float) time_seq) / ((float) time_par));

    // Deallocate dynamic memory

    err = pthread_barrier_destroy(&bar);
    if (err) {
        fprintf(stderr, "Error - pthread_barrier_destroy() return code: %d\n", err);
        exit(EXIT_FAILURE);
    }

    err = pthread_attr_destroy(&attr);
    if (err) {
        fprintf(stderr, "Error - pthread_attr_destroy() return code: %d\n", err);
        exit(EXIT_FAILURE);
    }

    free(partial_sums);
    free(thread_ids);
    free(threads);
    free(arr);

    exit(EXIT_SUCCESS);
}