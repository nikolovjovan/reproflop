#include <iostream>
#include <iomanip>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#include <chrono>
#include <pthread.h>
#include <thread>

using namespace std;

// Floating-point specification constants

constexpr int EXPONENT_MIN_VALUE = -126;
constexpr int EXPONENT_MAX_VALUE = 127;
constexpr int EXPONENT_BIAS = 127;

// Default values

constexpr int DEFAULT_THREAD_COUNT = 8;
constexpr int DEFAULT_ELEMENT_COUNT = 1000;
constexpr uint32_t DEFAULT_SEED = 1549813198;
constexpr int DEFAULT_EXPONENT_MIN_VALUE = -10;
constexpr int DEFAULT_EXPONENT_MAX_VALUE = 10;
constexpr int DEFAULT_REPEAT_COUNT = 100;

// Program parameters

int thread_count    = DEFAULT_THREAD_COUNT;
int element_count   = DEFAULT_ELEMENT_COUNT;
uint32_t seed       = DEFAULT_SEED;
int exponent_min    = DEFAULT_EXPONENT_MIN_VALUE;
int exponent_max    = DEFAULT_EXPONENT_MAX_VALUE;
int repeat_count    = DEFAULT_REPEAT_COUNT;
bool print_elements = false;

// Shared variables

vector<float> *elements;
default_random_engine *shuffle_engine;
int *start_indices;
vector<int> *reduction_map;
float *partial_sums;
pthread_barrier_t barrier;
bool result_valid;

// Results

float sum_sequential, sum_parallel;
uint64_t time_sequential, time_parallel;

void print_usage(char program_name[])
{
    cout << "Usage: " << program_name << " <options>" << endl;
    cout << "  -t <num>: Uses <num> threads for parallel execution (default value: 8)" << endl;
    cout << "  -n <num>: Generates <num> random floating-point numbers for analysis (default value: 1000)" << endl;
    cout << "  -s <num>: Generates the numbers using <num> as seed (default value: 1549813198)" << endl;
    cout << "  -l <num>: Uses <num> as exponent minimum value (default value: -10)" << endl;
    cout << "  -h <num>: Uses <num> as exponent maximum value (inclusive) (default value: 10)" << endl;
    cout << "  -r <num>: Repeats the execution of both implementations <num> times for reproducibility study (default value: 100)" << endl;
    cout << "  -p: Print generated numbers before analysis" << endl;
    cout << "  -?: Print this message" << endl;
}

void parse_parameters(int argc, char *argv[])
{
    for (int i = 1; i < argc; ++i) {
        int arglen = strlen(argv[i]);
        if (arglen < 2 || argv[i][0] != '-') {
            cout << "Invalid argument: \"" << argv[i] << '"' << endl;
            print_usage(argv[0]);
            exit(EXIT_FAILURE);
        }
        int n;
        switch (argv[i][1]) {
            case 't':
                if (i + 1 == argc) {
                    cout << "Thread count not specified!" << endl;
                    print_usage(argv[0]);
                    exit(EXIT_FAILURE);
                }
                n = atoi(argv[++i]);
                if (n < 1) {
                    cout << "Invalid thread count: " << n << "! Using default value: " << DEFAULT_THREAD_COUNT << endl;
                } else {
                    thread_count = n;
                }
                break;
            case 'n':
                if (i + 1 == argc) {
                    cout << "Element count not specified!" << endl;
                    print_usage(argv[0]);
                    exit(EXIT_FAILURE);
                }
                n = atoi(argv[++i]);
                if (n < 1) {
                    cout << "Invalid element count: " << n << "! Using default value: " << DEFAULT_ELEMENT_COUNT << endl;
                } else {
                    element_count = n;
                }
                break;
            case 's':
                if (i + 1 == argc) {
                    cout << "Seed not specified!" << endl;
                    print_usage(argv[0]);
                    exit(EXIT_FAILURE);
                }
                n = atoi(argv[++i]);
                if (n == 0) {
                    random_device rd;
                    seed = rd();
                    cout << "Special seed value: 0. Generated random seed: " << seed << endl;
                } else {
                    seed = n;
                }
                break;
            case 'l':
                if (i + 1 == argc) {
                    cout << "Exponent minimum value not specified!" << endl;
                    print_usage(argv[0]);
                    exit(EXIT_FAILURE);
                }
                n = atoi(argv[++i]);
                if (n < EXPONENT_MIN_VALUE) {
                    cout << "Invalid exponent minimum value: " << n << "! Using default value: " << DEFAULT_EXPONENT_MIN_VALUE << endl;
                } else {
                    exponent_min = n;
                }
                break;
            case 'h':
                if (i + 1 == argc) {
                    cout << "Exponent maximum value not specified!" << endl;
                    print_usage(argv[0]);
                    exit(EXIT_FAILURE);
                }
                n = atoi(argv[++i]);
                if (n > EXPONENT_MAX_VALUE) {
                    cout << "Invalid exponent maximum value: " << n << "! Using default value: " << DEFAULT_EXPONENT_MAX_VALUE << endl;
                } else {
                    exponent_max = n;
                }
                break;
            case 'r':
                if (i + 1 == argc) {
                    cout << "Repeat count not specified!" << endl;
                    print_usage(argv[0]);
                    exit(EXIT_FAILURE);
                }
                n = atoi(argv[++i]);
                if (n < 0) {
                    cout << "Invalid repeat count: " << n << "! Using default value: " << DEFAULT_REPEAT_COUNT << endl;
                } else {
                    repeat_count = n;
                }
                break;
            case 'p':
                print_elements = true;
                break;
            case '?':
                print_usage(argv[0]);
                exit(EXIT_SUCCESS);
            default:
                cout << "Invalid option: \"" << argv[i] << "\"!" << endl;
                exit(EXIT_FAILURE);
        }
    }
    if (thread_count > element_count) {
        cout << "There are more threads available than there are elements to process. Reducing thread count to element count." << endl;
        thread_count = element_count;
    }
}

void print_parameters()
{
    cout << "Parameters:" << endl;
    cout << "  Thread count:             " << thread_count << endl;
    cout << "  Element count:            " << element_count << endl;
    cout << "  Seed:                     " << seed << endl;
    cout << "  Exponent minimum value:   " << exponent_min << endl;
    cout << "  Exponent maximum value:   " << exponent_max << endl;
    cout << "  Repeat count:             " << repeat_count << endl;
    cout << "  Print elements:           " << (print_elements ? "YES" : "NO") << endl;
}

void generate_elements()
{
    mt19937 gen(seed);
    uniform_int_distribution<uint32_t> sign_dist(0, 1);
    // uniform_int_distribution<uint32_t> exponent_dist(0, (1 << 8) - 1);
    uniform_int_distribution<uint32_t> exponent_dist(exponent_min + EXPONENT_BIAS, exponent_max + EXPONENT_BIAS);
    uniform_int_distribution<uint32_t> mantisa_dist(0, (1 << 23) - 1);

    elements = new vector<float>(element_count);

    int positive_count = 0, negative_count = 0;
    for (int i = 0; i < element_count; ++i) {
        uint32_t bits = (sign_dist(gen) << 31) | (exponent_dist(gen) << 23) | (mantisa_dist(gen) << 0);
        static_assert(sizeof(uint32_t) == sizeof(float));
        float number;
        memcpy(&number, &bits, sizeof(uint32_t));
        (*elements)[i] = number;
        if (print_elements) {
            cout << i + 1 << ". element: " << fixed << setprecision(10) << number << " (" << scientific << setprecision(10) << number << ')' << endl;
            if (number > 0) {
                positive_count++;
            } else {
                negative_count++;
            }
        }
    }

    if (print_elements) {
        cout << "Number of positive elements: " << positive_count << endl;
        cout << "Number of negative elements: " << negative_count << endl;
    }

    cout << "Successfully generated " << element_count << " random floating-point numbers." << endl;
}

int compare(float f1, float f2)
{
    // TODO: Implement floating-point comparison for 6-7 significant digits...
    if (f1 < f2) return -1;
    else if (f1 > f2) return 1;
    return 0;
}

void run_sequential()
{
    chrono::steady_clock::time_point start;
    float sum;
    for (int run_idx = 0; run_idx <= repeat_count; ++run_idx) {
        if (run_idx == 0) {
            start = chrono::steady_clock::now();
        }
        sum = 0;
        for (int i = 0; i < element_count; ++i) {
            // NOTE: This operation results in non-reproducible results. The reason is that
            //       element order is shuffled after every repetition hence rounding errors
            //       and other inherent errors with floating-point arithmetic occur.
            sum += (*elements)[i];
        }
        if (run_idx == 0) {
            sum_sequential = sum;
            time_sequential = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();
            cout << "Sequential sum: " << fixed << setprecision(10) << sum_sequential << " (" << scientific << setprecision(10) << sum_sequential << ')' << endl;
        } else if (compare(sum, sum_sequential)) {
            cout << "Sequential sum not reproducible after " << run_idx << " runs!" << endl;
            break;
        }
        if (run_idx < repeat_count) {
            // Shuffle element vector to incur variability
            shuffle(elements->begin(), elements->end(), *shuffle_engine);
        }
    }
}

void generate_start_indices()
{
    int min = 1;
    int max = element_count - thread_count * min;
    if (max == 0) {
        for (int i = 0; i < thread_count; ++i) {
            start_indices[i] = 1;
        }
    } else {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<int> size_dist(0, max);
        start_indices[0] = 0;
        for (int i = 1; i < thread_count; ++i) {
            start_indices[i] = size_dist(gen);
        }
        sort(start_indices, start_indices + thread_count);
        for (int i = 1; i < thread_count; ++i) {
            start_indices[i - 1] = start_indices[i] - start_indices[i - 1] + min;
        }
        start_indices[thread_count - 1] = max - start_indices[thread_count - 1] + min;
        int size = 0;
        for (int i = 0; i < thread_count; ++i) {
            int tmp = start_indices[i];
            start_indices[i] = size;
            size += tmp;
        }
    }
}

void* kernel_sum(void *data)
{
    int id = *((int*) data);
    chrono::steady_clock::time_point start;
    for (int repeat_counter = 0; repeat_counter <= repeat_count; ++repeat_counter) {
        if (id == 0) {
            // Generate start indices
            generate_start_indices();
            if (repeat_counter == 0) {
                start = chrono::steady_clock::now();
            }
        }

        // Synchronize on barrier to get start index
        pthread_barrier_wait(&barrier);

        partial_sums[id] = 0;
        int end = id < thread_count - 1 ? start_indices[id + 1] : element_count;
        for (int i = start_indices[id]; i < end; ++i) {
            partial_sums[id] += (*elements)[i];
        }

        // Wait on barrier to synchronize all threads to start parallel reduction
        pthread_barrier_wait(&barrier);

        // Reduce partial sums
        int pow2_count = 1, step_count = 0;
        while (pow2_count < thread_count) {
            pow2_count <<= 1;
            step_count++;
        }
        pow2_count >>= 1;
        if (step_count > 0) {
            // First reduction step is not based on power of two reduction since thread_count may not be a power of two...
            if (id < thread_count - pow2_count) {
                partial_sums[(*reduction_map)[id]] += partial_sums[(*reduction_map)[id + pow2_count]];
            }
            // Wait on barrier to synchronize all threads for next reduction step
            pthread_barrier_wait(&barrier);
            // The rest of the steps are simple power of two reduction. There are now pow2_count partial sums to reduce...
            for (int i = 1; i < step_count; ++i) {
                pow2_count >>= 1;
                if (id < pow2_count) {
                    partial_sums[(*reduction_map)[id]] += partial_sums[(*reduction_map)[id + pow2_count]];
                }
                // Wait on barrier to synchronize all threads for next reduction step
                pthread_barrier_wait(&barrier);
            }
        }

        // Check results
        if (id == 0) {
            if (repeat_counter == 0) {
                time_parallel = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();
                sum_parallel = partial_sums[0];
                result_valid = true;
                cout << "Parallel sum: " << fixed << setprecision(10) << sum_parallel << " (" << scientific << setprecision(10) << sum_parallel << ')' << endl;
            } else if (compare(partial_sums[0], sum_parallel)) {
                cout << "Parallel sum not reproducible after " << repeat_counter << " runs!" << endl;
                result_valid = false;
            }
            if (repeat_counter < repeat_count && result_valid) {
                // Shuffle element vector to incur variability
                shuffle(elements->begin(), elements->end(), *shuffle_engine);
                // Shuffle reduction map to incur additional variability
                shuffle(reduction_map->begin(), reduction_map->end(), *shuffle_engine);
            }
        }

        // Wait on barrier to synchronize all threads for next repetition
        pthread_barrier_wait(&barrier);
        if (!result_valid) break;
    }
    pthread_exit(NULL);
}

void run_parallel()
{
    // Get number of available processors
    const int processor_count = thread::hardware_concurrency();
    // cout << "Processor count: " << processor_count << endl;

    // Initialize POSIX threads
    pthread_t *threads = new pthread_t[thread_count];
    pthread_attr_t attr;
    int *thread_ids = new int[thread_count];

    start_indices = new int[thread_count];
    reduction_map = new vector<int>(thread_count);
    partial_sums = new float[thread_count];

    int err;

    // Create thread attribute object (in case this code is used with compilers other than G++)
    err = pthread_attr_init(&attr);
    if (err) {
        fprintf(stderr, "Error - pthread_attr_init() return code: %d\n", err);
        exit(EXIT_FAILURE);
    }
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    // Create thread barrier
    err = pthread_barrier_init(&barrier, NULL, thread_count);
    if (err) {
        fprintf(stderr, "Error - pthread_barrier_init() return code: %d\n", err);
        exit(EXIT_FAILURE);
    }

    // Create threads with affinity
    cpu_set_t cpus;
    for (int i = 0; i < thread_count; ++i) {
        thread_ids[i] = i; // Generate thread ids for easy work sharing
        (*reduction_map)[i] = i; // Generate initial reduction map for each thread id (same as thread id)
        CPU_ZERO(&cpus);
        CPU_SET(i % processor_count, &cpus);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
        err = pthread_create(&(threads[i]), &attr, kernel_sum, &(thread_ids[i]));
        if (err) {
            fprintf(stderr, "Error - pthread_create() return code: %d\n", err);
            exit(EXIT_FAILURE);
        }
    }

    // Wait for other threads to complete
    for (int i = 0; i < thread_count; ++i) {
        err = pthread_join(threads[i], NULL);
        if (err) {
            fprintf(stderr, "Error - pthread_join() return code: %d\n", err);
            exit(EXIT_FAILURE);
        }
    }

    // Cleanup
    err = pthread_barrier_destroy(&barrier);
    if (err) {
        fprintf(stderr, "Error - pthread_barrier_destroy() return code: %d\n", err);
        exit(EXIT_FAILURE);
    }

    err = pthread_attr_destroy(&attr);
    if (err) {
        fprintf(stderr, "Error - pthread_attr_destroy() return code: %d\n", err);
        exit(EXIT_FAILURE);
    }

    delete partial_sums;
    delete reduction_map;
    delete start_indices;
    delete thread_ids;
    delete threads;
}

void cleanup()
{
    delete shuffle_engine;
    delete elements;
}

int main(int argc, char *argv[])
{
    parse_parameters(argc, argv);
    print_parameters();

    generate_elements();

    shuffle_engine = new default_random_engine(seed);

    run_sequential();
    run_parallel();

    cout << "Sequential execution time: " << time_sequential << " [us] (" << fixed << setprecision(10) << (float) time_sequential / 1000.0 << " [ms])" << endl;
    cout << "Parallel execution time: " << time_parallel << " [us] (" << fixed << setprecision(10) << (float) time_parallel / 1000.0 << " [ms])" << endl;
    cout << "Speedup: " << fixed << setprecision(10) << ((float) time_sequential) / ((float) time_parallel) << endl;

    cleanup();

    return EXIT_SUCCESS;
}