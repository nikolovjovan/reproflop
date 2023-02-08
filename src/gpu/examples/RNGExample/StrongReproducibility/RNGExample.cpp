#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <pthread.h>
#include <random>
#include <thread>
#include <vector>

#include "LongAccumulator.h"
#include "LongAccumulatorCPU.h"

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

int thread_count = DEFAULT_THREAD_COUNT;
int element_count = DEFAULT_ELEMENT_COUNT;
uint32_t seed = DEFAULT_SEED;
int exponent_min = DEFAULT_EXPONENT_MIN_VALUE;
int exponent_max = DEFAULT_EXPONENT_MAX_VALUE;
int repeat_count = DEFAULT_REPEAT_COUNT;
bool print_elements = false;
bool perf_test = false;

// Shared variables

vector<float> *elements;
int *start_indices;
vector<int> *reduction_map;
float *partial_sums;
pthread_barrier_t barrier;
bool result_valid;

// Results

float sum_gpu_reproducible;
uint64_t time_gpu_reproducible, time_gpu_reproducible_setup;

typedef struct {
    // Input
    //
    int tid;

    // Output
    //
    float result;
    uint64_t time;
} kernel_params;

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
    cout << "  -!: Split elements evenly between threads for performance testing" << endl;
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
                cout << "Invalid exponent minimum value: " << n
                     << "! Using default value: " << DEFAULT_EXPONENT_MIN_VALUE << endl;
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
                cout << "Invalid exponent maximum value: " << n
                     << "! Using default value: " << DEFAULT_EXPONENT_MAX_VALUE << endl;
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
        case '!':
            perf_test = true;
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
            cout << i + 1 << ". element: " << fixed << setprecision(10) << number << " (" << scientific
                 << setprecision(10) << number << ')' << endl;
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

    if (!perf_test) {
        cout << "Successfully generated " << element_count << " random floating-point numbers." << endl;
    }
}

void run_gpu_reproducible()
{
    chrono::steady_clock::time_point start;
    float sum;

    time_gpu_reproducible_setup = 0;

    start = chrono::steady_clock::now();
    LongAccumulator::InitializeOpenCL();
    time_gpu_reproducible_setup =
        chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();

    uint64_t run_time[repeat_count + 1];

    for (int run_idx = 0; run_idx <= repeat_count; ++run_idx) {
        start = chrono::steady_clock::now();
        sum = LongAccumulator::Sum(element_count, elements->data());
        time_gpu_reproducible =
            chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();

        if (perf_test) {
            run_time[run_idx] = time_gpu_reproducible;
        }

        if (run_idx == 0) {
            sum_gpu_reproducible = sum;

            if (!perf_test) {
                cout << "GPU reproducible sum: " << fixed << setprecision(10) << sum_gpu_reproducible
                    << " (" << scientific << setprecision(10) << sum_gpu_reproducible << ')' << endl;
            }
        } else if (sum != sum_gpu_reproducible) {
            cout << "GPU reproducible sum not reproducible after " << run_idx << " runs!" << endl;
            cout << "New (wrong) sum: " << fixed << setprecision(10) << sum
                 << " (" << scientific << setprecision(10) << sum << ')' << endl;
            break;
        }
    }

    start = chrono::steady_clock::now();
    LongAccumulator::CleanupOpenCL();
    time_gpu_reproducible_setup +=
        chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();

    if (perf_test) {
        for (int run_idx = 0; run_idx <= repeat_count; ++run_idx) {
            cout << fixed << setprecision(10) << (float) time_gpu_reproducible_setup / 1000.0 << (run_idx == repeat_count ? '\n' : '\t'); // ms
        }
        for (int run_idx = 0; run_idx <= repeat_count; ++run_idx) {
            cout << fixed << setprecision(10) << (float) run_time[run_idx] / 1000.0 << (run_idx == repeat_count ? '\n' : '\t'); // ms
        }
    }
}

int main(int argc, char *argv[])
{
    parse_parameters(argc, argv);

    // Retry 3 times.
    //
    repeat_count = 2;

    // Force performance testing to disable unnecessary logging and optimize parallel algorithm.
    //
    perf_test = true;

    // Used for GPU and CPU result comparison.
    //
    LongAccumulatorCPU acc;

    chrono::steady_clock::time_point start;
    uint64_t time_first_setup = 0;

    start = chrono::steady_clock::now();
    LongAccumulator::InitializeOpenCL();
    time_first_setup =
        chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();

    start = chrono::steady_clock::now();
    LongAccumulator::CleanupOpenCL();
    time_first_setup +=
        chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();

    cout << "OCL first (dummy) setup time : " << fixed << setprecision(10) << (float) time_first_setup / 1000.0 << endl << endl; // ms

    cout << "The following numbers represent setup and run times for each  run:" << endl;
    cout << "The first line contains setup times for each run (same for all runs as the setup is shared for each run)." << endl;
    cout << "The second line contains run times for each run." << endl;

    cout << "unit: [ms]\n\n";

    for (int step = 0; step < 2; ++step)
    {
        if (step == 0) {
            element_count = 100;
            exponent_min = DEFAULT_EXPONENT_MIN_VALUE;
            exponent_max = DEFAULT_EXPONENT_MAX_VALUE;
        } else {
            element_count = 10000000;
            exponent_min = DEFAULT_EXPONENT_MIN_VALUE;
            exponent_max = DEFAULT_EXPONENT_MAX_VALUE;
        }

        for (int i = 0; i < 7; ++i) {
            if (step == 0) {
                cout << "n = " << element_count << "\n\n";
            } else {
                cout << "maxabs(e) = " << exponent_max << "\n\n";
            }

            generate_elements();

            for (int i = 0; i < element_count; ++i) {
                acc += (*elements)[i];
            }

            run_gpu_reproducible();
            cout << '\n';

            float sum_cpu_reproducible = acc();

            if (sum_gpu_reproducible != sum_cpu_reproducible) {
                cout << "CPU and GPU reproducible sums do not match!\n";
                cout << "CPU reproducible sum: " << fixed << setprecision(10) << sum_cpu_reproducible
                    << " (" << scientific << setprecision(10) << sum_cpu_reproducible << ')' << endl;
                cout << "GPU reproducible sum: " << fixed << setprecision(10) << sum_gpu_reproducible
                    << " (" << scientific << setprecision(10) << sum_gpu_reproducible << ')' << endl;
            }

            acc = 0;

            delete elements;
            
            if (step == 0) {
                element_count *= 10;
            } else {
                exponent_min -= 15;
                exponent_max += 15;
            }
        }
    }

    return EXIT_SUCCESS;
}