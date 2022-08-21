#include <iomanip>
#include <iostream>

#include "LongAccumulator.h"

using namespace std;

int N = 100;

int main(int argc, char *argv[])
{
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    float *arr = new float[N];

    for (int i = 0; i < N; ++i) {
        arr[i] = (float) (i + 1);
        // arr[i] = 1.0;
    }

    LongAccumulator::InitializeOpenCL();

    float sum = LongAccumulator::Sum(N, arr);

    LongAccumulator::CleanupOpenCL();

    float expected_sum = (float) N * ((float) N + 1.0) / 2.0;
    // float expected_sum = N;
    bool is_pass = sum == expected_sum;

    cout << "Sum of first " << N << " numbers: " << sum;

    if (is_pass) {
        cout << " is correct!\n";
        cout << "TestPassed; ALL OK!\n";
    } else {
        cout << " is wrong! Expected sum: " << expected_sum << ".\n";
        cout << "TestFailed!\n";
    }

    delete[] arr;

    return 0;
}