#include <iomanip>
#include <iostream>

#include "LongAccumulator.h"

using namespace std;

int main()
{
    LongAccumulator::InitializeOpenCL();

    // LongAccumulator a(8388607.5);

    // cout << fixed << setprecision(10);

    // cout << "a = " << a() << " (" << a << ")\n";

    // a += 1.255;

    // cout << "a = " << a() << " (" << a << ")\n";

    LongAccumulator::CleanupOpenCL();

    return 0;
}