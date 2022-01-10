#include <iomanip>
#include <iostream>

#include "LongAccumulator.h"

using namespace std;

int main()
{
    LongAccumulator a(8388607.5);
    float fa = 8388607.5;

    bool is_pass = true;

    is_pass = fa == a();

    cout << fixed << setprecision(10);

    cout << "a = " << a() << " (" << a << ")\n";

    a += 1.255;
    fa += 1.255;

    cout << "a = " << a() << " (" << a << ")\n";

    is_pass = is_pass && (fa == a());

    if (is_pass) {
        cout << "TestPassed; ALL OK!\n";
    } else {
        cout << "TestFailed!\n";
    }

    return 0;
}