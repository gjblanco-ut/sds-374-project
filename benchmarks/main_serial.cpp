#include <chrono>
#include <algorithm>

#include "../openmp/linear_algebra.h"

#include <iostream>
using namespace std; 

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

vfloat tmp;


void bench_hadamard(int times, int sz) {
    vfloat x(sz);
    vfloat y(sz);
    auto t1 = high_resolution_clock::now();
    for(int i = 0; i < times; i++) {
        tmp = hadamard(x, y);
    }
    auto t2 = high_resolution_clock::now();
    cout << times << "x HADAMARD  (" << sz << ") * (" << sz << ") :: " << duration_cast<milliseconds>(t2 - t1).count() / 1000. << " (s)." << endl;

}

void bench_prod(int times, int rsz, int csz) {
    vfloat x(csz);
    Matrix M(rsz, vfloat(csz));
    auto t1 = high_resolution_clock::now();
    for(int i = 0; i < times; i++) {
        tmp = prod(M, x);
    }
    auto t2 = high_resolution_clock::now();
    cout << times << "x MATRIX x VECTOR  (" << rsz << " x " << csz << ") * (" << csz << ") :: " << duration_cast<milliseconds>(t2 - t1).count() / 1000. << " (s)." << endl;

}

Matrix TMP;
void bench_transpose(int times, int rsz, int csz) {
    Matrix M(rsz, vfloat(csz));
    auto t1 = high_resolution_clock::now();
    for(int i = 0; i < times; i++) {
        TMP = transpose(i%2 == 0? M : TMP);
    }
    auto t2 = high_resolution_clock::now();
    cout << times << "x MATRIX transpose  (" << rsz << " x " << csz << ")" << ") :: " << duration_cast<milliseconds>(t2 - t1).count() / 1000. << " (s)." << endl;

}


void bench_sigmoid(int times, int sz) {
    vfloat v(sz, 4.);
    auto t1 = high_resolution_clock::now();
    for(int i = 0; i < times; i++) {
        tmp = sigmoid(v);
    }
    auto t2 = high_resolution_clock::now();
    cout << times << "x VECTOR Sigmoid  (" << sz << ") :: " << duration_cast<milliseconds>(t2 - t1).count() / 1000. << " (s)." << endl;

}

int main() {
    

    cout << "Benchmark :: Duration(s)" << endl;

    vfloat x, y;

    // ---- 1000000x HADAMARD  (1000) * (1000) -------
    
    // ---- 1000x HADAMARD (1e6) * (1e6) -------

    bench_hadamard(1000000, (int)1e3);
    bench_hadamard(1000, (int)1e6);
    bench_hadamard(1, (int)1e9);

    bench_prod((int)1e5, 100, 100);
    bench_prod((int)1e3, 1000, 1000);
    bench_prod(10, 10000, 10000);

    bench_transpose((int)1e5, 100, 100);
    bench_transpose((int)1e3, 1000, 1000);
    bench_transpose(10, 10000, 10000);

    bench_sigmoid(1e5, 1000);
    bench_sigmoid(1e3, 100000);
    bench_sigmoid(10, 100000000);
}