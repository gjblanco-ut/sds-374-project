#include "linear_algebra.h"
#include <cmath>
#include <cassert>

double sigmoid(double x) {
    return 1. / (1 + exp(-x));
}

double dsigmoid(double x) {
    double exp_x = exp(-x);
    return 1./ (1 + exp_x) * (1 - 1. / (1 + exp_x));
}


vfloat sigmoid(const vfloat& v) {
    vfloat ret(v.size());
    #pragma omp parallel for
    for(int i = 0; i < (int)v.size(); i++)
        ret[i] = (float)sigmoid(v[i]);
    return ret;
}

vfloat dsigmoid(const vfloat& v) {
    vfloat ret(v.size());
    #pragma omp parallel for
    for(int i = 0; i < (int)v.size(); i++)
        ret[i] = (float)dsigmoid(v[i]);
    return ret;
}


vfloat hadamard(const vfloat& u, const vfloat& v) {
    vfloat ret(u.size());
    assert(u.size() == v.size());
    #pragma omp parallel for
    for(int i = 0; i < (int)ret.size(); i++) 
        ret[i] = u[i] * v[i];
    return ret;
}

double inner_prod(const vfloat& u, const vfloat& v) {
    double ret = 0;
    assert(u.size() == v.size());
    #pragma omp parallel for reduction(+: ret)
    for(int i = 0; i < (int)u.size(); i++)
        ret += u[i] * v[i];
    return ret;
}

Matrix transpose(const Matrix& m) {
    if(m.size() == 0) return Matrix();
    int n = m[0].size();
    Matrix ret(n, vfloat(m.size()));
    #pragma omp parallel for
    for(int i = 0; i < n; i++)
        for(int j = 0; j < (int)m.size(); j++)
            ret[i][j] = m[j][i];
    return ret;
}



vfloat prod(const Matrix& m, const vfloat& v) {
    assert(m.size() > 0);
    assert(m[0].size() == v.size());
    assert((int)v.size() > 0);
    vfloat ret(m.size());
    #pragma omp parallel for
    for(int i = 0; i < (int)m.size(); i++) { // i: row
        ret[i] = 0;
        assert(m[i].size() == v.size());
        for(int j = 0; j < (int)m[i].size(); j++) {
            ret[i] += m[i][j] * v[j];
        }
    }
    return ret;
}

vfloat sum(const vfloat& v, const vfloat& u) {
    assert(u.size() == v.size());
    vfloat ret(v.size());
    #pragma omp parallel for
    for(int i = 0; i < (int)v.size(); i++)
        ret[i] = u[i] + v[i];
    return ret;
}


double vnorm2(vfloat& v) {
    double norm = 0.;
    #pragma omp parallel for reduction(+: norm)
    for(int i = 0; i < (int)v.size(); i++)
        norm += v[i] * v[i];
    return norm;
}
