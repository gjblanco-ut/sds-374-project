#include <vector>
#include <algorithm>

#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H



typedef std::vector<float> vfloat;
typedef std::vector<vfloat > Matrix;

vfloat hadamard(const vfloat& u, const vfloat& v);
double inner_prod(const vfloat& u, const vfloat& v);
Matrix transpose(const Matrix& m);
vfloat prod(const Matrix& m, const vfloat& v);
vfloat sum(const vfloat& v, const vfloat& u);
double vnorm2(vfloat& v);
double sigmoid(double x);
double dsigmoid(double x);
vfloat sigmoid(const vfloat& v);
vfloat dsigmoid(const vfloat& v) ;


#endif