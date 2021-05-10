#include <vector>
#include <algorithm>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <string>
#include <fstream>
#include <random>
#include <cmath>
#include <cassert>
#include <mpi.h>

#include "linear_algebra.h"


#ifndef DISTRIB_NEURAL_NET_H 
#define DISTRIB_NEURAL_NET_H 

typedef std::pair<Matrix, vfloat> Layer;
typedef std::vector< std::pair< vfloat, int> > Dataset;

class DistribNeuralNet {
private:
    // layer info
    Layer layer;
    int nlayers;
    int net_output_size;

    // inter-process communication info
    int procno;
    int nprocs;
    MPI_Comm comm;

public:
    DistribNeuralNet(int inlsize, const std::vector<int>& sizes, MPI_Comm& c);
    void update_W(float r, const vfloat& d, const vfloat& a);
    void update_B(float r, const vfloat& d);

    int layer_id(int procno);

    std::pair<int,int> accuracy(const Dataset& dset, int ifirst, int ilast);
    double costval(const Dataset& dset, int ifirst, int ilast);
    void print() ;
    void train(const int EPOCHS, const float r, const Dataset& dataset, const int ifirst, const int ilast, int batchsize=10);
    std::vector<std::pair<vfloat, int> > evaluate(const Dataset& dset, int ifirst, int ilast);
};



#endif 