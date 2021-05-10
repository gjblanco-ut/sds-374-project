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

#include "linear_algebra.h"


#ifndef NEURAL_NET_H 
#define NEURAL_NET_H 

typedef std::pair<Matrix, vfloat> Layer;
typedef std::vector< std::pair< vfloat, int> > Dataset;

class NeuralNet {
private:
    int input_layer_size;
    std::vector<Layer> layers;
    int output_layer_size;
public:
    NeuralNet(int inlsize, const std::vector<int>& sizes, int outlsize);
    void update_net_W(int l, float r, const vfloat& d, const vfloat& a);

    void update_net_B(int l, float r, const vfloat& d);
    vfloat evaluate_net(const vfloat& input);

    std::pair<int,int> accuracy(const Dataset& dset, const std::pair<int,int>& prev_acc, int ifirst = 0, int ilast = -1);
    double costval(const Dataset& dset, int ifirst = 0, int ilast = -1);
    void print() ;

    int get_input_layer_size();
    void set_input_layer_size(int size);
    int get_output_layer_size();
    void set_output_layer_size(int size);

    void train(const int EPOCHS, const float r, const Dataset& dataset, const int ifirst = 0, const int ilast = -1);
};



#endif 