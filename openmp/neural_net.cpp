#include "neural_net.h"
#include "linear_algebra.h"

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
#include <chrono>
#include <omp.h>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

using namespace std;

NeuralNet::NeuralNet(int inlsize, const vector<int>& sizes, int outlsize)
    : input_layer_size(inlsize)
    , layers(sizes.size() + 2)
    , output_layer_size(outlsize)
    , eval_times(0., 0)
    , cost_fn_times(0., 0)
    , epoch_times(0., 0) {

    layers[0] = Layer(Matrix(0), vfloat(0)); // first layer. 
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0,1);

    for(int i = 1; i < (int)layers.size(); i++) {
        int ncols = i == 1? input_layer_size : sizes[i - 2];
        int nrows = i == (int)layers.size() - 1? output_layer_size : sizes[i - 1];
        layers[i] = Layer(Matrix(nrows, vfloat(ncols)), vfloat(nrows));
        
        for(int j = 0; j < nrows; j++) {
            layers[i].second[j] = distribution(generator) * .5;
            for(int k = 0; k < ncols; k++)
                layers[i].first[j][k] = distribution(generator) * .5;
        }
    }
}

int NeuralNet::get_input_layer_size() {
    return input_layer_size;
}
int NeuralNet::get_output_layer_size() {
    return output_layer_size;
}


void NeuralNet::update_net_W(int l, float r, const vfloat& d, const vfloat& a) {
    Matrix& W = layers[l].first;
    #pragma omp parallel for
    for(int i = 0; i < (int)W.size(); i++)
        for(int j = 0; j < (int)W[i].size(); j++) {
            W[i][j] -= r * d[i] * a[j];
        }
}

void NeuralNet::update_net_B(int l, float r, const vfloat& d) {
    vfloat& B = layers[l].second;
    #pragma omp parallel for
    for(int i = 0; i < (int)B.size(); i++) {
        B[i] -= r * d[i];
    }
}

// typedef vector<pair<Matrix, vfloat > > NeuralNet;
vfloat NeuralNet::evaluate_net(const vfloat& input) {
    vfloat a = input; // copy input
    for(int i = 1; i < (int)layers.size(); i++) {
        Layer layer = layers[i];
        a = sigmoid(sum(prod(layer.first, a), layer.second));
    }
    return a;
}

pair<int,int> NeuralNet::accuracy(const Dataset& dset, int ifirst, int ilast) {
    if(ilast < 0) ilast = (int)dset.size();
    int ncorrect = 0;
    double eval_time = 0;
    #pragma omp parallel for reduction (+: ncorrect)
    for(int i = ifirst; i < ilast; i++) {
        pair<vfloat, int> dat = dset[i];
        auto t1 = high_resolution_clock::now();
        vfloat a = evaluate_net(dat.first);
        auto t2 = high_resolution_clock::now();
        eval_time += duration_cast<milliseconds>(t2 - t1).count();
        pair<float, int> best = make_pair(a[0], 0);
        for(int label = 1; label < output_layer_size; label++) {
            best = max(make_pair(a[label], label), best);
        }
        if(best.second == dat.second) { // use the last reported to determine which is more likely to cause less slowdown
            ncorrect++;
        }
    }
    
    eval_times.first += eval_time / (ilast - ifirst);
    eval_times.second += ilast - ifirst;
    return make_pair(ncorrect, ilast-ifirst - ncorrect);
}


double NeuralNet::costval(const Dataset& dset, int ifirst, int ilast) {
    auto tc1 = high_resolution_clock::now();
    if(ilast < 0) ilast = (int)dset.size();
    double norm2 = 0;
    double eval_time = 0;
    #pragma omp parallel for reduction(+: norm, eval_time)
    for(int i = ifirst; i < ilast; i++) {
        auto dat = dset[i];
        auto t1 = high_resolution_clock::now();
        vfloat a = evaluate_net(dat.first);
        auto t2 = high_resolution_clock::now();
        eval_time += duration_cast<milliseconds>(t2 - t1).count();
        vfloat y(output_layer_size, 0);
        y[dat.second] = 1;
        double norm = 0;
        for(int j = 0; j < (int)y.size(); j++) {
            norm += (y[j] - a[j]) * (y[j] - a[j]);
        }
        norm2 += norm;
    }
    eval_times.first += eval_time / (ilast - ifirst); // take the avg per iteration.
    eval_times.second += ilast - ifirst;
    

    auto tc2 = high_resolution_clock::now();
    cost_fn_times.first += duration_cast<milliseconds>(tc2 - tc1).count();
    cost_fn_times.second++;
    return norm2;
}



void NeuralNet::print() {
    for(int l = 0; l < (int)layers.size(); l++) {
        cout << "Layer " << l << endl << "Matrix: " << endl;
        for(int i = 0; i < (int)layers[l].first.size(); i++) {
            cout << endl;
            for(int j = 0; j < (int)layers[l].first[i].size(); j++) {
                cout << " " << layers[l].first[i][j];
            }
        }
        cout << "Bias vector: " << endl;
        for(int i = 0; i < (int)layers[l].second.size(); i++) cout << layers[l].second[i] << " ";
        cout << endl;
        cout << "||\n||" << endl;
    }
}

void NeuralNet::train(const int EPOCHS, const float r, const Dataset& dataset, const int ifirst, const int ilast) {
    const int ntrain = ilast - ifirst;
    std::random_device rd;  
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distrib(0, ntrain);

    const int L = (int)layers.size();
    pair<int,int> previous_accuracy(0,1);
    auto te1 = high_resolution_clock::now();
    cout << "Data set size: " << dataset.size() << endl;
    for(int i = 0; i < EPOCHS; i++) {
        
        if(i % (int)dataset.size() == 0 && i > 0) {
            cout << "EPOCH " << i << endl;
            auto te2 = high_resolution_clock::now();
            epoch_times.first += duration_cast<milliseconds>(te2 - te1).count();
            epoch_times.second++;
            te1 = high_resolution_clock::now();
        }
        int k = distrib(gen);

        auto& x = dataset[k];

        vector<vfloat> a(L);
        a[0] = x.first; 

        vector<vfloat> D(L); // D[0] never gets used, but its just an empty vector
        // Forward pass
        for(int l = 1; l < L; l++) {
            Layer& layer = layers[l];
            vfloat z = sum(prod(layer.first, a[l - 1]), layer.second);
            a[l] = sigmoid(z); 
            D[l] = dsigmoid(z);
        }
        vector<vfloat> delta(L);
        vfloat minus_y(output_layer_size, 0.);
        minus_y[x.second] = -1.;
        // Backward pass
        delta[L - 1] = hadamard(D[L - 1], sum(a[L - 1], minus_y));
        for(int l = L - 2; l >= 1; l--) {
            delta[l] = hadamard(D[l], prod(transpose(layers[l + 1].first), delta[l + 1]));
        }
        // Gradient step
        for(int l = L - 1; l >= 1; l--) {
            update_net_W(l, r, delta[l], a[l - 1]);
            update_net_B(l, r, delta[l]);
        }

        // if(i % 20 == 0)
        //     cout << "End of epoch " << i << endl;
        if(i % 50000 == 0) {
            cout << "Calculating accuracy" << endl;
            pair<int,int> acc = accuracy(dataset, 0, ntrain);
            previous_accuracy = acc;
            cout << "Accuracy on training data: " << (acc.first * 1. / (acc.first + acc.second)) << endl;
            acc = accuracy(dataset, ntrain);
            cout << "Accuracy on leftover data: " << (acc.first * 1. / (acc.first + acc.second)) << endl;
        }
        if(i % 50000 == 0) {
            cout << "Calculating cost" << endl;
            double cost = costval(dataset, 0, ntrain);
            cout << cost << endl;
        }
        if(i % 10000 == 0 && epoch_times.second > 0 && eval_times.second > 0 && cost_fn_times.second > 0) {
            cout.precision(15);
            cout << ":::::Timing:::::\n";
            cout << "Average time per 100K evaluation ::: " << eval_times.first * 100000 / eval_times.second << " (msec) over " << eval_times.second << " times.\n";
            cout << "Average time per epoch ::: "<< epoch_times.first / epoch_times.second << " (msec) over " << epoch_times.second << " times.\n";
            cout << "Average time for cost function evaluation ::: " << cost_fn_times.first / cost_fn_times.second << " (msec) over " << cost_fn_times.second << " times.\n";
        }
        // if(i % 1000000 == 0) {
        //     print(net);
        // }
    }
}