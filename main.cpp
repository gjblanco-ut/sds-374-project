// /project/mine# make && ./main -e 1000000000 -l 5 -s 50 -r 0.05 -d ../mnist
#include "myfile.h"
#include <iostream>
#include <cstdio>
#include <cstring>
#include <string>
#include <fstream>
#include <random>
#include <cmath>
#include <cassert>

#include "cxxopts.hpp"

using namespace std;

const int IMSIZE = 28; // img

const int INPUT_LAYER_SIZE = IMSIZE * IMSIZE; //img
const int OUTPUT_LAYER_SIZE = 10; // img
// const int INPUT_LAYER_SIZE = 2; // xor
// const int OUTPUT_LAYER_SIZE = 2; // xor

typedef vector<float> vfloat;
typedef vector<vfloat > Matrix;
typedef pair<Matrix, vfloat> Layer;
typedef vector<Layer> NeuralNet;
typedef vector< pair< vector<float>, int> > Dataset;

void readXorDataset(Dataset& dataset) {
    dataset.push_back(pair<vector<float>, int>(vfloat({1, 0}), 1));
    dataset.push_back(pair<vector<float>, int>(vfloat({0, 1}), 1));
    dataset.push_back(pair<vector<float>, int>(vfloat({1, 1}), 0));
    dataset.push_back(pair<vector<float>, int>(vfloat({0, 0}), 0));
}

void readMnistDb(const string& directory, Dataset& dataset) {

    dataset.resize(0);

    for(int digit = 0; digit < 10; digit++) {
        string fname = directory + "/data" + to_string(digit);
        cout << "Reading " << fname << endl;
        ifstream fin(fname, std::ios::binary);
        if(!fin.good()) cout << "File " << fname << " does not exist!" << endl;
        vector<char> buffer (IMSIZE * IMSIZE,0); 

        while(!fin.eof()) {
            fin.read(buffer.data(), buffer.size());
            std::streamsize s=fin.gcount();
            // for(int i = 0; i < IMSIZE * IMSIZE; i++) if(buffer[i]) cout << "a" << hex << (int)buffer[i] << " ";
            vfloat data(buffer.size());
            for(int i = 0; i < buffer.size(); i++) data[i] = buffer[i] / 255.;
            dataset.push_back(pair<vector<float>, int>(data, digit));

            ///do with buffer
        }
    }
    
}

vfloat prod(const Matrix& m, const vfloat& v) {
    assert(m.size() > 0);
    assert(m[0].size() == v.size());
    assert(v.size() > 0);
    vfloat ret(m.size());
    #pragma omp parallel for 
    for(int i = 0; i < m.size(); i++) { // i: row
        ret[i] = 0;
        assert(m[i].size() == v.size());
        for(int j = 0; j < m[i].size(); j++) {
            ret[i] += m[i][j] * v[j];
        }
    }
    return ret;
}

vfloat sum(const vfloat& v, const vfloat& u) {
    assert(u.size() == v.size());
    vfloat ret(v.size());
    #pragma omp parallel for
    for(int i = 0; i < v.size(); i++)
        ret[i] = u[i] + v[i];
    return ret;
}

double sigmoid(double x) {
    return 1. / (1 + exp(-x));
}

double dsigmoid(double x) {
    double exp_x = exp(-x);
    return 1./ (1 + exp_x) * (1 - 1. / (1 + exp_x));
}

vfloat sigmoid(const vfloat& v) {
    vfloat ret(v.size());
    for(int i = 0; i < v.size(); i++)
        ret[i] = (float)sigmoid(v[i]);
    return ret;
}

vfloat dsigmoid(const vfloat& v) {
    vfloat ret(v.size());
    for(int i = 0; i < v.size(); i++)
        ret[i] = (float)dsigmoid(v[i]);
    return ret;
}

vfloat hadamard(const vfloat& u, const vfloat& v) {
    vfloat ret(u.size());
    assert(u.size() == v.size());
    #pragma omp parallel for
    for(int i = 0; i < ret.size(); i++) 
        ret[i] = u[i] * v[i];
    return ret;
}

double inner_prod(const vfloat& u, const vfloat& v) {
    double ret = 0;
    assert(u.size() == v.size());
    #pragma omp parallel for
    for(int i = 0; i < u.size(); i++)
        ret += u[i] * v[i];
    return ret;
}

Matrix transpose(const Matrix& m) {
    if(m.size() == 0) return Matrix();
    int n = m[0].size();
    Matrix ret(n, vfloat(m.size()));
    #pragma omp parallel for
    for(int i = 0; i < n; i++)
        for(int j = 0; j < m.size(); j++)
            ret[i][j] = m[j][i];
    return ret;
}

void update_net_W(Matrix& W, float r, const vfloat& d, const vfloat& a) {
    #pragma omp parallel for
    for(int i = 0; i < W.size(); i++)
        for(int j = 0; j < W[i].size(); j++) {
            W[i][j] -= r * d[i] * a[j];
        }
}

void update_net_B(vfloat& B, float r, const vfloat& d) {
    #pragma omp parallel for
    for(int i = 0; i < B.size(); i++) {
        B[i] -= r * d[i];
    }
}

// typedef vector<pair<Matrix, vfloat > > NeuralNet;
vfloat evaluate_net(const NeuralNet& net, const vfloat& input) {
    vfloat a = input; // copy input
    for(int i = 1; i < net.size(); i++) {
        Layer layer = net[i];
        a = sigmoid(sum(prod(layer.first, a), layer.second));
    }
    return a;
}

pair<int,int> net_accuracy(const NeuralNet& net, const Dataset& dset, int ifirst = 0, int ilast = -1) {
    if(ilast < 0) ilast = dset.size();
    int ncorrect = 0, nincorrect = 0;
    #pragma omp parallel for
    for(int i = ifirst; i < ilast; i++) {
        auto dat = dset[i];
        vfloat a = evaluate_net(net, dat.first);
        pair<float, int> best = make_pair(a[0], 0);
        for(int label = 1; label < OUTPUT_LAYER_SIZE; label++) {
            best = max(make_pair(a[label], label), best);
        }
        if(best.second == dat.second) {
            #pragma omp critical
            ncorrect++;
        } else {
            #pragma omp critical
            nincorrect++;
        }
    }
    return make_pair(ncorrect, nincorrect);
}


double costval(const NeuralNet& net, const Dataset& dset, int ifirst = 0, int ilast = -1) {
    if(ilast < 0) ilast = dset.size();
    int ncorrect = 0, nincorrect = 0;
    double norm2 = 0;
    #pragma omp parallel for
    for(int i = ifirst; i < ilast; i++) {
        auto dat = dset[i];
        vfloat a = evaluate_net(net, dat.first);
        vfloat y(OUTPUT_LAYER_SIZE, 0);
        y[dat.second] = 1;
        double norm = 0;
        for(int j = 0; j < y.size(); j++) {
            norm += (y[j] - a[j]) * (y[j] - a[j]);
        }
        norm2 += norm;
    }
    return norm2;
}

double vnorm2(vfloat& v) {
    double norm = 0.;
    #pragma omp parallel for reduction(+: norm)
    for(int i = 0; i < v.size(); i++)
        norm += v[i] * v[i];
    return norm;
}

void print(NeuralNet& nn) {
    for(int l = 0; l < nn.size(); l++) {
        cout << "Layer " << l << endl << "Matrix: " << endl;
        for(int i = 0; i < nn[l].first.size(); i++) {
            cout << endl;
            for(int j = 0; j < nn[l].first[i].size(); j++) {
                cout << " " << nn[l].first[i][j];
            }
        }
        cout << "Bias vector: " << endl;
        for(int i = 0; i < nn[l].second.size(); i++) cout << nn[l].second[i] << " ";
        cout << endl;
        cout << "||\n||" << endl;
    }
}

int main(int argc,char **argv) {

    cxxopts::Options options("EduDL", "FFNNs w BLIS");

    options.add_options() 
    ("h,help","usage information")
    ("d,dir", "Dataset directory",cxxopts::value<std::string>())
    ("l,levels", "Number of levels in the network",cxxopts::value<int>()->default_value("2"))
    ("s,sizes","Sizes of the levels",cxxopts::value<std::vector<int>>())
    ("o,optimizer", "Optimizer to be used, 0: SGD, 1: RMSprop",cxxopts::value<int>()->default_value("0"))
    ("e,epochs", "Number of epochs to train the network", cxxopts::value<int>()->default_value("1"))
    ("r,learningrate", "Learning rate for the optimizer", cxxopts::value<float>()->default_value("0.05"))
    ("b,batchsize", "Batch size for the training data", cxxopts::value<int>()->default_value("256"))
    ;

    auto result = options.parse(argc,argv);
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 1;
    }

    int network_optimizer = result["o"].as<int>();

    int epochs = epochs = result["e"].as<int>();
    cout << "EPOCHS" << epochs << endl;

    float r = result["r"].as<float>();
    int L = result["l"].as<int>();
    if(L < 2) L = 2; // l must be at least 2
    int EPOCHS = result["e"].as<int>();
    vector<int> sizes = result["s"].as<vector<int> >();

    if(sizes.size() > L - 2)
        sizes.resize(L - 2);
    while(sizes.size() < L - 2) {
        sizes.push_back(*sizes.rbegin());
    }
    sizes.push_back(OUTPUT_LAYER_SIZE);

    int batchSize = result["b"].as<int>();

    if (!result.count("dir")) {
        std::cout << "Must specify directory with -d/--dir option" << std::endl;
        return 1;
    }

    string mnist_loc = result["dir"].as<string>();
    std::cout << mnist_loc << std::endl;

    Dataset dataset;

    readMnistDb(mnist_loc, dataset); // img

    // readXorDataset(dataset); // xor

    // dataset IN dataset

    // net = pairs of W/b
    NeuralNet net(L);

    net[0] = Layer(Matrix(0), vfloat(0)); // first layer. 
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0,1);

    for(int i = 1; i < L; i++) {
        int ncols = i == 1? INPUT_LAYER_SIZE : sizes[i - 2];
        int nrows = sizes[i - 1];
        net[i] = Layer(Matrix(nrows, vfloat(ncols)), vfloat(nrows));
        
        for(int j = 0; j < nrows; j++) {
            net[i].second[j] = distribution(generator) * .5;
            for(int k = 0; k < ncols; k++)
                net[i].first[j][k] = distribution(generator) * .5;
        }
    }
    
    int seed = 0;
    shuffle (dataset.begin(), dataset.end(), std::default_random_engine(seed));
    const double training_fraction = .75;
    int ntrain = (int)(training_fraction * dataset.size());

    std::random_device rd;  
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distrib(0, ntrain-1);

    // net_accuracy(net, dataset, 0, ntrain);

    for(int i = 0; i < EPOCHS; i++) {
        if(i % 10000 == 0) cout << "EPOCH " << i << endl;
        int k = distrib(gen);

        auto& x = dataset[k];

        vector<vfloat> a(L);
        a[0] = x.first;

        vector<vfloat> D(L); // D[0] never gets used, but its just an empty vector
        // Forward pass
        for(int l = 1; l < L; l++) {
            Layer& layer = net[l];
            vfloat z = sum(prod(layer.first, a[l - 1]), layer.second);
            a[l] = sigmoid(z);
            D[l] = dsigmoid(z);
        }
        vector<vfloat> delta(L);
        vfloat minus_y(OUTPUT_LAYER_SIZE, 0.);
        minus_y[x.second] = -1.;
        // Backward pass
        delta[L - 1] = hadamard(D[L - 1], sum(a[L - 1], minus_y));
        for(int l = L - 2; l >= 1; l--) {
            delta[l] = hadamard(D[l], prod(transpose(net[l + 1].first), delta[l + 1]));
        }
        // Gradient step
        for(int l = L - 1; l >= 1; l--) {
            update_net_W(net[l].first, r, delta[l], a[l - 1]);
            update_net_B(net[l].second, r, delta[l]);
        }
        // if(i % 20 == 0)
        //     cout << "End of epoch " << i << endl;
        if(i % 50000 == 0) {
            cout << "Calculating accuracy" << endl;
            auto accuracy = net_accuracy(net, dataset, 0, ntrain);
            cout << "Accuracy on training data: " << (accuracy.first * 1. / (accuracy.first + accuracy.second)) << endl;
            accuracy = net_accuracy(net, dataset, ntrain);
            cout << "Accuracy on leftover data: " << (accuracy.first * 1. / (accuracy.first + accuracy.second)) << endl;
        }
        if(i % 50000 == 0) {
            cout << "Calculating cost" << endl;
            double cost = costval(net, dataset, 0, ntrain);
            cout << cost << endl;
            if(cost < 1e-1) {
                cout << "LESS THAN" << endl;
            }
        }
        // if(i % 1000000 == 0) {
        //     print(net);
        // }
    }

}