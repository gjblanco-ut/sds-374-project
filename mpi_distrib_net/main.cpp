// /project/mine# make && ./main -e 1000000000 -l 5 -s 50 -r 0.05 -d ../mnist
#include <iostream>
#include <cstdio>
#include <cstring>
#include <string>
#include <fstream>
#include <random>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <mpi.h>

#include "../cxxopts.hpp"

#include "linear_algebra.h"
#include "distrib_neural_net.h"

using namespace std;

const int IMSIZE = 28; // img

const int INPUT_LAYER_SIZE = IMSIZE * IMSIZE;
const int OUTPUT_LAYER_SIZE = 10;

// typedef pair<Matrix, vfloat> Layer;
// typedef vector<Layer> NeuralNet;


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
            // std::streamsize s=fin.gcount();
            // for(int i = 0; i < IMSIZE * IMSIZE; i++) if(buffer[i]) cout << "a" << hex << (int)buffer[i] << " ";
            vfloat data(buffer.size());
            for(int i = 0; i < (int)buffer.size(); i++) data[i] = buffer[i] / 255.;
            dataset.push_back(pair<vector<float>, int>(data, digit));

            ///do with buffer
        }
    }
    
}


int main(int argc,char **argv) {

    MPI_Comm comm = MPI_COMM_WORLD;
    int nprocs, procno;
    MPI_Init(0, 0);
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &procno);

    cxxopts::Options options("EduDL", "FFNNs w BLIS");

    options.add_options() 
    ("h,help","usage information")
    ("d,dir", "Dataset directory",cxxopts::value<std::string>())
    ("l,levels", "Number of levels in the network",cxxopts::value<int>()->default_value("2"))
    ("s,sizes","Sizes of the levels",cxxopts::value<std::vector<int>>())
    ("e,epochs", "Number of epochs to train the network", cxxopts::value<int>()->default_value("1"))
    ("r,learningrate", "Learning rate for the optimizer", cxxopts::value<float>()->default_value("0.05"))
    ;

    auto result = options.parse(argc,argv);
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 1;
    }

    int epochs = result["e"].as<int>();
    cout << "EPOCHS" << epochs << endl;

    float r = result["r"].as<float>();
    int L = result["l"].as<int>();
    if(L < 2) L = 2; // l must be at least 2
    int EPOCHS = result["e"].as<int>();
    vector<int> sizes = result["s"].as<vector<int> >();

    if((int)sizes.size() > L - 2)
        sizes.resize(L - 2);
    while((int)sizes.size() < L - 2) {
        sizes.push_back(*sizes.rbegin());
    }
    sizes.push_back(OUTPUT_LAYER_SIZE);

    if (!result.count("dir")) {
        std::cout << "Must specify directory with -d/--dir option" << std::endl;
        return 1;
    }

    string mnist_loc = result["dir"].as<string>();
    std::cout << mnist_loc << std::endl;

    Dataset dataset;
    if(procno == 0) {
        readMnistDb(mnist_loc, dataset); // img
    }
    
    // readXorDataset(dataset); // xor

    // dataset IN dataset

    // net = pairs of W/b 
    DistribNeuralNet net(INPUT_LAYER_SIZE, sizes, comm);
    
    
    int seed = 0;
    std::shuffle (dataset.begin(), dataset.end(), std::default_random_engine(seed));
    const double training_fraction = .75;
    int ntrain = (int)(training_fraction * dataset.size());
    net.train(EPOCHS, r, dataset, 0, ntrain);

    MPI_Finalize();

    // net_accuracy(net, dataset, 0, ntrain);

    // optimize

}