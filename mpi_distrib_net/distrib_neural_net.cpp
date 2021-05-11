#include "distrib_neural_net.h"

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
using namespace std;

const int EVAL_BATCH = 1000;

DistribNeuralNet::DistribNeuralNet(int inlsize, const std::vector<int>& sizes, MPI_Comm& c) {
    comm = c;
    nlayers = (int)sizes.size();
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &procno);

    assert(nprocs % nlayers == 0);

    if(procno == 0) {
        cout << "NLAYERS " << nlayers << endl;
        cout << "NPROCS " << nprocs << endl;
    }

    int l = procno % nlayers;
    net_output_size = sizes[nlayers - 1];

    int ncols = l == 0? inlsize : sizes[l - 1];
    int nrows = sizes[l];
    // cout << procno << ": in constructor. W rows: " << nrows << "; W cols: " << ncols << endl;
    layer = Layer(Matrix(nrows, vfloat(ncols)), vfloat(nrows));

    // need to find out how to make this truly random, although it probably doesn't matter since layer init need not be random
    if(procno % nlayers == procno) {
        std::default_random_engine generator(l);
        std::normal_distribution<double> distribution(0,1);
        
        for(int j = 0; j < nrows; j++) {
            layer.second[j] = distribution(generator) * .5;
            for(int k = 0; k < ncols; k++)
                layer.first[j][k] = distribution(generator) * .5;
        }
        vector<MPI_Request> send_requests;
        vfloat sendvec(ncols * nrows + nrows);
        for(int i = 0; i < nrows; i++) {
            sendvec[ncols * nrows + i] = layer.second[i];
            for(int j = 0; j < ncols; j++) {
                sendvec[i * ncols + j] = layer.first[i][j];
            }
        }
        // sync layer between workers
        for(int sendto = procno + nlayers; sendto < nprocs; sendto += nlayers) {
            // cout << procno << ": ISEND size: " << sendvec.size() << " sendto: " << sendto << " tag: " << 0 << endl; 
            send_requests.push_back(MPI_Request());
            MPI_Isend(sendvec.data(), 
                        (int)sendvec.size(), 
                        MPI_FLOAT, 
                        sendto, 
                        0,
                        comm, &send_requests[send_requests.size() - 1]); 
        }
        // cout << "PROCNO " << procno << " Wait all " << endl;
        MPI_Waitall((int)send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);
        // cout << "PROCNO " << procno << " Done waiting " << endl;
    } else {
        // grab layer matrix
        vfloat buf(ncols * nrows + nrows);
        // cout << procno << ": RECV size: " << buf.size() << " recvfrom: " << (procno % nlayers) << " tag: " << 0 << endl; 
        MPI_Recv(buf.data(), (int)buf.size(), MPI_FLOAT, procno % nlayers, 0, comm, MPI_STATUS_IGNORE);
        for(int i = 0; i < nrows; i++) {
            layer.second[i] = buf[ncols * nrows + i];
            for(int j = 0; j < ncols; j++) {
                layer.first[i][j] = buf[i * ncols + j];
            }
        }
    }
    MPI_Barrier(comm);
    // assert(layer.first.size() == nrows);
    // for(int i = 0; i < nrows; i++) assert(layer.first[i].size() == ncols);
    if(procno == 0) cout << "Neural Network Constructed" << endl;
}

vector<pair<vfloat, int> > DistribNeuralNet::evaluate(const Dataset& dset, int ifirst, int ilast) {
    // cout << procno << ": SEND 777 size: " << W.size() << " sendto: " << sendto << " tag: " << procno << endl;
    // cout << procno << ": RECV 777 size: " << W.size() << " recvfrom: " << procno % nlayers << " tag: " << procno % nlayers << endl;
    assert(ilast > ifirst);
    assert(nlayers > 0);
    assert(nprocs % nlayers == 0); 
    vector<pair<vfloat, int> > a; // previous node entries 
    int layer_number = procno % nlayers;
    int topology_depth = nprocs / nlayers;
    int depth = procno / nlayers;
    int batchportionsize = (ilast - ifirst) / topology_depth + ((depth < (ilast - ifirst) % topology_depth)? 1 : 0); // round robin calculation
    // procno 0 init coworkers
    if(procno == 0) {
        // cout << "Evaluating " << ilast - ifirst << " samples" << endl;
        vector<MPI_Request> reqs;
        for(int i = 0; i < ilast - ifirst; i++) {
            if(i % topology_depth != 0) { // if sample i is not 0's
                reqs.push_back(MPI_Request());
                vfloat buf(dset[i + ifirst].first.begin(), dset[i + ifirst].first.end());
                buf.push_back(dset[i + ifirst].second);
                // cout << procno << ": ISEND *** size: " << buf.size() << " sendto: " << i % topology_depth * nlayers << " tag: " << i / topology_depth << endl;
                MPI_Isend(buf.data(), 
                            (int)buf.size(), 
                            MPI_FLOAT, 
                            i % topology_depth * nlayers, // sendto = procno + (i % depth) * nlayers 
                            i / topology_depth,
                            comm, &reqs[reqs.size() - 1]); 
            } else {
                a.push_back(dset[i + ifirst]);
            }
        }
        MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
        // cout << procno << ": The wait is over" << endl;
    }
    // everyone else: get output of previous layer
    if(procno != 0) {
        a.resize(batchportionsize);
        for(int i = 0; i < batchportionsize; i++) { // for each sample that procnum is processing
            vfloat buf(layer.first[0].size() + 1);
            //cout << procno << ": RECV *** size: " << buf.size() << " recvfrom: " <<(procno%nlayers == 0? 0 : procno - 1) << " tag: " << i << endl;
            MPI_Recv(buf.data(), (int)buf.size(), MPI_FLOAT, procno%nlayers == 0? 0 : procno - 1, i, comm, MPI_STATUS_IGNORE);
            //cout << procno << ": Received*** from " << (procno%nlayers == 0? 0 : procno - 1) << "tag " << i << endl; 
            for(int j = 0; j < (int)buf.size() - 1; j++) 
                a[i].first.push_back(buf[j]);
            a[i].second = buf[buf.size() - 1];
            // cout << procno << ": Calculated a" << endl; 
        }
    }
    // cout << procno << ": STOP1" << endl;
    vector<MPI_Request> reqs;
    // for ALL layers: send input to next layer (the last one sends to the first)
    for(int i = 0; i < batchportionsize; i++) { // for each sample that procnum is processing
        vfloat next_layer_input = sigmoid(sum(prod(layer.first, a[i].first), layer.second));
        reqs.push_back(MPI_Request());
        vfloat buf(next_layer_input.begin(), next_layer_input.end());
        buf.push_back(a[i].second);
        // cout << procno << ": ISEND xxx size: " << buf.size() << " sendto: " << (layer_number < nlayers - 1? procno + 1 : procno - nlayers + 1) << " tag: " << i << endl;
        MPI_Isend(buf.data(), 
                    (int)buf.size(), 
                    MPI_FLOAT, 
                    layer_number < nlayers - 1? procno + 1 : procno - nlayers + 1,
                    i,
                    comm, &reqs[reqs.size() - 1]); 
    }
    
    MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    // cout << procno << ": STOP3" << endl;

    // layer zero reduce
    if(layer_number == 0) {
        vector<pair<vfloat, int> > a;
        for(int i = procno / nlayers; i < ilast - ifirst; i += topology_depth) {
            // cout << procno << ": STOP4" << endl;
            vfloat buf(net_output_size + 1);
            // cout << procno << ": STOP5" << endl;
            // cout << procno << ": RECV AAA size: " << buf.size() << " recvfrom: " << layer_number + nlayers - 1 << " tag: " << i / topology_depth << endl;
            MPI_Recv(buf.data(), (int)buf.size(), MPI_FLOAT, procno + nlayers - 1, i / topology_depth, comm, MPI_STATUS_IGNORE);
            // cout << procno << ": Received AAA" << endl;
            int label = (int)*buf.rbegin();
            buf.pop_back();
            a.push_back(make_pair(buf, label));
        }
        // cout << procno << ": RET LAYER 0" << endl;
        return a;
    }
    // cout << procno << ": RET LAYER NOT 0" << endl;
    return vector<pair<vfloat,int> >();
}

pair<int,int> DistribNeuralNet::accuracy(const Dataset& dset, int ifirst, int ilast) {
    // EVALUATE NETWORK. results will be accross mpi nodes of layer 0
    //cout << procno << ": SEND 777 size: " << W.size() << " sendto: " << sendto << " tag: " << procno << endl;
    // cout << procno << ": RECV 777 size: " << W.size() << " recvfrom: " << procno % nlayers << " tag: " << procno % nlayers << endl;
    
    assert(ilast > ifirst);
    vector<pair<vfloat, int> > a = evaluate(dset, ifirst, ilast);
    int layer_number = procno % nlayers;
    if(layer_number != 0) return make_pair(-1,-1);

    int ncorrect = 0;
    int nincorrect = 0;
    
    for(int i = 0; i < (int)a.size(); i++) {
        vfloat& ai = a[i].first;
        int real_label = a[i].second;

        pair<float, int> best = make_pair(ai[0], 0);
        for(int label = 1; label < net_output_size; label++) {
            best = max(make_pair(ai[label], label), best);
        }
        if(best.second == real_label) { // use the last reported to determine which is more likely to cause less slowdown
            ncorrect++;
        } else if (best.second != real_label){
            nincorrect++;
        }
    }

    int acc[2] = {ncorrect, nincorrect};

    if(procno != 0) {
        MPI_Send(acc, 2, MPI_INT, 0, 0, comm);
    }
    else {
        for(int recvfrom = nlayers; recvfrom < nprocs; recvfrom += nlayers) {
            int buf[2];
            MPI_Recv(buf, 2, MPI_INT, recvfrom, 0, comm, MPI_STATUS_IGNORE);
            acc[0] += buf[0];
            acc[1] += buf[1];
        }
        return pair<int,int> (acc[0], acc[1]);
    }
    return pair<int,int>(-1,-1);
}

double DistribNeuralNet::costval(const Dataset& dset, int ifirst, int ilast) {
    //cout << procno << ": SEND 777 size: " << W.size() << " sendto: " << sendto << " tag: " << procno << endl;
    // cout << procno << ": RECV 777 size: " << W.size() << " recvfrom: " << procno % nlayers << " tag: " << procno % nlayers << endl;
    assert(ilast > ifirst);
    double total_norm = 0;
    for(int k = ifirst; k < ilast; k += EVAL_BATCH) {
        MPI_Barrier(comm);
        vector<pair<vfloat, int> > a = evaluate(dset, k, min(k + EVAL_BATCH, ilast));
        // cout << "HERE AFTER EVAL" << endl;
        int layer_number = procno % nlayers;
        if(layer_number != 0) continue;
        
        double norm2 = 0;
        for(int i = 0; i < (int)a.size(); i++) {
            vfloat& ai = a[i].first;
            int real_label = a[i].second;
            vfloat y(net_output_size, 0);
            y[real_label] = 1;
            double norm = 0;
            for(int j = 0; j < (int)y.size(); j++) {
                norm += (y[j] - ai[j]) * (y[j] - ai[j]);
            }
            norm2 += norm;
            // cout << procno << ": NORM " << norm2 << endl;
        }

        if(procno != 0) {
            MPI_Send(&norm2, 1, MPI_DOUBLE, 0, 0, comm);
        } else {
            // procno zero
            for(int recvfrom = nlayers; recvfrom < nprocs; recvfrom += nlayers) {
                double norm;
                MPI_Recv(&norm, 1, MPI_DOUBLE, recvfrom, 0, comm, MPI_STATUS_IGNORE);
                norm2 += norm;
                // cout << procno << ": RECV NORM " << norm2 << endl;
            }
            total_norm += norm2;
        }
    }
    if(procno == 0)
        return total_norm;
    else return -1;
    
}

// process 0 takes care of the disk IO (loading the dataset)
// process 0 also works as layer 1 (layer 0 is the input layer)
void DistribNeuralNet::train(const int EPOCHS, const float r, const Dataset& dataset, int ifirst, int ilast, int batchsize) {

    // cout << procno << ": ISEND size: " << sendvec.size() << " sendto: " << sendto << " tag: " << 0 << endl; 
    // cout << procno << ": RECV size: " << buf.size() << " recvfrom: " << (procno % nlayers) << " tag: " << 0 << endl;

    // send arguments to other processes
    if(procno == 0) {
        int args[2] = {ifirst, ilast};
        for(int i = 1; i < nprocs; i++)
            MPI_Send(args, 2, MPI_INT, i, 0, comm);
    } else {
        int buf[2];
        MPI_Recv(buf, 2, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
        ifirst = buf[0];
        ilast = buf[1];
    }
    
    int ntrain = ilast - ifirst;
    std::random_device rd;  
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, ntrain);

    assert(nprocs % nlayers == 0); // throw error here
    const int topology_depth = nprocs / nlayers;
    if(batchsize % topology_depth) {
        batchsize += topology_depth - batchsize % topology_depth;
        if(procno == 0) {
            cout << "Chose new batch size: " << batchsize << "." << endl;
        }
    }

    int layer_number = procno % nlayers;
    int current_layer_size = (int)layer.first[0].size();
    int next_layer_size = (int)layer.first.size();
    // procno % nlayers is the layer of the node
    // ----- N LAYERS -------
    // p0   p1   p2   ...  pN-1
    // pN   pN+1 pN+2 ... p2N
    // ...

    for(int epoch = 0; epoch < EPOCHS; epoch++) {

        // cout << procno << ": Training started " << procno << endl;

        if(epoch % 10000 == 0) {
            double cost = costval(dataset, ifirst, ilast);
            if(procno == 0) {
                cout << "Cost: " << cost << endl;
            }
        }
        
        if(procno == 0 && epoch % 1000 == 0)
            cout << "EPOCH " << epoch << endl;
        // first send all the inputs from zero to everyone else in the first layer
        
        // STEP 1: GET THE prev_data ARRAY FOR ALL THE NODES
        vector<pair<vfloat, int> > prev_data; // a^{[l-1]}, y
        if(procno == 0) { // procno 0: send datasets to everyone else in layer 0
            vector<MPI_Request> send_requests;
            for(int i = 0; i < batchsize; i++) {
                int k = distrib(gen);
                auto& x = dataset[k];
                int sendto = (i % topology_depth) * nlayers; // round robin on the nodes of the same layer
                if(sendto == 0) {
                    prev_data.push_back(x);
                } else {
                    vfloat send_vec(x.first.begin(), x.first.end());
                    send_vec.push_back((float)x.second); 
                    send_requests.push_back(MPI_Request());
                    // if(procno == 5) cout << procno << ": ISEND 1 size: " << send_vec.size() << " sendto: " << sendto << " tag: " << i / topology_depth << endl; 
                    
                    MPI_Isend(send_vec.data(), 
                        (int)send_vec.size(), 
                        MPI_FLOAT, 
                        sendto, 
                        i / topology_depth, // tag: sample number in the batch 
                        comm, &send_requests[send_requests.size() - 1]);
                }
            }
            MPI_Waitall((int)send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);
            // cout << procno << ": SENT ALL" << endl;
        }
        // if(procno == 5) cout << procno << ": ISEND 1 size: " << endl;
        const int nsamples_per_node = batchsize / topology_depth;
        // Everyone else receives the a^{[l-1]} data (although nodes in layers l+1 receive it later)
        if(procno != 0) {
            // procno nlayers+1 receives 1, 1 + topology_depth, 1 + 2*topology_depth, ... , etc
            for(int tag = 0; tag < nsamples_per_node; tag++) {
                vfloat recv_vec(current_layer_size + 1);
                // cout << procno << ": RECV 1 size: " << recv_vec.size() << " recvfrom: " << (layer_number == 0? 0 : procno - 1) << " tag: " << tag << "layer number:" << layer_number << endl;
                MPI_Recv(recv_vec.data(), 
                        (int)recv_vec.size(), 
                        MPI_FLOAT, 
                        layer_number == 0? 0 : procno - 1, 
                        tag, 
                        comm, MPI_STATUS_IGNORE);
                
                prev_data.push_back(make_pair(recv_vec, *recv_vec.rbegin()));
                prev_data.rbegin()->first.pop_back(); // delete last element (pair.second => label)
                // cout << procno << ": PASS " << tag << endl;
            }
        }
        // END STEP 1
        //cout << procno << ": STEP 2" << endl;
        // if(procno == 5) cout << procno << ": STEP 2: " << endl;
        // STEP 2: Calculate z, a, D (first loop in pdf pseudocode) and Send "a" to next layer mpi procs
        vector<vfloat> z(nsamples_per_node); // one z vector per sample in batch portion
        vector<vfloat> a(nsamples_per_node);
        vector<vfloat> D(nsamples_per_node);
        vector<MPI_Request> send_requests;
        // cout << procno << ": SAMP PER NODE " << nsamples_per_node << endl;

        for(int i = 0; i < nsamples_per_node; i++) {
            vfloat& a_1 = prev_data[i].first;
            z[i] = sum(prod(layer.first, a_1), layer.second);
            a[i] = sigmoid(z[i]);
            D[i] = dsigmoid(z[i]);
            assert(D[i].size() > 0);

            if(procno % nlayers != nlayers - 1) { // if current layer is NOT the last layer
                int sendto = procno + 1;
                // cout << procno << ": IM HERE" << 3%3 << endl;
                vfloat send_vec(a[i].begin(), a[i].end());
                send_vec.push_back((float)prev_data[i].second); 
                send_requests.push_back(MPI_Request());
                // cout << procno << ": ISEND ** size: " << send_vec.size() << " sendto: " << sendto << " tag: " << i << endl; 
                MPI_Isend(send_vec.data(),
                    (int)send_vec.size(), 
                    MPI_FLOAT, 
                    sendto, 
                    i, // tag: sample number, procno ==> "nproc sent me its i-th sample"
                    comm, &send_requests[send_requests.size() - 1]);
            }
        }
        MPI_Waitall((int)send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);

        // END STEP 2
        vector<pair<vfloat, int> > next_data(nsamples_per_node);

        vector<vfloat> delta(nsamples_per_node);
        // if(procno == 5) cout << procno << ": STEP 3: " << endl;
        // STEP 3: Calculate and send delta of last layer
        //cout << procno << ": STEP 3 " << endl;
        if(procno % nlayers == nlayers - 1) {
            assert(batchsize / topology_depth == (int)nsamples_per_node);
            for(int i = 0; i < nsamples_per_node; i++) {
                vfloat minus_y(next_layer_size, 0.);
                minus_y[prev_data[i].second] = -1.;
                // Backward pass
                delta[i] = hadamard(D[i], sum(a[i], minus_y));
                // cout << procno << ": delta size " << delta[i].size() << endl;
            }
        }
        // END STEP 3
        // if(procno == 5) cout << procno << ": STEP 4: " << endl;
        // STEP 4: Receive next layer's data for all but the last layer and Calculate delta. Then Send it to the previous layer
        //cout << procno << ": STEP 4 " << endl;
        if(procno % nlayers != nlayers - 1) { // if not last layer
            for(int tag = 0; tag < nsamples_per_node; tag++) {
                // cout << procno << ":HEREHERE" << endl;
                vfloat recv_vec(next_layer_size + 1);
                // cout << procno << ": RECV size: " << recv_vec.size() << " recvfrom: " << procno + 1 << " tag: " << tag << endl;
                MPI_Recv(recv_vec.data(), (int)recv_vec.size(), MPI_FLOAT, procno + 1, tag, comm, MPI_STATUS_IGNORE);
                // cout << procno << ": GOT NEXT LAYER DATA " << recv_vec.size() << endl;
                next_data[tag] = make_pair(recv_vec, *recv_vec.rbegin());
                next_data[tag].first.pop_back(); // delete last element (pair.second => label)
                assert(next_data[tag].first.size() == next_layer_size);
                // cout << procno << ": NEXT DATA SIZE " << next_data.rbegin()->first.size() << ", how many => " << next_data.size() << "TAG " << tag << endl;
            }
            // vector<MPI_Request> send_requests;
            // if(procno % nlayers != 0) { // for all but the first layer
            for(int i = 0; i < nsamples_per_node; i++) {
                // cout << procno << ": GOT NEXT LAYER DATA data i:" << i << "dims=> " << next_data[i].first.size() << " and " << D[i].size() << endl;
                delta[i] = hadamard(D[i], next_data[i].first);
                // cout << procno << ": GOT NEXT LAYER DATA " << delta[i].size() << endl;
            }
            // }
        }
        // cout << procno << ":BEFORE" << endl;
        // STEP 4 1/2 SEND DELTAS TO PREVIOUS LAYER
        vector<MPI_Request> delta_calc_requests;
        if(procno % nlayers != 0) { // for all but the first layer
            for(int i = 0; i < nsamples_per_node; i++) {
                // cout << procno << ": i/nsamples" << i << "/" << nsamples_per_node << endl;
                int sendto = procno - 1;
                vfloat send_vec = prod(transpose(layer.first), delta[i]);
                // cout << procno << ":AFTER" << endl;
                send_vec.push_back((float)prev_data[i].second); 
                delta_calc_requests.push_back(MPI_Request());
                // cout << procno << ": ISEND ** size: " << send_vec.size() << " sendto: " << sendto << " tag: " << i << endl; 
                MPI_Isend(send_vec.data(), 
                    (int)send_vec.size(), 
                    MPI_FLOAT, 
                    sendto, 
                    i, // tag: sample number, procno ==> "procno sent me its i-th sample"
                    comm, &delta_calc_requests[delta_calc_requests.size() - 1]);
            }
        }
        MPI_Waitall((int)delta_calc_requests.size(), delta_calc_requests.data(), MPI_STATUSES_IGNORE);

        // END STEP 4
        // if(procno == 5) cout << procno << ": STEP 5: " << endl;

        // STEP 5: Combine the deltas in each layer
        //cout << procno << ": STEP 5 " << endl;
        int wcols = layer.first[0].size();
        int wrows = layer.first.size();
        int wsize = wrows * wcols;
        vfloat W(wsize + wrows, 0.);

        // for each b vector: b[j] -= delta[i][j] / batchsize;
        // for each W matrix: W[j][k] -= r*delta[j] * prev_data[i].first[k] / batchsize; where prevdata[i].first[k] is a^[l-1] [k] for the ith sample
        if(procno % nlayers == procno) { // if this node is the first in the layer


            // vector<float> deltaW(wsize, 0);
            //cout << procno << ": SOME DATA " << wrows << " " << wcols << " " << nsamples_per_node << " disize " << delta[0].size() << endl;
            // update weights and biases matrix
            // #pragma omp parallel for
            for(int i = 0; i < nsamples_per_node; i++) {
                for(int j = 0; j < wrows; j++) {
                    assert(layer.second.size() == delta[i].size());
                    // layer.second[j] += delta[i][j] / batchsize;
                    layer.second[j] -= delta[i][j] / batchsize;
                    for(int k = 0; k < wcols; k++) {
                        assert(layer.first[j].size() == prev_data[i].first.size());
                        // layer.first[j][k] += r * delta[i][j] * prev_data[i].first[k] / batchsize;
                        layer.first[j][k] -= r * delta[i][j] * prev_data[i].first[k] / batchsize;
                        // deltaW[j * wcols + k] += r * delta[i][j] * prev_data[i].first[k] / batchsize;
                    }
                }
            }
            //cout << procno << ": STEP 5 LOOP" << endl;
            // W: delta avgs (W is the product r * delta[l]a[l-1] in the other nodes)
            // receive other vector's delta W and delta b and update
            for(int tag = 1, sendto = procno + nlayers; sendto < nprocs; sendto += nlayers, tag++) { // tag is the order of the coworker node
                // cout << procno << ": RECV size: " << W.size() << " recvfrom: " << sendto << " tag: " << tag << "epoch " << epoch << endl;
                MPI_Recv(W.data(), (int)W.size(), MPI_FLOAT, sendto, tag, comm, MPI_STATUS_IGNORE);
                // cout << procno << ": Received all deltas complete" << endl;
                #pragma omp parallel for
                for(int i = 0; i < nsamples_per_node; i++) {
                    for(int j = 0; j < wrows; j++) {
                        // layer.second[j] += W[wsize + j];
                        layer.second[j] -= W[wsize + j];
                        for(int k = 0; k < wcols; k++) {
                            // layer.first[j][k] += W[j * wcols + k];
                            layer.first[j][k] -= W[j * wcols + k];
                            // deltaW[j * wcols + k] += W[j * wcols + k];
                        }
                    }
                }
            }
            //cout << procno << ": ADFASDFASFASF " << layer.first.size() << endl;
            // sync layer with coworkers
            for(int i = 0; i < (int)layer.first.size(); i++) {
                assert(W.size() - wsize == layer.second.size());
                assert(wsize + i < W.size());
                assert(i < layer.second.size());
                W[wsize + i] = layer.second[i];
                for(int j = 0; j < (int)layer.first[i].size(); j++) {
                    // cout << "WCOLS " << wcols << " layer.first[" << i << "].size() " << layer.first[i].size() << endl;
                    assert(layer.first[i].size() == wcols);
                    assert(i * wcols + j < W.size());
                    W[i * wcols + j] = layer.first[i][j];
                }
            }
            //cout << procno << ": ADFASDFASFASF 2" << endl;
            for(int tag = 1, sendto = procno + nlayers; sendto < nprocs; sendto += nlayers, tag++) {
                // cout << procno << ": SEND 777 size: " << W.size() << " sendto: " << sendto << " tag: " << procno << endl;
                MPI_Send(W.data(), W.size(), MPI_FLOAT, sendto, procno, comm);
            }
            //cout << procno << ": ADFASDFASFASF 3" << endl;
            // cout << procno << ": WMATRIX";
            // for(int i = 0; i < wrows; i++) {
            //     cout << endl;
            //     for(int j = 0; j < wcols; j++) {
            //         cout << deltaW[i * wcols + j] << " ";
            //     }
            // }
            
        } else { // LAST LINE I DEBUGGED
            // if(procno == 5) cout << procno << ": STEP 6: " << endl;
            // send delta w and delta bias data
            // #pragma omp parallel for
            for(int i = 0; i < nsamples_per_node; i++) {
                // if(procno == 5) cout << procno << ": STEP 6z: " << endl;
                for(int j = 0; j < wrows; j++) {
                    W[wsize + j] += delta[i][j] / batchsize;
                    for(int k = 0; k < wcols; k++) {
                        assert(i < delta.size());
                        assert(j < delta[i].size());
                        assert(j * wcols + k < W.size());
                        assert(i < prev_data.size());
                        assert(k < prev_data[i].first.size());
                        W[j * wcols + k] += r * delta[i][j] * prev_data[i].first[k] / batchsize;
                    }
                }
            }
            // if(procno == 5) cout << procno << ": STEP 6b: " << W.size() << endl;
            // send W and b to zero
            //cout << procno << ": ISEND ** size: " << W.size() << " sendto: " << procno % nlayers << " tag: " << procno / nlayers << endl; 
            // if(procno == 5) 
            // cout << procno << ": SEND " << "size " << W.size() << " to " << procno % nlayers << " tag " << procno / nlayers << " epoch " << epoch << endl;
            MPI_Send(W.data(), W.size(), MPI_FLOAT, procno % nlayers, procno / nlayers, comm);
            // if(procno == 5) cout << procno << ": SENT: " << wsize + wrows << endl;

            //cout << procno << ": FINAL SEND SENT" << endl;

            // receive layer matrix and bias. Update
            // if(procno == 5) 
            vfloat W2(wsize + wrows);
            // cout << procno << ": RECV 777 size: " << W2.size() << " recvfrom: " << procno % nlayers << " tag: " << procno % nlayers << endl;
            MPI_Recv(W2.data(), (int)W2.size(), MPI_FLOAT, procno % nlayers, procno % nlayers, comm, MPI_STATUS_IGNORE);
            // if(procno == 5) cout << procno << ": AFTER RECV" << endl;

            for(int i = 0; i < (int)layer.first.size(); i++) {
                assert(layer.second.size() == layer.first.size());
                assert(W2.size() > wsize + i);
                layer.second[i] = W2[wsize + i];
                for(int j = 0; j < (int)layer.first[i].size(); j++) {
                    assert(i * wcols + j < W2.size());
                    layer.first[i][j] = W2[i * wcols + j];
                }
            }
        }
        // if(procno == 5) cout << procno << ": BARRIER: " << endl;
        MPI_Barrier(comm);
    }
}