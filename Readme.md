# Project structure

The project consists of three implementations of a neural network that classifies handwritten digits. 

The three implementations are in the subdirectories /serial, /openmp and /mpi_distrib_net.

# Setup

Check that the MNIST data is in the folder ../mnist (assuming that you are in the root folder of the project).

To run either version, execute the following commands:

```
cd openmp
make run
```

To try out your own parameters, edit the Makefile of the corresponding project, and modify the options in the `run` section.
To run the same test cases as in the project report, run 
```
make run_large 
# or 
make run_medium 
# or 
make run_small
```

Finally, to see some benchmarks on the linear_algebra library, go in the /benchmarks folder and run
```
make benchmark_serial
```
or
```
make benchmark_omp
```

They are not compiles with the `-O2` flag so you can clearly see the effect of the openmp library.


# Interpreting the output

"Cost" refers to the cost function, which is the sum of the norms of the differences between the output vectors and the expected vector. (the standard cost function)

For openmp and serial, there is an additional "accuracy" output, that measures how well the network performs for the training and test subsets.


