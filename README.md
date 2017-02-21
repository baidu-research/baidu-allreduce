# `baidu-allreduce`

`baidu-allreduce` is a small C++ library, demonstrating the ring allreduce and
ring allgather techniques. The goal is to provide a template for deep learning
framework authors to use when implementing these communication algorithms
within their respective frameworks.

A description of the ring allreduce with its application to deep learning is
available on the [Baidu SVAIL blog](http://research.baidu.com/bringing-hpc-techniques-deep-learning/).

## Installation

**Prerequisites:** Before compiling `baidu-allreduce`, make sure you have
installed CUDA (7.5 or greater) and an MPI implementation.

`baidu-allreduce` has been tested with [OpenMPI](https://www.open-mpi.org/),
but should work with any CUDA-aware MPI implementation, such as MVAPICH.

To compile `baidu-allreduce`, run

```bash
# Modify MPI_ROOT to point to your installation of MPI.
# You should see $MPI_ROOT/include/mpi.h and $MPI_ROOT/lib/libmpi.so.
# Modify CUDA_ROOT to point to your installation of CUDA.
make MPI_ROOT=/usr/lib/openmpi CUDA_ROOT=/path/to/cuda/lib64
```

You may need to modify your `LD_LIBRARY_PATH` environment variable to point to
your MPI implementation as well as your CUDA libraries.

To run the `baidu-allreduce` tests after compiling it, run
```bash
# On CPU.
mpirun --np 3 allreduce-test cpu

# On GPU. Requires a CUDA-aware MPI implementation.
mpirun --np 3 allreduce-test gpu
```

## Interface

The `baidu-allreduce` library provides the following C++ functions:

```c++
// Initialize the library, including MPI and if necessary the CUDA device.
// If device == NO_DEVICE, no GPU is used; otherwise, the device specifies which CUDA
// device should be used. All data passed to other functions must be on that device.
#define NO_DEVICE -1
void InitCollectives(int device);

// The ring allreduce. The lengths of the data chunks passed to this function
// must be the same across all MPI processes. The output memory will be
// allocated and written into `output`.
void RingAllreduce(float* data, size_t length, float** output);

// The ring allgather. The lengths of the data chunks passed to this function
// may differ across different devices. The output memory will be allocated and
// written into `output`.
void RingAllgather(float* data, size_t length, float** output);
```

The interface is simple and inflexible and is meant as a demonstration. The
code is fairly straightforward and the same technique can be integrated into
existing codebases in a variety of ways.
