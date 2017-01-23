#include <vector>
#include <stdexcept>
#include <cassert>
#include <cstring>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>
#include <mpi.h>

#include "collectives.h"

struct MPIGlobalState {
    // The CUDA device to run on, or -1 for CPU-only.
    int device = -1;

    // A CUDA stream (if device >= 0) initialized on the device
    cudaStream_t stream;

    // Whether the global state (and MPI) has been initialized.
    bool initialized = false;
};

// MPI relies on global state for most of its internal operations, so we cannot
// design a library that avoids global state. Instead, we centralize it in this
// single global struct.
static MPIGlobalState global_state;

// Initialize the library, including MPI and if necessary the CUDA device.
// If device == -1, no GPU is used; otherwise, the device specifies which CUDA
// device should be used. All data passed to other functions must be on that device.
//
// An exception is thrown if MPI or CUDA cannot be initialized.
void InitCollectives(int device) {
    if(device < 0) {
        // CPU-only initialization.
        int mpi_error = MPI_Init(NULL, NULL);
        if(mpi_error != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Init failed with an error");
        }

        global_state.device = -1;
    } else {
        // GPU initialization on the given device.
        //
        // For CUDA-aware MPI implementations, cudaSetDevice must be called
        // before MPI_Init is called, because MPI_Init will pick up the created
        // CUDA context and use it to create its own internal streams. It uses
        // these internal streams for data transfers, which allows it to
        // implement asynchronous sends and receives and allows it to overlap
        // GPU data transfers with whatever other computation the GPU may be
        // doing.
        //
        // It is not possible to control which streams the MPI implementation
        // uses for its data transfer.
        cudaError_t error = cudaSetDevice(device);
        if(error != cudaSuccess) {
            throw std::runtime_error("cudaSetDevice failed with an error");
        }

        // When doing a CUDA-aware allreduce, the reduction itself (the
        // summation) must be done on the GPU with an elementwise arithmetic
        // kernel. We create our own stream to launch these kernels on, so that
        // the kernels can run independently of any other computation being done
        // on the GPU.
        cudaStreamCreate(&global_state.stream);
        if(error != cudaSuccess) {
            throw std::runtime_error("cudaStreamCreate failed with an error");
        }

        int mpi_error = MPI_Init(NULL, NULL);
        if(mpi_error != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Init failed with an error");
        }

        global_state.device = device;
    }
    global_state.initialized = true;
}

// Allocate a new memory buffer on CPU or GPU.
float* alloc(size_t size) {
    if(global_state.device < 0) {
        // CPU memory allocation through standard allocator.
        return new float[size];
    } else {
        // GPU memory allocation through CUDA allocator.
        void* memory;
        cudaError_t error = cudaMalloc(&memory, sizeof(float) * size);
        if(error != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed with an error");
        }
        return (float*) memory;
    }
}

// Deallocate an allocated memory buffer.
void dealloc(float* buffer) {
    if(global_state.device < 0) {
        // CPU memory deallocation through standard allocator.
        delete[] buffer;
    } else {
        // GPU memory deallocation through CUDA allocator.
        cudaFree(buffer);
    }
}

// Copy data from one memory buffer to another on CPU or GPU.
// Both buffers must resize on the same device.
void copy(float* dst, float* src, size_t size) {
    if(global_state.device < 0) {
        // CPU memory allocation through standard allocator.
        std::memcpy((void*) dst, (void*) src, size * sizeof(float));
    } else {
        // GPU memory allocation through CUDA allocator.
        cudaMemcpyAsync((void*) dst, (void*) src, size * sizeof(float),
                        cudaMemcpyDeviceToDevice, global_state.stream);
        cudaStreamSynchronize(global_state.stream);
    }
}

// GPU kernel for adding two vectors elementwise.
__global__ void kernel_add(const float* x, const float* y, const int N, float* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
      out[i] = x[i] + y[i];
    }
}


// Copy data from one memory buffer to another on CPU or GPU.
// Both buffers must resize on the same device.
void reduce(float* dst, float* src, size_t size) {
    if(global_state.device < 0) {
        // Accumulate values from `src` into `dst` on the CPU.
        for(size_t i = 0; i < size; i++) {
            dst[i] += src[i];
        }
    } else {
        // Launch a GPU kernel to accumulate values from `src` into `dst`.
        kernel_add<<<32, 256, 0, global_state.stream>>>(src, dst, size, dst);
        cudaStreamSynchronize(global_state.stream);
    }
}

// Collect the input buffer sizes from all ranks using standard MPI collectives.
// These collectives are not as efficient as the ring collectives, but they
// transmit a very small amount of data, so that is OK.
std::vector<size_t> AllgatherInputLengths(int size, size_t this_rank_length) {
    std::vector<size_t> lengths(size);
    MPI_Allgather(&this_rank_length, 1, MPI_UNSIGNED_LONG,
                  &lengths[0], 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
    return lengths;
}


/* Perform a ring allreduce on the data. The lengths of the data chunks passed
 * to this function must be the same across all MPI processes. The output
 * memory will be allocated and written into `output`.
 *
 * Assumes that all MPI processes are doing an allreduce of the same data,
 * with the same size.
 *
 * A ring allreduce is a bandwidth-optimal way to do an allreduce. To do the allreduce,
 * the nodes involved are arranged in a ring:
 *
 *                   .--0--.
 *                  /       \
 *                 3         1
 *                  \       /
 *                   *--2--*
 *
 *  Each node always sends to the next clockwise node in the ring, and receives
 *  from the previous one.
 *
 *  The allreduce is done in two parts: a scatter-reduce and an allgather. In
 *  the scatter reduce, a reduction is done, so that each node ends up with a
 *  chunk of the final output tensor which has contributions from all other
 *  nodes.  In the allgather, those chunks are distributed among all the nodes,
 *  so that all nodes have the entire output tensor.
 *
 *  Both of these operations are done by dividing the input tensor into N
 *  evenly sized chunks (where N is the number of nodes in the ring).
 *
 *  The scatter-reduce is done in N-1 steps. In the ith step, node j will send
 *  the (j - i)th chunk and receive the (j - i - 1)th chunk, adding it in to
 *  its existing data for that chunk. For example, in the first iteration with
 *  the ring depicted above, you will have the following transfers:
 *
 *      Segment 0:  Node 0 --> Node 1
 *      Segment 1:  Node 1 --> Node 2
 *      Segment 2:  Node 2 --> Node 3
 *      Segment 3:  Node 3 --> Node 0
 *
 *  In the second iteration, you'll have the following transfers:
 *
 *      Segment 0:  Node 1 --> Node 2
 *      Segment 1:  Node 2 --> Node 3
 *      Segment 2:  Node 3 --> Node 0
 *      Segment 3:  Node 0 --> Node 1
 *
 *  After this iteration, Node 2 has 3 of the four contributions to Segment 0.
 *  The last iteration has the following transfers:
 *
 *      Segment 0:  Node 2 --> Node 3
 *      Segment 1:  Node 3 --> Node 0
 *      Segment 2:  Node 0 --> Node 1
 *      Segment 3:  Node 1 --> Node 2
 *
 *  After this iteration, Node 3 has the fully accumulated Segment 0; Node 0
 *  has the fully accumulated Segment 1; and so on. The scatter-reduce is complete.
 *
 *  Next, the allgather distributes these fully accumululated chunks across all nodes.
 *  Communication proceeds in the same ring, once again in N-1 steps. At the ith step,
 *  node j will send chunk (j - i + 1) and receive chunk (j - i). For example, at the
 *  first iteration, the following transfers will occur:
 *
 *      Segment 0:  Node 3 --> Node 0
 *      Segment 1:  Node 0 --> Node 1
 *      Segment 2:  Node 1 --> Node 2
 *      Segment 3:  Node 2 --> Node 3
 *
 * After the first iteration, Node 0 will have a fully accumulated Segment 0
 * (from Node 3) and Segment 1. In the next iteration, Node 0 will send its
 * just-received Segment 0 onward to Node 1, and receive Segment 3 from Node 3.
 * After this has continued for N - 1 iterations, all nodes will have a the fully
 * accumulated tensor.
 *
 * Each node will do (N-1) sends for the scatter-reduce and (N-1) sends for the allgather.
 * Each send will contain K / N bytes, if there are K bytes in the original tensor on every node.
 * Thus, each node sends and receives 2K(N - 1)/N bytes of data, and the performance of the allreduce
 * (assuming no latency in connections) is constrained by the slowest interconnect between the nodes.
 *
 */
void RingAllreduce(float* data, size_t length, float** output_ptr) {
    // Get MPI size and rank.
    int rank;
    int mpi_error = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(mpi_error != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_rank failed with an error");

    int size;
    mpi_error = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(mpi_error != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_size failed with an error");

    // Check that the lengths given to every process are the same.
    std::vector<size_t> lengths = AllgatherInputLengths(size, length);
    for(size_t other_length : lengths) {
        if(length != other_length) {
            throw std::runtime_error("RingAllreduce received different lengths");
        }
    }

    // Partition the elements of the array into N approximately equal-sized
    // chunks, where N is the MPI size.
    const size_t segment_size = length / size;
    std::vector<size_t> segment_sizes(size, segment_size);

    const size_t residual = length % size;
    for (size_t i = 0; i < residual; ++i) {
        segment_sizes[i]++;
    }

    // Compute where each chunk ends.
    std::vector<size_t> segment_ends(size);
    segment_ends[0] = segment_sizes[0];
    for (size_t i = 1; i < segment_ends.size(); ++i) {
        segment_ends[i] = segment_sizes[i] + segment_ends[i - 1];
    }

    // The last segment should end at the very end of the buffer.
    assert(segment_ends[size - 1] == length);

    // Allocate the output buffer.
    float* output = alloc(length);
    *output_ptr =  output;

    // Copy your data to the output buffer to avoid modifying the input buffer.
    copy(output, data, length);

    // Allocate a temporary buffer to store incoming data.
    // We know that segment_sizes[0] is going to be the largest buffer size,
    // because if there are any overflow elements at least one will be added to
    // the first segment.
    float* buffer = alloc(segment_sizes[0]);

    // Receive from your left neighbor with wrap-around.
    const size_t recv_from = (rank - 1 + size) % size;

    // Send to your right neighbor with wrap-around.
    const size_t send_to = (rank + 1) % size;

    MPI_Status recv_status;
    MPI_Request recv_req;
    MPI_Datatype datatype = MPI_FLOAT;

    // Now start ring. At every step, for every rank, we iterate through
    // segments with wraparound and send and recv from our neighbors and reduce
    // locally. At the i'th iteration, sends segment (rank - i) and receives
    // segment (rank - i - 1).
    for (int i = 0; i < size - 1; i++) {
        int recv_chunk = (rank - i - 1 + size) % size;
        int send_chunk = (rank - i + size) % size;
        float* segment_send = &(output[segment_ends[send_chunk] -
                                   segment_sizes[send_chunk]]);

        MPI_Irecv(buffer, segment_sizes[recv_chunk],
                datatype, recv_from, 0, MPI_COMM_WORLD, &recv_req);

        MPI_Send(segment_send, segment_sizes[send_chunk],
                MPI_FLOAT, send_to, 0, MPI_COMM_WORLD);

        float *segment_update = &(output[segment_ends[recv_chunk] -
                                         segment_sizes[recv_chunk]]);

        // Wait for recv to complete before reduction
        MPI_Wait(&recv_req, &recv_status);

        reduce(segment_update, buffer, segment_sizes[recv_chunk]);
    }

    // Now start pipelined ring allgather. At every step, for every rank, we
    // iterate through segments with wraparound and send and recv from our
    // neighbors. At the i'th iteration, rank r, sends segment (rank + 1 - i)
    // and receives segment (rank - i).
    for (size_t i = 0; i < size_t(size - 1); ++i) {
        int send_chunk = (rank - i + 1 + size) % size;
        int recv_chunk = (rank - i + size) % size;
        // Segment to send - at every iteration we send segment (r+1-i)
        float* segment_send = &(output[segment_ends[send_chunk] -
                                       segment_sizes[send_chunk]]);

        // Segment to recv - at every iteration we receive segment (r-i)
        float* segment_recv = &(output[segment_ends[recv_chunk] -
                                       segment_sizes[recv_chunk]]);
        MPI_Sendrecv(segment_send, segment_sizes[send_chunk],
                datatype, send_to, 0, segment_recv,
                segment_sizes[recv_chunk], datatype, recv_from,
                0, MPI_COMM_WORLD, &recv_status);
    }

    // Free temporary memory.
    dealloc(buffer);
}

// The ring allgather. The lengths of the data chunks passed to this function
// may differ across different devices. The output memory will be allocated and
// written into `output`.
//
// For more information on the ring allgather, read the documentation for the
// ring allreduce, which includes a ring allgather as the second stage.
void RingAllgather(float* data, size_t length, float** output_ptr) {
    // Get MPI size and rank.
    int rank;
    int mpi_error = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(mpi_error != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_rank failed with an error");

    int size;
    mpi_error = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(mpi_error != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_size failed with an error");

    // Get the lengths of data provided to every process, so that we know how
    // much memory to allocate for the output buffer.
    std::vector<size_t> segment_sizes = AllgatherInputLengths(size, length);
    size_t total_length = 0;
    for(size_t other_length : segment_sizes) {
        total_length += other_length;
    }

    // Compute where each chunk ends.
    std::vector<size_t> segment_ends(size);
    segment_ends[0] = segment_sizes[0];
    for (size_t i = 1; i < segment_ends.size(); ++i) {
        segment_ends[i] = segment_sizes[i] + segment_ends[i - 1];
    }

    assert(segment_sizes[rank] == length);
    assert(segment_ends[size - 1] == total_length);

    // Allocate the output buffer and copy the input buffer to the right place
    // in the output buffer.
    float* output = alloc(total_length);
    *output_ptr = output;

    copy(output + segment_ends[rank] - segment_sizes[rank],
         data, segment_sizes[rank]);

    // Receive from your left neighbor with wrap-around.
    const size_t recv_from = (rank - 1 + size) % size;

    // Send to your right neighbor with wrap-around.
    const size_t send_to = (rank + 1) % size;

    // What type of data is being sent
    MPI_Datatype datatype = MPI_FLOAT;

    MPI_Status recv_status;

    // Now start pipelined ring allgather. At every step, for every rank, we
    // iterate through segments with wraparound and send and recv from our
    // neighbors. At the i'th iteration, rank r, sends segment (rank + 1 - i)
    // and receives segment (rank - i).
    for (size_t i = 0; i < size_t(size - 1); ++i) {
        int send_chunk = (rank - i + size) % size;
        int recv_chunk = (rank - i - 1 + size) % size;
        // Segment to send - at every iteration we send segment (r+1-i)
        float* segment_send = &(output[segment_ends[send_chunk] -
                                       segment_sizes[send_chunk]]);

        // Segment to recv - at every iteration we receive segment (r-i)
        float* segment_recv = &(output[segment_ends[recv_chunk] -
                                       segment_sizes[recv_chunk]]);
        MPI_Sendrecv(segment_send, segment_sizes[send_chunk],
                datatype, send_to, 0, segment_recv,
                segment_sizes[recv_chunk], datatype, recv_from,
                0, MPI_COMM_WORLD, &recv_status);
    }
}
