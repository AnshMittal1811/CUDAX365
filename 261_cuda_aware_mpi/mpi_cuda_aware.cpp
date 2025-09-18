#include <mpi.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int count = 1024;
    float *d_buf = nullptr;
    cudaMalloc(&d_buf, count * sizeof(float));

    if (rank == 0) {
        cudaMemset(d_buf, 0, count * sizeof(float));
        MPI_Send(d_buf, count, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Recv(d_buf, count, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    cudaFree(d_buf);
    MPI_Finalize();
    std::printf("mpi cuda aware done on rank %d\n", rank);
    return 0;
}
