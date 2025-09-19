#include <mpi.h>

#include <cstdio>
#include <vector>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nx = 128;
    int ny = 128;
    int local_nx = nx / size;
    std::vector<float> local(ny * local_nx, 0.0f);

    for (int step = 0; step < 10; ++step) {
        if (rank > 0) {
            MPI_Sendrecv(local.data(), ny, MPI_FLOAT, rank - 1, 0,
                         local.data(), ny, MPI_FLOAT, rank - 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank + 1 < size) {
            MPI_Sendrecv(local.data() + (local_nx - 1), ny, MPI_FLOAT, rank + 1, 1,
                         local.data() + (local_nx - 1), ny, MPI_FLOAT, rank + 1, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    MPI_Finalize();
    std::printf("mpi domain split done on rank %d\n", rank);
    return 0;
}
