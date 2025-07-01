#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include "MPI_GEMM.h"

void testMPI(int rank, int numtasks) {
    printf("Hello from process = %d, total number of processes: %d\n", rank, numtasks);
}

int main(int argc, char **argv) {
    int rank, numtasks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    testMPIGemm(rank, numtasks);
    MPI_Finalize();
	return 0;
}