#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include "MPI_GEMM.h"

void testMPI(int rank, int numtasks) {
    printf("Hello from process = %d, total number of processes: %d\n", rank, numtasks);
}

int main(int argc, char **argv) {

//    testMPI(rank, numtasks);
    testMPIGemm(argc, argv);

	return 0;
}