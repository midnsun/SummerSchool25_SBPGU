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
//    testMPIGemm_old(argc, argv);
    MPI_Finalize();

	//double* A, * B, * C;
	//int M = 2, N = 3, K = 4;
	//A = new double[M * K] {};
	//B = new double[K * N] {};
	//C = new double[M * N] {};
	//simpleGenerate(M, K, A);
	//simpleGenerate(K, N, B);

	//printAs2D(M, K, A);
	//std::cout << std::endl;
	//printAs2D(K, N, B);
	//std::cout << std::endl;
	//simpleGEMM(M, N, K, A, B, C);
	//printAs2D(M, N, C);
	//std::cout << std::endl;

	return 0;
}