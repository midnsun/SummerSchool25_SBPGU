#include <mpi.h>

//void simpleGEMM(int m, int n, int k, double* A, double* B, double* C, double alpha, double beta);

void simple_GEMM(int M, int N, int K, double* A, double* B, double* C);

void simpleSquareGEMM(int N, double* A, double* B, double* C);

//void determinedGenerate(int m, int n, double* M);

void simple_generate(int M, int N, double* A);

void printAs2D(int m, int n, double* M);

void printAs1D(int m, int n, double* M);

double get_err(int M, int N, double* example, double* result);

void gather_result_blocks(double* block, double* RES, int N, int q, int block_size, MPI_Comm grid_comm);

void gather_blocks(int M, int N, int linear_proc, double* block, double* RES, MPI_Comm grid_comm);

void distribute_matrix(double* full, double* local, int N, int q, int block_size);

void distribute_martix(int rank, int M, int N, int linear_proc, double* full, double* local);

void MPIGemm_old(int rank, int numtasks, MPI_Comm grid_comm, int dims[2], int periods[2], int coords[2], int block_size, double* M1, double* M2, double* M3, int q, double* RES);

void MPI_GEMM_square(int rank, int numtasks, int N, double* A, double* B, double* C);

void MPI_GEMM(int rank, int numtasks, int M, int N, int K, double* A, double* B, double* C);

void testMPIGemm_old(int argc, char** argv);

void testMPIGemm(int rank, int numtasks);