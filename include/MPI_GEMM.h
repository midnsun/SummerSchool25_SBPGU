#include <mpi.h>

void simple_GEMM(int M, int N, int K, double* A, double* B, double* C);
void simple_generate(int M, int N, double* A);
void printAs2D(int m, int n, double* M);
void printAs1D(int m, int n, double* M);
double get_err(int M, int N, double* example, double* result);
void gather_blocks(int M, int N, int linear_proc, double* block, double* RES, MPI_Comm grid_comm);
void distribute_martix(int rank, int M, int N, int linear_proc, double* full, double* local);
void MPI_GEMM(int rank, int numtasks, int M, int N, int K, double* A, double* B, double* C);
void testMPIGemm(int rank, int numtasks);