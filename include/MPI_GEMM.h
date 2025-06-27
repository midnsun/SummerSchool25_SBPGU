#include <mpi.h>

void simpleGEMM(int m, int n, int k, double* A, double* B, double* C, double alpha, double beta);

void simplerGEMM(int m, int n, int k, double* A, double* B, double* C);

void determinedGenerate(int m, int n, double* M);

void simpleGenerate(int m, int n, double* M);

void printAs2D(int m, int n, double* M);

void printAs1D(int m, int n, double* M);

double getErr(int m, int n, double* example, double* result);

void testSimpleGemm();

void gather_result_blocks(double* block, double* RES,
    int N, int q, int block_size, MPI_Comm grid_comm);

void distribute_matrix(double* full, double* local, int N, int q, int block_size);

void MPIGemm(int rank, int numtasks, MPI_Comm grid_comm, int dims[2], int periods[2], int coords[2], int block_size, double* M1, double* M2, double* M3, int q, double* RES);

void testMPIGemm(int argc, char** argv);