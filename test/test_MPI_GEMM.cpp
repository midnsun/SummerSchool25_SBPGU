#include <gtest.h>
#include "MPI_GEMM.h"

TEST(MPI_GEMM, can_MPI_GEMM)
{
    int N = 2000;
    int rank, numtasks;
    double err;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    double* A = nullptr, * B = nullptr, * C = nullptr, * RES = nullptr;
    if (rank == 0) {
        A = new double[N * N] {};
        B = new double[N * N] {};
        C = new double[N * N] {};
        simpleGenerate(N, N, A);
        simpleGenerate(N, N, B);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    else {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    ASSERT_NO_THROW(MPI_GEMM_square(rank, numtasks, N, A, B, C));

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] RES;
}

TEST(MPI_GEMM, MPI_GEMM_is_correct) {
    int N = 2000;
    int rank, numtasks;
    double err;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    double* A = nullptr, * B = nullptr, * C = nullptr, * RES = nullptr;
    if (rank == 0) {
        A = new double[N * N] {};
        B = new double[N * N] {};
        C = new double[N * N] {};
        simpleGenerate(N, N, A);
        simpleGenerate(N, N, B);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    else {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_GEMM_square(rank, numtasks, N, A, B, C);

    if (rank == 0) {
        RES = new double[N * N] {};
        simpleGEMM(N, N, N, A, B, RES);
        err = getErr(N, N, C, RES);
        EXPECT_LT(err, 1.0e-10);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    else {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] RES;
}
