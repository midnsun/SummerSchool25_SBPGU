#include <gtest.h>
#include <cmath>
#include "MPI_GEMM.h"

TEST(MPI_GEMM, can_simple_generate) {
    int M = 10;
    int N = 5;
    double* A = new double[M * N] {};
    ASSERT_NO_THROW(simple_generate(M, N, A));
}

TEST(MPI_GEMM, can_get_err) {
    int M = 10;
    int N = 5;
    double* A = new double[M * N] {};
    double* B = new double[M * N] {};
    ASSERT_NO_THROW(get_err(M, N, A, B));
}

TEST(MPI_GEMM, get_err_is_correct) {
    int M = 2;
    int N = 3;
    double* A = new double[M * N] {};
    double* B = new double[M * N] {};
    for (size_t i = 0; i < M * N; ++i) {
        A[i] = i;
        B[i] = 2 * i;
    }
    EXPECT_DOUBLE_EQ(5.0, get_err(M, N, A, B));
}

TEST(MPI_GEMM, can_distribute_matrix) { // supposed 4 tasks
    int rank, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    int M = 4;
    int N = 6;
    double* A = nullptr;
    if (rank == 0) {
        A = new double[M * N] {};
        for (size_t i = 0; i < M * N; ++i) A[i] = i;
        MPI_Barrier(MPI_COMM_WORLD);
    }
    else {
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double* B = new double[M * N / numtasks] {};
    ASSERT_NO_THROW(distribute_martix(rank, M, N, std::sqrt(numtasks), A, B));

    delete[] A;
    delete[] B;
}

TEST(MPI_GEMM, distribute_matrix_is_correct) { // supposed 4 tasks
    int rank, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    int M = 4;
    int N = 6;
    double* A = nullptr;
    if (rank == 0) {
        A = new double[M * N] {};
        for (size_t i = 0; i < M * N; ++i) A[i] = i;
        MPI_Barrier(MPI_COMM_WORLD);
    }
    else {
        MPI_Barrier(MPI_COMM_WORLD);
    }
    int q = std::sqrt(numtasks);
    double* B = new double[M * N / numtasks] {};
    distribute_martix(rank, M, N, q, A, B);

    if (rank == 0) {
        EXPECT_DOUBLE_EQ(0.0, B[0]);
        EXPECT_DOUBLE_EQ(1.0, B[1]);
        EXPECT_DOUBLE_EQ(4.0, B[2]);
        EXPECT_DOUBLE_EQ(5.0, B[3]);
        EXPECT_DOUBLE_EQ(8.0, B[4]);
        EXPECT_DOUBLE_EQ(9.0, B[5]);
    }
    if (rank == 1) {
        EXPECT_DOUBLE_EQ(2.0, B[0]);
        EXPECT_DOUBLE_EQ(3.0, B[1]);
        EXPECT_DOUBLE_EQ(6.0, B[2]);
        EXPECT_DOUBLE_EQ(7.0, B[3]);
        EXPECT_DOUBLE_EQ(10.0, B[4]);
        EXPECT_DOUBLE_EQ(11.0, B[5]);
    }
    if (rank == 2) {
        EXPECT_DOUBLE_EQ(12.0, B[0]);
        EXPECT_DOUBLE_EQ(13.0, B[1]);
        EXPECT_DOUBLE_EQ(16.0, B[2]);
        EXPECT_DOUBLE_EQ(17.0, B[3]);
        EXPECT_DOUBLE_EQ(20.0, B[4]);
        EXPECT_DOUBLE_EQ(21.0, B[5]);
    }
    if (rank == 3) {
        EXPECT_DOUBLE_EQ(14.0, B[0]);
        EXPECT_DOUBLE_EQ(15.0, B[1]);
        EXPECT_DOUBLE_EQ(18.0, B[2]);
        EXPECT_DOUBLE_EQ(19.0, B[3]);
        EXPECT_DOUBLE_EQ(22.0, B[4]);
        EXPECT_DOUBLE_EQ(23.0, B[5]);
    }

    delete[] A;
    delete[] B;
}

TEST(MPI_GEMM, can_gather_matrix) { // 4 tasks
    int rank, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    int M = 4;
    int N = 6;
    double* A = nullptr;
    if (rank == 0) {
        A = new double[M * N] {};
        MPI_Barrier(MPI_COMM_WORLD);
    }
    else {
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double* B = new double[M * N / numtasks] {};
    ASSERT_NO_THROW(gather_blocks(M, N, std::sqrt(numtasks), B, A, MPI_COMM_WORLD));
}

TEST(MPI_GEMM, gather_matrix_is_correct) { // 4 tasks
    int rank, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    int M = 4;
    int N = 6;
    double* A = nullptr;
    if (rank == 0) {
        A = new double[M * N] {};
        MPI_Barrier(MPI_COMM_WORLD);
    }
    else {
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double* B = new double[M * N / numtasks];
    if (rank == 0) {
        B[0] = 0.0;
        B[1] = 1.0;
        B[2] = 4.0;
        B[3] = 5.0;
        B[4] = 8.0;
        B[5] = 9.0;
    }
    if (rank == 1) {
        B[0] = 2.0;
        B[1] = 3.0;
        B[2] = 6.0;
        B[3] = 7.0;
        B[4] = 10.0;
        B[5] = 11.0;
    }
    if (rank == 2) {
        B[0] = 12.0;
        B[1] = 13.0;
        B[2] = 16.0;
        B[3] = 17.0;
        B[4] = 20.0;
        B[5] = 21.0;
    }
    if (rank == 3) {
        B[0] = 14.0;
        B[1] = 15.0;
        B[2] = 18.0;
        B[3] = 19.0;
        B[4] = 22.0;
        B[5] = 23.0;
    }
    gather_blocks(M, N, std::sqrt(numtasks), B, A, MPI_COMM_WORLD);

    if (rank == 0) for (size_t i = 0; i < M * N; ++i) EXPECT_EQ(i, A[i]);

    delete[] A;
    delete[] B;
}

TEST(MPI_GEMM, can_MPI_GEMM)
{
    int N = 2000;
    int M = 1500;
    int K = 1000;
    int rank, numtasks;
    double err;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    double* A = nullptr, * B = nullptr, * C = nullptr, * RES = nullptr;
    if (rank == 0) {
        A = new double[M * K] {};
        B = new double[K * N] {};
        C = new double[M * N] {};
        simple_generate(M, K, A);
        simple_generate(K, N, B);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    else {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    ASSERT_NO_THROW(MPI_GEMM(rank, numtasks, M, N, K, A, B, C));

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] RES;
}

TEST(MPI_GEMM, MPI_GEMM_is_correct) {
    int N = 2000;
    int M = 1500;
    int K = 1000;
    int rank, numtasks;
    double err;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    double* A = nullptr, * B = nullptr, * C = nullptr, * RES = nullptr;
    if (rank == 0) {
        A = new double[M * K] {};
        B = new double[K * N] {};
        C = new double[M * N] {};
        simple_generate(M, K, A);
        simple_generate(K, N, B);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    else {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_GEMM(rank, numtasks, M, N, K, A, B, C);

    if (rank == 0) {
        RES = new double[M * N] {};
        simple_GEMM(M, N, K, A, B, RES);
        err = get_err(M, N, C, RES);
        EXPECT_LT(err, 1.0e-12);
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
