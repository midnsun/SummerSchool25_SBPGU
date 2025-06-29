#include <mpi.h>
#include <iostream>
#include "MPI_GEMM.h"
#include <chrono>
#include <iomanip>
#include <random>
#include <stdexcept>

const int setwConst = 10;

void simple_GEMM(int M, int N, int K, double* A, double* B, double* C) { // colomn-major
    int i, j, p;
    double* Atmp, * Btmp, * Ctmp, tmp;
    for (j = 0; j < N; ++j) {
        Btmp = B + j * K;
        Ctmp = C + j * M;
        for (p = 0; p < K; ++p) {
            tmp = Btmp[p];
            Atmp = A + p * M;
            for (i = 0; i < M; ++i)
                Ctmp[i] += tmp * Atmp[i];
        }
    }
}

void simple_generate(int M, int N, double* A) {
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_real_distribution<double> coef_gen(-1.0, 1.0);
    double coef;

    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < M; ++i) {
            A[j * M + i] = coef_gen(e);
        }
    }
}

void printAs2D(int m, int n, double* M) {
    std::cout << std::fixed << std::setprecision(2);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            std::cout << std::setw(setwConst) << M[j * m + i];
        }
        std::cout << std::endl;
    }
}

void printAs1D(int m, int n, double* M) {
    std::cout << std::fixed << std::setprecision(2);
    for (size_t i = 0; i < n * m; ++i) {
        std::cout << std::setw(setwConst) << M[i];
    }
    std::cout << std::endl;
}

double get_err(int M, int N, double* example, double* result) {
    double err = 0.0;
    for (size_t i = 0; i < M * N; ++i) {
        err = std::max(err, std::abs(example[i] - result[i]));
    }
    return err;
}

void distribute_martix(int rank, int M, int N, int linear_proc, double* full, double* local) { // scatter matrix from node 0 to all nodes
    int M_block_size = M / linear_proc;
    int N_block_size = N / linear_proc;
    if (rank == 0) {
        // move all blocks from given matrix to 1 array named 'blocks'
        double* blocks = new double[M * N];
        int proc_row, proc_col, global_i, global_j;

        for (int proc = 0; proc < linear_proc * linear_proc; ++proc) {
            proc_row = proc % linear_proc;
            proc_col = proc / linear_proc;

            for (int i = 0; i < M_block_size; ++i) {
                global_i = proc_row * M_block_size + i;
                for (int j = 0; j < N_block_size; ++j) {
                    global_j = proc_col * N_block_size + j;
                    blocks[proc * M_block_size * N_block_size + j * M_block_size + i] = full[global_j * M + global_i];
                }
            }
        }

        // scatter blocks array on node 0
        MPI_Scatter(blocks, M_block_size * N_block_size, MPI_DOUBLE, local, M_block_size * N_block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        delete[] blocks;
        MPI_Barrier(MPI_COMM_WORLD);
    }
    else {
        // scatter blocks array on other nodes
        MPI_Scatter(nullptr, M_block_size * N_block_size, MPI_DOUBLE, local, M_block_size * N_block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void gather_blocks(int M, int N, int linear_proc, double* block, double* RES, MPI_Comm grid_comm) {
    int rank;
    MPI_Comm_rank(grid_comm, &rank);

    double* gathered = nullptr;
    if (rank == 0) {
        gathered = new double[M * N];
    }
    int M_block_size = M / linear_proc;
    int N_block_size = N / linear_proc;

    MPI_Gather(block, M_block_size * N_block_size, MPI_DOUBLE, gathered, M_block_size * N_block_size, MPI_DOUBLE, 0, grid_comm);
    int proc_row, proc_col, global_i, global_j;

    if (rank == 0) {
        for (int proc = 0; proc < linear_proc * linear_proc; ++proc) {
            proc_row = proc % linear_proc;
            proc_col = proc / linear_proc;

            for (int i = 0; i < M_block_size; ++i) {
                global_i = proc_row * M_block_size + i;
                for (int j = 0; j < N_block_size; ++j) {
                    global_j = proc_col * N_block_size + j;
                    RES[global_j * M + global_i] = gathered[proc * M_block_size * N_block_size + j * M_block_size + i];
                }
            }
        }
    }
    delete[] gathered;
}

void MPI_GEMM(int rank, int numtasks, int M, int N, int K, double* A, double* B, double* C) { 
    // I'm sorry for such a simple implementation, I'm having 2 exams before the 3rd of July and I have to be ready for them :(
    // 
    // Getting sizes
    double* M1, * M2, * M3;
    int q = sqrt(numtasks);
    int M_blockSize = M / q;
    int N_blockSize = N / q;
    int K_blockSize = K / q;
    if (q * q != numtasks)
        throw std::runtime_error("Number of processes must be a perfect square");
    if (M % q + N % q + K % q != 0)
        throw std::runtime_error("Number of processes must be matrice's linear size divider");

    // Create 2D grid
    MPI_Comm grid_comm;
    int dims[2] = { q, q }, periods[2] = { 1, 1 }, coords[2];
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);
    MPI_Cart_coords(grid_comm, rank, 2, coords);

    // Memory allocation and distribution
    M1 = new double[M_blockSize * K_blockSize] {};
    M2 = new double[K_blockSize * N_blockSize] {};
    M3 = new double[M_blockSize * N_blockSize] {};
    distribute_martix(rank, M, K, q, A, M1);
    distribute_martix(rank, K, N, q, B, M2);

    // 2D Grid preparations
    int left, right, up, down;
    MPI_Cart_shift(grid_comm, 1, -coords[0], &right, &left);
    MPI_Sendrecv_replace(M2, K_blockSize * N_blockSize, MPI_DOUBLE, left, 0, right, 0, grid_comm, MPI_STATUS_IGNORE);
    MPI_Cart_shift(grid_comm, 0, -coords[1], &down, &up);
    MPI_Sendrecv_replace(M1, M_blockSize * K_blockSize, MPI_DOUBLE, up, 0, down, 0, grid_comm, MPI_STATUS_IGNORE);
    MPI_Cart_shift(grid_comm, 1, -1, &right, &left);
    MPI_Cart_shift(grid_comm, 0, -1, &down, &up);

    // Cannon's cycle
    for (int step = 0; step < q; ++step) {
        simple_GEMM(M_blockSize, N_blockSize, K_blockSize, M1, M2, M3);
        MPI_Sendrecv_replace(M2, K_blockSize * N_blockSize, MPI_DOUBLE, left, 0, right, 0, grid_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(M1, M_blockSize * K_blockSize, MPI_DOUBLE, up, 0, down, 0, grid_comm, MPI_STATUS_IGNORE);
    }

    // Gather
    gather_blocks(M, N, q, M3, C, grid_comm);

    // Free memory
    delete[] M1;
    delete[] M2;
    delete[] M3;
}

void testMPIGemm(int rank, int numtasks) {
    int M = 1000;
    int N = 1500;
    int K = 2000;

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

    std::chrono::steady_clock::time_point start, finish; //
    uint64_t time; //
    start = std::chrono::steady_clock::now(); //
    MPI_GEMM(rank, numtasks, M, N, K, A, B, C);
    finish = std::chrono::steady_clock::now(); //
    time = std::chrono::duration_cast<std::chrono::milliseconds> (finish - start).count(); //
    std::cout << "Time is: " << time << std::endl; //

    if (rank == 0) {
        RES = new double[M * N] {};
        start = std::chrono::steady_clock::now(); //
        simple_GEMM(M, N, K, A, B, RES);
        finish = std::chrono::steady_clock::now(); //
        time = std::chrono::duration_cast<std::chrono::milliseconds> (finish - start).count(); //
        std::cout << std::endl << "Time for naive implementation: " << time << " , error is: " << get_err(M, N, C, RES) << std::endl; //
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