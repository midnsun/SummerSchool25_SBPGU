#include <mpi.h>
#include <iostream>
#include "MPI_GEMM.h"
#include <chrono>
#include <iomanip>
#include <random>
#include <stdexcept>

const int setwConst = 10;
//bool coutFlag = false;

//void simpleGEMM(int m, int n, int k, double* A, double* B, double* C, double alpha, double beta) { // colomn-major
//    int i, j, p;
//    for (i = 0; i < m; ++i) {
//        for (j = 0; j < n; ++j) {
//            C[j * m + i] *= beta;
//            for (p = 0; p < k; ++p) {
//                C[j * m + i] += alpha * A[p * m + i] * B[j * k + p];
//            }
//        }
//    }
//}

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

void simpleSquareGEMM(int N, double* A, double* B, double* C) {
    for (size_t i = 0; i < N; ++i) {
        double* Btmp = &B[i * N];
        double* Ctmp = &C[i * N];
        for (size_t k = 0; k < N; ++k) {
            double tmp = Btmp[k];
            double* Atmp = &A[k * N];
            for (size_t j = 0; j < N; ++j)
                Ctmp[j] += tmp * Atmp[j];
        }
    }
}

//void determinedGenerate(int m, int n, double* M) {
//    for (size_t i = 0; i < m; ++i) {
//        for (size_t j = 0; j < n; ++j) {
//            M[j * m + i] = j * m + i;
//        }
//    }
//}

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

void gather_result_blocks(double* block, double* RES,
    int N, int q, int block_size, MPI_Comm grid_comm) {
    int rank;
    MPI_Comm_rank(grid_comm, &rank);

    double* gathered = nullptr;
    if (rank == 0) {
        gathered = new double[N * N];
    }

    MPI_Gather(block, block_size * block_size, MPI_DOUBLE,
        gathered, block_size * block_size, MPI_DOUBLE,
        0, grid_comm);

    if (rank == 0) {
        for (int p = 0; p < q * q; ++p) {
            int proc_row = p / q;
            int proc_col = p % q;

            for (int i = 0; i < block_size; ++i)
                for (int j = 0; j < block_size; ++j) {
                    int global_i = proc_row * block_size + i;
                    int global_j = proc_col * block_size + j;
                    RES[global_i * N + global_j] =
                        gathered[p * block_size * block_size + i * block_size + j];
                }
        }
    }

    delete[] gathered;
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

void distribute_matrix(double* full, double* local, int N, int q, int block_size) {
    double* blocks = new double[N * N];

    for (int proc = 0; proc < q * q; ++proc) {
        int proc_row = proc / q;
        int proc_col = proc % q;

        for (int i = 0; i < block_size; ++i)
            for (int j = 0; j < block_size; ++j) {
                int global_i = proc_row * block_size + i;
                int global_j = proc_col * block_size + j;
                blocks[proc * block_size * block_size + i * block_size + j] =
                    full[global_i * N + global_j];
            }
    }

    MPI_Scatter(blocks, block_size * block_size, MPI_DOUBLE,
        local, block_size * block_size, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    delete[] blocks;
}

//void distribute_martix(int M, int N, int linear_proc, double* full, double* local) {
//    std::cout << "DistM0" << std::endl;
//    double* blocks = new double[M * N];
//    int M_block_size = M / linear_proc;
//    int N_block_size = N / linear_proc;
//    int proc_row, proc_col, global_i, global_j;
//
//    for (int proc = 0; proc < linear_proc * linear_proc; ++proc) {
//        proc_row = proc / linear_proc;
//        proc_col = proc % linear_proc;
//
//        for (int i = 0; i < N_block_size; ++i) {
//            global_i = proc_row * M_block_size + i;
//            for (int j = 0; j < M_block_size; ++j) {
//                global_j = proc_col * N_block_size + j;
//                blocks[proc * M_block_size * N_block_size + i * M_block_size + j] = full[global_i * M + global_j];
//            }
//        }
//    }
//    std::cout << "DistM2" << std::endl;
//
//    MPI_Scatter(blocks, M_block_size * N_block_size, MPI_DOUBLE, local, M_block_size * N_block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//
//    std::cout << "DistM1" << std::endl;
//    delete[] blocks;
//}

void distribute_martix(int rank, int M, int N, int linear_proc, double* full, double* local) {
//    std::cout << "DistM0" << std::endl;
    int M_block_size = M / linear_proc;
    int N_block_size = N / linear_proc;
    if (rank == 0) {
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
        //    std::cout << "DistM2" << std::endl;

        MPI_Scatter(blocks, M_block_size * N_block_size, MPI_DOUBLE, local, M_block_size * N_block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //    std::cout << "DistM1" << std::endl;
        delete[] blocks;
        MPI_Barrier(MPI_COMM_WORLD);
    }
    else {
        MPI_Scatter(nullptr, M_block_size * N_block_size, MPI_DOUBLE, local, M_block_size * N_block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void MPIGemm_old(int rank, int numtasks, MPI_Comm grid_comm, int dims[2], int periods[2], int coords[2], int block_size, double* M1, double* M2, double* M3, int q, double* RES) {

    //    if (rank == 1) {
    //        if (coutFlag) std::cout << "rank is: " << rank << std::endl;
    //        if (coutFlag) printAs2D(block_size, block_size, M1);
    //        if (coutFlag) std::cout << std::endl;
    //        if (coutFlag) printAs2D(block_size, block_size, M2);
    //        if (coutFlag) std::cout << std::endl;
    //    }

    int left, right, up, down;

    MPI_Cart_shift(grid_comm, 1, -coords[0], &right, &left);
    MPI_Sendrecv_replace(M2, block_size * block_size, MPI_DOUBLE,
        left, 0, right, 0, grid_comm, MPI_STATUS_IGNORE);

    MPI_Cart_shift(grid_comm, 0, -coords[1], &down, &up);
    MPI_Sendrecv_replace(M1, block_size * block_size, MPI_DOUBLE,
        up, 0, down, 0, grid_comm, MPI_STATUS_IGNORE);

    //    if (rank == 1) {
    //        if (coutFlag) std::cout << "rank is: " << rank << std::endl;
    //        if (coutFlag) printAs2D(block_size, block_size, M1);
    //        if (coutFlag) std::cout << std::endl;
    //        if (coutFlag) printAs2D(block_size, block_size, M2);
    //        if (coutFlag) std::cout << std::endl;
    //    }

    MPI_Cart_shift(grid_comm, 1, -1, &right, &left);
    MPI_Cart_shift(grid_comm, 0, -1, &down, &up);

    // Cannon's cycle
    for (int step = 0; step < q; ++step) {
        //        if (rank == 1) {
        //            if (coutFlag) std::cout << "rank is: " << rank << std::endl;
        //            if (coutFlag) printAs2D(block_size, block_size, M3);
        //            if (coutFlag) std::cout << std::endl;
        //        }

        simple_GEMM(block_size, block_size, block_size, M1, M2, M3);

        //        if (rank == 1) {
        //            if (coutFlag) std::cout << "rank is: " << rank << std::endl;
        //            if (coutFlag) printAs2D(block_size, block_size, M3);
        //            if (coutFlag) std::cout << std::endl;
        //        }

        MPI_Sendrecv_replace(M2, block_size * block_size, MPI_DOUBLE,
            left, 0, right, 0, grid_comm, MPI_STATUS_IGNORE);

        MPI_Sendrecv_replace(M1, block_size * block_size, MPI_DOUBLE,
            up, 0, down, 0, grid_comm, MPI_STATUS_IGNORE);
    }

    // Gather
    gather_result_blocks(M3, RES, q * block_size, q, block_size, grid_comm);
}

void MPI_GEMM_square(int rank, int numtasks, int N, double* A, double* B, double* C) {
    // Getting sizes
    double* M1, * M2, * M3;
    int q = sqrt(numtasks);
    int blockSize = N / q;
    if (q * q != numtasks) {
        if (rank == 0)
            throw std::runtime_error("Number of processes must be a perfect square");
        if (rank == 0 && q * blockSize != N) 
            throw std::runtime_error("Number of processes must be matrice's divider");
        return;
    }
    
    // Create 2D grid
    MPI_Comm grid_comm;
    int dims[2] = { q, q }, periods[2] = { 1, 1 }, coords[2];
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);
    MPI_Cart_coords(grid_comm, rank, 2, coords);

    // Memory allocation and distribution
    M1 = new double[blockSize * blockSize] {};
    M2 = new double[blockSize * blockSize] {};
    M3 = new double[blockSize * blockSize] {};
    distribute_martix(rank, N, N, q, A, M1);
    distribute_martix(rank, N, N, q, B, M2);
//    if (rank == 0) {
////        distribute_matrix(A, M1, N, q, blockSize);
////        distribute_matrix(B, M2, N, q, blockSize);
//        MPI_Barrier(MPI_COMM_WORLD);
//    }
//    else {
//        MPI_Scatter(nullptr, blockSize * blockSize, MPI_DOUBLE,
//            M1, blockSize * blockSize, MPI_DOUBLE,
//            0, MPI_COMM_WORLD);
//        MPI_Scatter(nullptr, blockSize * blockSize, MPI_DOUBLE,
//            M2, blockSize * blockSize, MPI_DOUBLE,
//            0, MPI_COMM_WORLD);
//        MPI_Barrier(MPI_COMM_WORLD);
//    }

    // 2D Grid preparations
    int left, right, up, down;
    MPI_Cart_shift(grid_comm, 1, -coords[0], &right, &left);
    MPI_Sendrecv_replace(M2, blockSize * blockSize, MPI_DOUBLE,
        left, 0, right, 0, grid_comm, MPI_STATUS_IGNORE);
    MPI_Cart_shift(grid_comm, 0, -coords[1], &down, &up);
    MPI_Sendrecv_replace(M1, blockSize * blockSize, MPI_DOUBLE,
        up, 0, down, 0, grid_comm, MPI_STATUS_IGNORE);
    MPI_Cart_shift(grid_comm, 1, -1, &right, &left);
    MPI_Cart_shift(grid_comm, 0, -1, &down, &up);

    // Cannon's cycle
    for (int step = 0; step < q; ++step) {
//        simpleGEMM(blockSize, blockSize, blockSize, M1, M2, M3);
        simpleSquareGEMM(blockSize, M1, M2, M3);
        MPI_Sendrecv_replace(M2, blockSize * blockSize, MPI_DOUBLE,
            left, 0, right, 0, grid_comm, MPI_STATUS_IGNORE);

        MPI_Sendrecv_replace(M1, blockSize * blockSize, MPI_DOUBLE,
            up, 0, down, 0, grid_comm, MPI_STATUS_IGNORE);
    }

    // Gather
//    gather_result_blocks(M3, C, q * blockSize, q, blockSize, grid_comm);
    gather_blocks(N, N, q, M3, C, grid_comm);

    // Free memory
    delete[] M1;
    delete[] M2;
    delete[] M3;
}

void MPI_GEMM(int rank, int numtasks, int M, int N, int K, double* A, double* B, double* C) {
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

//    std::cout << rank << "A" << std::endl;

    // Memory allocation and distribution
    M1 = new double[M_blockSize * K_blockSize] {};
    M2 = new double[K_blockSize * N_blockSize] {};
    M3 = new double[M_blockSize * N_blockSize] {};
    distribute_martix(rank, M, K, q, A, M1);
    distribute_martix(rank, K, N, q, B, M2);
//    if (rank == 0) {
////        std::cout << rank << "AA" << std::endl;
//        distribute_martix(M, K, q, A, M1);
//        distribute_martix(K, N, q, B, M2);
////       std::cout << rank << "AB" << std::endl;
//        MPI_Barrier(MPI_COMM_WORLD);
//    }
//    else {
////        std::cout << rank << "BA" << std::endl;
//        MPI_Scatter(nullptr, M_blockSize * K_blockSize, MPI_DOUBLE, M1, M_blockSize * K_blockSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//        MPI_Scatter(nullptr, K_blockSize * N_blockSize, MPI_DOUBLE, M2, K_blockSize * N_blockSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
////        std::cout << rank << "BB" << std::endl;
//        MPI_Barrier(MPI_COMM_WORLD);
//    }

//    std::cout << rank << "B" << std::endl;

    // 2D Grid preparations
    int left, right, up, down;
    MPI_Cart_shift(grid_comm, 1, -coords[0], &right, &left);
    MPI_Sendrecv_replace(M2, K_blockSize * N_blockSize, MPI_DOUBLE, left, 0, right, 0, grid_comm, MPI_STATUS_IGNORE);
    MPI_Cart_shift(grid_comm, 0, -coords[1], &down, &up);
    MPI_Sendrecv_replace(M1, M_blockSize * K_blockSize, MPI_DOUBLE, up, 0, down, 0, grid_comm, MPI_STATUS_IGNORE);
    MPI_Cart_shift(grid_comm, 1, -1, &right, &left);
    MPI_Cart_shift(grid_comm, 0, -1, &down, &up);

//    std::cout << rank << "C" << std::endl;

    // Cannon's cycle
    for (int step = 0; step < q; ++step) {
        simple_GEMM(M_blockSize, N_blockSize, K_blockSize, M1, M2, M3);
        MPI_Sendrecv_replace(M2, K_blockSize * N_blockSize, MPI_DOUBLE, left, 0, right, 0, grid_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(M1, M_blockSize * K_blockSize, MPI_DOUBLE, up, 0, down, 0, grid_comm, MPI_STATUS_IGNORE);
    }

//    std::cout << rank << "D" << std::endl;

    // Gather
    gather_blocks(M, N, q, M3, C, grid_comm);

//    std::cout << rank << "E" << std::endl;

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

//    std::cout << rank << "befB" << std::endl;
//    delete[] B;
//    std::cout << rank << "afB" << std::endl;
//    return;

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
//    MPI_GEMM_square(rank, numtasks, N, A, B, C);
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

//    std::cout << rank << "J" << std::endl;

    delete[] A;
//    std::cout << rank << "MA" << std::endl;
    delete[] B;
//    std::cout << rank << "MB" << std::endl;
    delete[] C;
//    std::cout << rank << "MC" << std::endl;
    delete[] RES;

//    std::cout << rank << "H" << std::endl;
}

void testMPIGemm_old(int argc, char** argv) {
    int numtasks, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    int q = sqrt(numtasks);
    int blockSize = 200;
    int N = q * blockSize;
    int n = 3;
    int m = 4;
    int k = 2;
    n = m = k = N;
    double alpha = 1.0;
    double beta = -1.0;
    std::chrono::steady_clock::time_point start, finish;
    uint64_t time;
    double* M1, * M2, * M3, * RES = nullptr, * A = nullptr, * B = nullptr, * C = nullptr;

    if (q * q != numtasks) {
        if (rank == 0)
            std::cerr << "Number of processes must be a perfect square\n";
        MPI_Finalize();
        return;
    }

    // Create 2D grid
    MPI_Comm grid_comm;
    int dims[2] = { q, q }, periods[2] = { 1, 1 }, coords[2];
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);
    MPI_Cart_coords(grid_comm, rank, 2, coords);

    M1 = new double[blockSize * blockSize] {};
    M2 = new double[blockSize * blockSize] {};
    M3 = new double[blockSize * blockSize] {};
    if (rank == 0) {
        RES = new double[N * N] {};
        A = new double[N * N];
        B = new double[N * N];
        simple_generate(N, N, A);
        simple_generate(N, N, B);
        distribute_matrix(A, M1, N, q, blockSize);
        distribute_matrix(B, M2, N, q, blockSize);
    }
    else {
        MPI_Scatter(nullptr, blockSize * blockSize, MPI_DOUBLE,
            M1, blockSize * blockSize, MPI_DOUBLE,
            0, MPI_COMM_WORLD);
        MPI_Scatter(nullptr, blockSize * blockSize, MPI_DOUBLE,
            M2, blockSize * blockSize, MPI_DOUBLE,
            0, MPI_COMM_WORLD);
    }

//    if (coutFlag) std::cout << "rank: " << rank << std::endl;
//    if (coutFlag) printAs2D(blockSize, blockSize, M1);
//    if (coutFlag) std::cout << std::endl;
//    if (coutFlag) printAs2D(blockSize, blockSize, M2);
//    if (coutFlag) std::cout << std::endl;

    // Distribute matrices to 2D mesh ...
    start = std::chrono::steady_clock::now();
    MPIGemm_old(rank, numtasks, grid_comm, dims, periods, coords, blockSize, M1, M2, M3, q, RES);
    finish = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds> (finish - start).count();
    // double MPI_Wtime() ...

    std::cout << "Time is: " << time << std::endl;

    delete[] M1;
    delete[] M2;
    delete[] M3;

    if (rank == 0) {
//        if (coutFlag) printAs2D(N, N, A);
//        if (coutFlag) std::cout << std::endl;
//        if (coutFlag) printAs2D(N, N, B);
//        if (coutFlag) std::cout << std::endl;
        C = new double[N * N] {};

        start = std::chrono::steady_clock::now();
        simple_GEMM(N, N, N, A, B, C);
        finish = std::chrono::steady_clock::now();
        time = std::chrono::duration_cast<std::chrono::milliseconds> (finish - start).count();
        std::cout << std::endl << "Time for naive implementation: " << time << " , error is: " << get_err(N, N, C, RES) << std::endl;
//        if (coutFlag) printAs2D(N, N, C);
//        if (coutFlag) std::cout << std::endl;
//        if (coutFlag) printAs2D(N, N, RES);
//        if (coutFlag) std::cout << std::endl;
    }

    delete[] RES;
    delete[] A;
    delete[] B;
    delete[] C;

    MPI_Finalize();
}