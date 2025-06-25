#include <iostream>
#include <omp.h>
#include <cmath>
#include <random>
#include <iomanip>
#include <chrono>
#include <immintrin.h>
#include <mpi.h>
#include <vector>
#include <string>
#include <cstdlib>

const int setwConst = 6;

void simpleGEMM(int m, int n, int k, double* A, double* B, double* C, double alpha, double beta) { // colomn-major
	int i, j, p;
	for (i = 0; i < m; ++i) {
		for (j = 0; j < n; ++j) {
            C[j * m + i] *= beta;
			for (p = 0; p < k; ++p) {
				C[j * m + i] += alpha * A[p * m + i] * B[j * k + p];
			}
		}
	}
}

void simplerGEMM(int m, int n, int k, double* A, double* B, double* C) { // colomn-major
    int i, j, p;
    for (j = 0; j < n; ++j) {
        for (p = 0; p < k; ++p) {
            for (i = 0; i < m; ++i) {
                C[j * m + i] = std::fma(A[p * m + i], B[j * k + p], C[j * m + i]);
            }
        }
    }
}

void simpleGenerate(int m, int n, double* M) {
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_real_distribution<double> coef_gen(-1.0, 1.0);
    double coef;

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            M[j * m + i] = coef_gen(e);
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

double getErr(int m, int n, double* example, double* result) {
    double err = 0.0;
    for (size_t i = 0; i < m * n; ++i) {
        err = std::max(err, std::abs(example[i] - result[i]));
    }
    return err;
}

void testSimpleGemm() {
    int N = 2000;
    int n = 3;
    int m = 4;
    int k = 2;
    n = m = k = N;
    double alpha = 1.0;
    double beta = -1.0;
    double* M1 = new double[m * k] {};
    double* M2 = new double[k * n] {};
    double* M3 = new double[m * n] {};
    std::chrono::steady_clock::time_point start, finish;
    uint64_t time;

    simpleGenerate(m, k, M1);
    simpleGenerate(k, n, M2);
    //    simpleGenerate(m, n, M3);

    //    printAs2D(m, k, M1); std::cout << std::endl; printAs1D(m, k, M1); std::cout << std::endl << std::endl;
    //    printAs2D(k, n, M2); std::cout << std::endl; printAs1D(k, n, M2); std::cout << std::endl << std::endl;
    //    printAs2D(m, n, M3); std::cout << std::endl; printAs1D(m, n, M3); std::cout << std::endl << std::endl;

    start = std::chrono::steady_clock::now();
    simplerGEMM(m, n, k, M1, M2, M3);
    //    simpleGEMM(m, n, k, M1, M2, M3, alpha, beta);
    finish = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds> (finish - start).count();

    //    printAs2D(m, n, M3); std::cout << std::endl; printAs1D(m, n, M3); std::cout << std::endl << std::endl;
    std::cout << "Time is: " << time << std::endl;
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

void MPIGemm(int rank, int numtasks, MPI_Comm grid_comm, int dims[2], int periods[2], int coords[2], int block_size, double* M1, double* M2, double* M3, int q, double *RES) {
    int left, right, up, down;
    MPI_Cart_shift(grid_comm, 1, -coords[0], &right, &left);
    MPI_Sendrecv_replace(M1, block_size * block_size, MPI_DOUBLE,
        left, 0, right, 0, grid_comm, MPI_STATUS_IGNORE);

    MPI_Cart_shift(grid_comm, 0, -coords[1], &down, &up);
    MPI_Sendrecv_replace(M2, block_size * block_size, MPI_DOUBLE,
        up, 0, down, 0, grid_comm, MPI_STATUS_IGNORE);

    // Cannon's cycle
    for (int step = 0; step < q; ++step) {
        simplerGEMM(block_size, block_size, block_size, M1, M2, M3);

        MPI_Cart_shift(grid_comm, 1, -1, &right, &left);
        MPI_Sendrecv_replace(M1, block_size * block_size, MPI_DOUBLE,
            left, 0, right, 0, grid_comm, MPI_STATUS_IGNORE);

        MPI_Cart_shift(grid_comm, 0, -1, &down, &up);
        MPI_Sendrecv_replace(M2, block_size * block_size, MPI_DOUBLE,
            up, 0, down, 0, grid_comm, MPI_STATUS_IGNORE);
    }

    // Gather
    gather_result_blocks(M3, RES, q * block_size, q, block_size, grid_comm);
}

void testMPIGemm(int rank, int numtasks) {
    int q = sqrt(numtasks);
    int blockSize = 1;
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
        simpleGenerate(N, N, A);
        simpleGenerate(N, N, B);
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

    // Distribute matrices to 2D mesh ...
    start = std::chrono::steady_clock::now();
    MPIGemm(rank, numtasks, grid_comm, dims, periods, coords, blockSize, M1, M2, M3, q, RES);
    finish = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds> (finish - start).count();

    std::cout << "Time is: " << time << std::endl;

    delete[] M1;
    delete[] M2;
    delete[] M3;

    if (rank == 0) {
        printAs2D(N, N, A);
        std::cout << std::endl;
        printAs2D(N, N, B);
        std::cout << std::endl;
        C = new double[N * N] {};

        start = std::chrono::steady_clock::now();
        simplerGEMM(N, N, N, A, B, C);
        finish = std::chrono::steady_clock::now();
        time = std::chrono::duration_cast<std::chrono::milliseconds> (finish - start).count();
        std::cout << std::endl << "Time for naive implementation: " << time << " , error is: " << getErr(N, N, C, RES) << std::endl;
        printAs2D(N, N, C);
        std::cout << std::endl;
        printAs2D(N, N, RES);
        std::cout << std::endl;
    }

    delete[] RES;
    delete[] A;
    delete[] B;
    delete[] C;
}

void testMPI(int rank, int numtasks) {
    printf("Hello from process = %d, total number of processes: %d\n", rank, numtasks);
}

int main(int argc, char **argv) {
    int numtasks, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    testMPI(rank, numtasks);
    testMPIGemm(rank, numtasks);

    MPI_Finalize();
	return 0;
}