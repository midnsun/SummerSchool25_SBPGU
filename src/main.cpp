#include <iostream>
#include <omp.h>
#include <cmath>
#include <random>
#include <iomanip>
#include <chrono>
#include <immintrin.h>
#include <mpi.h>

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
//#pragma omp parallel for
        for (p = 0; p < k; ++p) {
            for (i = 0; i < m; ++i) {
//                C[j * m + i] = C[j * m + i] + A[p * m + i] * B[j * k + p];
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

void testGemm() {
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

void testMPI(int* argc, char **argv) {
    int numtasks, rank;
    MPI_Init(argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

//    printf("Hello from process = %d, total number of processes: %d\n", rank, numtasks);

    MPI_Finalize();
}

int main(int* argc, char **argv) {
    
    testMPI(argc, argv);

	return 0;
}