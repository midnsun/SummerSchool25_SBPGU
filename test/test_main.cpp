#include <gtest.h>
#include <mpi.h>

void testMPI(int rank, int numtasks) {
    printf("Hello from process = %d, total number of processes: %d\n", rank, numtasks);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int i;
    for (i = 0; i < argc + 1; i++) {
        if (argv[i] == nullptr) continue;
        // Printing all the Arguments
        printf("%s | ", argv[i]);
    }
    printf("\n");

    int numtasks, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    testMPI(rank, numtasks);

    MPI_Finalize();

    return RUN_ALL_TESTS();
}
