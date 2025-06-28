#include <gtest.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int numtasks, rank;
    MPI_Init(&argc, &argv);

    ::testing::InitGoogleTest(&argc, argv);
    int retVal = RUN_ALL_TESTS();

    MPI_Finalize();
    return retVal;
}
