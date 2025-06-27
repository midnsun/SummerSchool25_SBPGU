#include <gtest.h>
#include <mpi.h>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    std::cout << "Hello, Gtest!" << std::endl;
    int i;
    for (i = 0; i < argc + 1; i++) {
        if (argv[i] == nullptr) continue;
        // Printing all the Arguments
        printf("%s ", argv[i]);
    }
    printf("\n");

    return RUN_ALL_TESTS();
}
