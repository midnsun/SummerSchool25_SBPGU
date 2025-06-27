#include <gtest.h>
#include "MPI_GEMM.h"

TEST(MPI_GEMM, MPI_GEMM_TEST)
{
	int argc = 1;
	char** argv = new char*[argc];
	argv[0] = const_cast<char*>(".\build\bin\test_MPI_GEMM.exe\0");

	ASSERT_NO_THROW(testMPIGemm(argc, argv));

	delete[] argv[0];
	delete[] argv;
}
