# SummerSchool25_SBPGU
Repository of the testing project of the Summer School of St. Petersburg State University 2025:
'Implementation of a distributed algorithm for matrix multiplication using the MPI library'
Completed by a 2nd year student of UNN, ITMM, Maksim Zagryadskov

## How to build this project:

**On windows, with Visual Studio 17 2022 and CMake**
1. Install and setup [MS-MPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi-release-notes) for Visual Studio
2. Clone this repository with `git clone https://github.com/midnsun/SummerSchool25_SBPGU`
3. Enter the directory with this project and run commands below:
    ```
    mkdir build
    cmake -B build
    cmake --build build
    ```
To run tests, type `mpiexec -n 4 .\build\bin\test_MPI_GEMM.exe`  
To run sample, type `mpiexec -n 4 .\build\bin\sample.exe`


**On linux, with CMake**
1. Install mpi with `sudo apt install mpich`
2. Clone this repository with `git clone https://github.com/midnsun/SummerSchool25_SBPGU`
3. Enter the directory with this project and run commands below:
    ```
    mkdir build
    cmake -B build
    cmake --build build
    ```
To run tests, type `mpirun -n 4 ./build/bin/test_MPI_GEMM`  
To run sample, type `mpirun -n 4 ./build/bin/sample`


