#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include <fstream>
#include <string>
#include "CombBLAS/CombBLAS.h"

using namespace std;
using namespace combblas;

#ifdef TIMING
double cblas_alltoalltime;
double cblas_allgathertime;
#endif

#ifdef _OPENMP
int cblas_splits = omp_get_max_threads(); 
#else
int cblas_splits = 1;
#endif

template <class NT>
class PSpMat 
{ 
public: 
	typedef SpDCCols < int64_t, NT > DCCols;
	typedef SpParMat < int64_t, NT, DCCols > MPI_DCCols;
};

typedef PSpMat<bool>::MPI_DCCols PSpMatBool;
typedef PlusTimesSRing<bool, bool> PTBOOLBOOL;	
typedef SelectMaxSRing<bool, int64_t> SR;	

PSpMatBool operator* (PSpMatBool& A, PSpMatBool& B) {
	return Mult_AnXBn_Synch<PTBOOLBOOL, bool, PSpMat<bool>::DCCols >(A,B);
}

PSpMatBool operator* (PSpMatBool&& A, PSpMatBool& B) {
	return Mult_AnXBn_Synch<PTBOOLBOOL, bool, PSpMat<bool>::DCCols >(A,B);
}

int main(int argc, char* argv[])
{
	int nprocs, myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	if(argc < 3)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./MultTest <BASEADDR> start_vertex" << endl;
			cout << "Baseaddr dir must contain alf.txt with alfabet, starts.mtx, finals.mtx, output.mtx, zero.mtx and matrices for each symbol in directories graph and FA" << endl;
		}
		MPI_Finalize(); 
		return -1;
	}				
	{
		string Dir(argv[1]);		
		MPI_Barrier(MPI_COMM_WORLD);	

		shared_ptr<CommGrid> fullWorld;
		fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
		std::vector<PSpMatBool> FA;
		std::vector<PSpMatBool> Graph;
		std::vector<std::string> Alf;
		std::string Start(argv[2]);
		std::string str;
		ifstream AlfSt(Dir+"/alf.txt");
		while (std::getline(AlfSt, str)) Alf.push_back(str);

		for (size_t i = 0; i < Alf.size(); ++i) {
			str = Alf[i];
			FA.push_back(PSpMatBool(fullWorld));
			Graph.push_back(PSpMatBool(fullWorld));
			FA[FA.size() - 1].ParallelReadMM(Dir+"/FA/"+str+".mtx", true, maximum<bool>());
			FA[FA.size() - 1].Transpose();
			Graph[Graph.size() - 1].ParallelReadMM(Dir+"/graph/"+str+".mtx", true, maximum<bool>());
		}
		PSpMatBool P(fullWorld);
		PSpMatBool Zero(fullWorld);
		PSpMatBool tmpP(fullWorld);
		PSpMatBool oldP(fullWorld);
		P.ParallelReadMM(Dir+"/starts.mtx", true, maximum<bool>());
		Zero.ParallelReadMM(Dir+"/zero.mtx", true, maximum<bool>());
		PSpMatBool F(fullWorld);
		F.ParallelReadMM(Dir+"/finals.mtx", true, maximum<bool>());

		bool changed = true;
		int iteration = 0;
		while (changed) {
			if (myrank == 0) std::cout << iteration << "\n";
			changed = false;
			tmpP = Zero;
			for (size_t i = 0; i < Alf.size(); ++i) {
				tmpP += FA[i] * P * Graph[i];
				MPI_Barrier(MPI_COMM_WORLD);	
			}
			oldP = P;
			P += tmpP;
			if (!(oldP == P)) changed = true;
//			P.SaveGathered(Dir + "/log/" + to_string(iteration) + ".mtx"); //
			++iteration;
			MPI_Barrier(MPI_COMM_WORLD);	
		}

		PSpMatBool Res(fullWorld);
		Res = F * P;
		Res.SaveGathered(Dir + "/output.mtx"); 
	}
	MPI_Finalize();
	return 0;
}

