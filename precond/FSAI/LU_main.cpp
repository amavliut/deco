// #include "LU_local.cpp"
#include <cblas.h>
#include <lapacke.h>

// #include "mkl.h" // v2024.2.1
// #include "mkl_lapacke.h"
// #include "mkl_spblas.h"


/*
    1. Find the LU decomposition of a number of dense matrices (symmetric pattern but nonsymmetric numberical value)
    2. Solve these dense local systems
*/
template<typename R,typename C,typename V>
void linsolve_LAPACK(double* full_A_U,double* rhs_U,double* rhs_L,int Ns,int nrows_A,int* cumsize,
  int* size_rhs,int* sizeOfRows,shared_ptr<CSR<R,C,V>> FU,shared_ptr<CSR<R,C,V>> FL,
  int* position = nullptr) {
  using namespace alg;

  int i;

  int* iatFL = FL->r;
  int* jaFL = FL->c;
  double* coefFL = FL->v;

  // only one pattern
  int* iatFU = FL->r;
  int* jaFU = FL->c;
  double* coefFU = FU->v;

#pragma omp parallel private(i)
  {
    int ipiv[sizeOfRows[0]];
    int info;

#pragma omp for schedule(static,2) // workload balance vs cache use
    for (i = 0; i < Ns; i++) {

      if (sizeOfRows[i] > 0) {
        // since LAPACK by uses column major order it is 20% more efficient to use it instead of row major which involves transpose 
        info = LAPACKE_dgetrf(LAPACK_COL_MAJOR,sizeOfRows[i],sizeOfRows[i],full_A_U + cumsize[i],sizeOfRows[i],ipiv);
        info = LAPACKE_dgetrs(LAPACK_COL_MAJOR,'T',sizeOfRows[i],1,full_A_U + cumsize[i],sizeOfRows[i],ipiv,rhs_U + size_rhs[i],sizeOfRows[i]);
        info = LAPACKE_dgetrs(LAPACK_COL_MAJOR,'N',sizeOfRows[i],1,full_A_U + cumsize[i],sizeOfRows[i],ipiv,rhs_L + size_rhs[i],sizeOfRows[i]); // //

      }
    }
  }
}
