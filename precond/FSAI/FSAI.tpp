#include "gather.cpp"
#include "LU_main.cpp"
#include "diagonal.cpp"

#include "symbolic.tpp"
#include "numeric.tpp"


// explore loop unrolling and __builtin_prefetch
template <typename R, typename C, typename V>
inline void FSAI<R,C,V>::setup(shared_ptr<CSR<R,C,V>> A, int k){

  profiler.start("symbolic");
  symbolic(A, FL->nz, FL->r, FL->c, k);               // compute the power pattern
  profiler.stop();

  FL->v = memres<V>(FL->nz);
  FU->v = memres<V>(FL->nz);

  // use the same pattern for FL and FU
  FU->nz = FL->nz;
  FU->r  = FL->r;
  FU->c  = FL->c;

  // std::cout << "matrix size:\t";
  // std::cout << "n = " << FL->n << ", m = " << FL->m << ", nz = " << FL->nz << "\n";

  profiler.start("numeric");
  numeric(A, FL, FU);      // compute the coefficients
  profiler.stop();  
  
}