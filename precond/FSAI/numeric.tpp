/*
  compute the size of rows of prec pattern
*/
template<typename R>
void cpt_sizeOfRows(R n,R* r,int* sizeOfRows,int* position) {

#pragma omp parallel for //shared(sizeOfRows,r,position)
  for (R i = 0; i < n; ++i) {
    sizeOfRows[i] = r[i + 1] - r[i] - 1;
    position[i] = i;
  }
}


/*
  compute the cumulative size of dense local systems
  compute the cumulative size of rhs vectors
*/
void cpt_cum_size(int* cumsize,int* sizerhs,int* sizeOfRows,int nrows) {

  cumsize[0] = 0;
  sizerhs[0] = 0;

  for (int i = 0; i < nrows; ++i) {
    int size = sizeOfRows[i];
    cumsize[i + 1] = cumsize[i] + size * size;
    sizerhs[i + 1] = sizerhs[i] + size;
  }
}


// scatter the local system solution back into the preconditioner
template<typename R,typename C,typename V>
void scatter(shared_ptr<CSR<R,C,V>> FL,shared_ptr<CSR<R,C,V>> FU,
  double* x_FL,double* x_FU,int Ns,
  int* size_rhs,int* sizeOfRows,int* position) {

  auto* iatFL = FL->r;
  auto* coefFL = FL->v;
  auto* coefFU = FU->v;
  int i;

#pragma omp parallel for private(i)
  for (i = 0; i < Ns; i++) {
    auto* fL = x_FL + size_rhs[i];
    auto* fU = x_FU + size_rhs[i];
    auto size_rhs_length = sizeOfRows[i];
    auto pos = iatFL[position[i]];
    for (int k = 0; k < size_rhs_length; ++k) {
      coefFL[pos + k] = fL[k];
      coefFU[pos + k] = fU[k];
    }
  }
}




template <typename R,typename C,typename V>
void numeric(shared_ptr<CSR<R,C,V>> A,shared_ptr<CSR<R,C,V>> FL,shared_ptr<CSR<R,C,V>> FU) {

  int nrows = A->n;
  auto* sizeOfRows = memres<int>(nrows);
  auto* position = memres<int>(nrows);
  auto* cumsize = memres<int>(nrows + 1);
  auto* sizerhs = memres<int>(nrows + 1);

  // compute the sizes of rows of the preconditioner
  profiler.start("cpt_cum_size");
  cpt_sizeOfRows(nrows,FL->r,sizeOfRows,position);
  profiler.stop();

  // sort sizeOfRows to have a better workload balance
  profiler.start("sort");
  alg::sortByKey(sizeOfRows,position,nrows);
  profiler.stop();

  // compute the auxilary arrays required to locate fast the data for a specific local matrix 
  cpt_cum_size(cumsize,sizerhs,sizeOfRows,nrows);

  auto* full_A = memput<double>(cumsize[nrows]);
  auto* rhsL = memput<double>(sizerhs[nrows]);
  auto* rhsU = memput<double>(sizerhs[nrows]);
  auto* D = memres<double>(nrows);

  // gather local systems
  profiler.start("gather");
  gather(A,full_A,FL,nrows,cumsize,sizerhs,rhsL,rhsU,position);
  profiler.stop();

  // solve local systems
  profiler.start("linsolve");
  linsolve_LAPACK(full_A,rhsU,rhsL,nrows,nrows,cumsize,sizerhs,sizeOfRows,FU,FL,position);
  profiler.stop();

  // scatter the local solution back to the FSAI factors
  profiler.start("scatter");
  scatter(FL,FU,rhsL,rhsU,nrows,sizerhs,sizeOfRows,position);
  profiler.stop();

  // compute the preconditioned diagonal
  profiler.start("diagonal");
  // diagonal(FL, A, FU, D);
  MxM_diag(FL,A,FU,D);
  profiler.stop();

  // Jacobi scale the diagonal
  profiler.start("Jacobi_scale");
  Jacobi_scale(nrows,D,FL->r,FL->v,FU->v);
  profiler.stop();

  // Transpose FU into upper triangular matrix
  profiler.start("Transpose");
  FU->trans_mat_full();
  profiler.stop();


#ifdef print_norm
  std::ofstream("outnorm.txt",std::ios::app) << std::fixed << std::setprecision(5) << std::setw(15) << alg::l2norm(FL->nz,FL->v);
  std::ofstream("outnorm.txt",std::ios::app) << std::fixed << std::setprecision(5) << std::setw(15) << alg::l2norm(FL->nz,FU->v);
#endif

  // std::cout << "numeric part is done\n";
  free(sizeOfRows);
  free(position);
  free(cumsize);
  free(sizerhs);
  free(rhsU);
  free(rhsL);
  free(full_A);
  free(D);

}