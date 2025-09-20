// simple Jacobi preconditioner
template <typename R,typename C,typename V>
inline void Jacobi<R,C,V>::classic(shared_ptr<CSR<R,C,V>> A) {

  n = A->n;
  const auto* __restrict__ ra = A->r;
  const auto* __restrict__ ca = A->c;
  const auto* __restrict__ va = A->v;
  const auto shift_nr = A->shift_nr;

  J = memres<V>(n);

  // compute the values
#pragma omp parallel for
  for (C i = 0; i < n; ++i) {
    auto ia = ra[i];
    auto ib = ra[i + 1];
    auto j = core::binarysearch(ca,ia,ib,i + shift_nr);
    J[i] = 1 / va[j];
  }
}


// block Jacobi preconditioner of size [bs × bs]
template <typename R,typename C,typename V>
inline void Jacobi<R,C,V>::blocked(shared_ptr<CSR<R,C,V>> A,const int bs) {

  n = A->n;
  sn = n / bs; // (# of cells) 
  J = memres<V>(sn * bs * bs);

  // A matrix
  const auto* ra = A->r;
  const auto* ca = A->c;
  const auto* va = A->v;

  const auto shift_nr = A->shift_nr;

  meminit(J,sn * bs * bs); // initialize J to zero

  // compute the values
#pragma omp parallel
  {
    int info;
    int ipiv[bs]; // permutations array
#pragma omp for // partition cells continously
    for (C s = 0; s < sn; ++s) { // loop over cells
      V* Ap = J + s * bs * bs;
      for (int p = 0; p < bs; ++p) { // loop the cell's entries
        C i = s * bs + p; // global A row index
        auto ia = ra[i];
        auto ib = ra[i + 1];
        // assuming that ca is sorted!
        auto j = core::binarysearch(ca,ia,ib,i + shift_nr); // find the diagonal (assuming there is always a diagonal entry)
        for (int k = 0; k < bs; ++k) { // loop over block columns
          if ((ca[j + k - p] - shift_nr) / bs == s) { // check if col entry is in the diagonal cell
            if (j + k - p < ib && j + k - p >= ia) { // check if you are still in the same row
              Ap[p * bs + (ca[j + k - p] - shift_nr) % bs] = va[j + k - p]; // no need to transpose (you take the inverse of a dense mat, order doesn't matter)
            }
          }
        }
      }

      info = CHECK(LAPACKE_dgetrf(LAPACK_COL_MAJOR,bs,bs,Ap,bs,ipiv)); // LU factorization
      info = CHECK(LAPACKE_dgetri(LAPACK_COL_MAJOR,bs,Ap,bs,ipiv));    // invert the local matrix

    }
  }
}
