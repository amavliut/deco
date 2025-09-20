/*********************************************************************
  Symbolic stage is essential SpGEMM operation performed symbolically,
  therefore, it follows closely the nsparse algorithm of computing
  the matrix product between two sparse matrices
**********************************************************************/

/*
  Compute the number of intermediate prodcuts of C = A·A
*/
template <typename R,typename C,typename V>
R* max_size(shared_ptr<CSR<R,C,V>> A) {
  auto   n = A->n;
  auto* ra = A->r;
  auto* ca = A->c;

  auto row_nz = memres<R>(n);

  for (C i = 0; i < n; i++) {
    auto ia = ra[i];
    auto ib = ra[i + 1];
    R rnz = 0.0;
    for (auto j = ia; j < ib; j++) {
      auto col = ca[j];
      rnz += ra[col + 1] - ra[col];
    }
    row_nz[i] = rnz;
  }
  return row_nz;
}

/*
  Count the actual number of nonzeros per row in the output
  in order to find the row pointer for the CSR matrix
*/
template <typename R,typename C>
void MxM_pattern_size(const R n,const R* r,const C* c,const int sz,R* row_nz) {

#pragma omp parallel for
  for (R i = 0; i < n; ++i) {
    C hashtb[sz];
    for (int j = 0; j < sz; ++j) {
      hashtb[j] = -1;
    }
    int nz = 0;
    for (R j = r[i]; j < r[i + 1]; ++j) {
      C cA = c[j];
      for (R k = r[cA]; k < r[cA + 1]; ++k) {
        C key = c[k];
        if (key > i) break; // take only the lower diagonal part
        hashmap_symbolic(nz,hashtb,key,sz);
      }
    }
    row_nz[i] = nz;
  }
}

/*
  Compute the column indices of the output pattern
*/
template <typename R,typename C>
void MxM_pattern(const R n,const R* r,const C* c,
  const R* rp,C* cp,const unsigned int sz) {

#pragma omp parallel for
  for (R i = 0; i < n; ++i) {
    C hashtb[sz];
    for (int i = 0; i < sz; ++i) {
      hashtb[i] = -1;
    }
    int nz = 0;
    for (R j = r[i]; j < r[i + 1]; ++j) {
      C cA = c[j];
      for (R k = r[cA]; k < r[cA + 1]; ++k) {
        C key = c[k];
        if (key > i) break; // take only the lower diagonal part
        hashmap_pattern_bit(hashtb,key,sz - 1);
        // hashmap_pattern(hashtb, key, sz);
      }
    }
    // compact
    for (C l = 0; l < sz; ++l) {
      C key = hashtb[l];
      if (key != -1) {
        hashtb[nz] = key;
        nz++;
      }
    }
    // sort the hash table
    alg::radixSort(hashtb,nz);
    // dispatch the column indices
    R offset = rp[i];
    for (C l = 0; l < nz; ++l) {
      cp[offset + l] = hashtb[l];
    }
  }
}



// Main symbolic function routine
template <typename R,typename C,typename V>
void symbolic(shared_ptr<CSR<R,C,V>> A,R& nz_,R*& r,C*& c,int k = 1) {

  if (k == 1) {
    C   n = A->n;
    R* rA = A->r;
    C* cA = A->c;
    V* vA = A->v;

    R* rF = memres<R>(n + 1);

#pragma omp parallel for
    for (C i = 0; i < n; ++i) {
      R ia = rA[i];
      R ib = rA[i + 1];
      int nnz = 0;
      for (R j = ia; j < ib; ++j) {
        if (cA[j] <= i) {
          nnz++;
        } else break;
      }
      rF[i + 1] = nnz;
    }
    rF[0] = 0;
    for (C i = 1; i <= n; ++i) {
      rF[i] += rF[i - 1];
    }

    auto* cF = memres<C>(rF[n]);

#pragma omp parallel for
    for (C i = 0; i < n; ++i) {
      R ia = rA[i];
      R ib = rA[i + 1];
      R off = rF[i];
      int nnz = 0;
      for (R j = ia; j < ib; ++j) {
        if (cA[j] <= i) {
          cF[off + nnz] = cA[j];
          nnz++;
        } else break;
      }
    }

    r = rF;
    c = cF;
    nz_ = rF[n];

  } else if (k == 2) {
    auto n = A->n;
    auto* row_nz = max_size(A);                         // compute intermediate products (max possible row size per row)
    auto nz = alg::int_sum(n,row_nz);               // total number of the intermediate products
    auto hsz = alg::int_max(n,row_nz);              // find the maximum row size (the size of the hash table)

    MxM_pattern_size(n,A->r,A->c,hsz,row_nz);        // compute the actual nonzero count (later to compute rp)
    hsz = alg::int_max(n,row_nz);                    // update max row_nz (you can actully just pass row_nz and use hash table size per row)

    auto* rp = memres<R>(n + 1);                          // allocate the space for the pattern
    alg::prefix_sum(row_nz,rp,n);                    // compute the row pointer
    nz = rp[n];                                         // nz of the pattern
    auto* cp = memres<C>(nz);                           // column indices

    auto sz = find_htable_sz(hsz * 1.07); // take 7% extra

    // compute the pattern indices
    MxM_pattern(n,A->r,A->c,rp,cp,sz);             // compute the actual pattern

    r = rp;
    c = cp;
    nz_ = nz;

    free(row_nz);
  } else {
    std::cerr << "k > 2 is not implemented, k = " << k << std::endl;
    exit(EXIT_FAILURE);
  }
}