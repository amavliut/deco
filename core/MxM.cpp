/*********************************************************************
  nsparse on CPU
**********************************************************************/

/*
  Compute the number of intermediate prodcuts of C = A·B
*/
template <typename R,typename C,typename V>
R* max_size(std::shared_ptr<CSR<R,C,V>> A,std::shared_ptr<CSR<R,C,V>> B) {
  auto   n = A->n;
  auto* ra = A->r;
  auto* ca = A->c;

  auto* rb = B->r;

  auto row_nz = memres<R>(n);

  // C i; R rnz = 0.0;
#pragma omp parallel for
  for (C i = 0; i < n; i++) {
    auto ia = ra[i];
    auto ib = ra[i + 1];
    R rnz = 0.0;
    for (auto j = ia; j < ib; j++) {
      auto colA = ca[j];
      rnz += rb[colA + 1] - rb[colA];
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
void MxM_pattern_size(const R n,const R* rA,const C* cA,const R* rB,const C* cB,const int sz,R* row_nz) {

  C hashtb[sz];
#pragma omp parallel for private(hashtb)
  for (C i = 0; i < n; ++i) {
    for (int j = 0; j < sz; ++j) {
      hashtb[j] = -1;
    }
    int nz = 0;
    for (R j = rA[i]; j < rA[i + 1]; ++j) {
      C colA = cA[j];
      for (R k = rB[colA]; k < rB[colA + 1]; ++k) {
        C key = cB[k];
        // if (key > i) break; // take only the lower diagonal part
        hashmap_symbolic(nz,hashtb,key,sz);
      }
    }
    row_nz[i] = nz;
  }
}


/*
  Compute the column indices of the output pattern
*/
template <typename R,typename C,typename V>
void MxM_compute(const R n,const unsigned int sz,
  const R* rA,const C* cA,const V* vA,
  const R* rB,const C* cB,const V* vB,
  const R* rC,C* cC,V* vC) {

  alg::Pair0<C,V> hashtb[sz]; // change the order from AoS to SoA
#pragma omp parallel for private(hashtb) schedule(dynamic,1)
  for (C i = 0; i < n; ++i) {
    for (int j = 0; j < sz; ++j) {
      hashtb[j].key = -1;   // use memset
      hashtb[j].val = 0.0;  // use memset
    }
    for (R j = rA[i]; j < rA[i + 1]; ++j) {
      C colA = cA[j];
      V valA = vA[j];
      for (R k = rB[colA]; k < rB[colA + 1]; ++k) {
        C key = cB[k];
        V val = vB[k] * valA;
        // if (key > i) break; // take only the lower diagonal part
        // hashmap_pattern_bit(hashtb, key, sz-1);
        // hashmap(hashtb, key, val, sz);
        hashmap_bit(hashtb,key,val,sz - 1);
      }
    }
    // compact
    int nz = 0;
    for (int l = 0; l < sz; ++l) {
      C key = hashtb[l].key;
      if (key != -1) {
        hashtb[nz].key = key;
        hashtb[nz].val = hashtb[l].val;
        nz++;
      }
    }
    // sort the hash table
    // qsort(hashtb, nz, sizeof(*hashtb), cmpr<Pair<C,V>>);
    alg::radixSort(hashtb,nz);
    // alg::bitonicSort(hashtb, 0, nz, true);
    // dispatch the column indices
    R offset = rC[i];
    for (int l = 0; l < nz; ++l) {
      cC[offset + l] = hashtb[l].key;
      vC[offset + l] = hashtb[l].val;
    }
  }
}


template <typename R,typename C,typename V>
std::shared_ptr<CSR<R,C,V>> MxM(std::shared_ptr<CSR<R,C,V>> A,std::shared_ptr<CSR<R,C,V>> B) {

  auto n = A->n;
  auto* row_nz = max_size(A,B);                        // compute intermediate products (max possible row size per row)
  auto nz = alg::int_sum(n,row_nz);                    // total number of the intermediate products
  auto hsz = alg::int_max(n,row_nz);                   // find the maximum row size (the size of the hash table)

  profiler.start("symbolic");
  MxM_pattern_size(n,A->r,A->c,B->r,B->c,hsz,row_nz);  // compute the actual nonzero count (later to compute rC)
  profiler.stop();

  hsz = alg::int_max(n,row_nz);                        // update max row_nz (you can actully just pass row_nz and use hash table size per row)

  auto* rC = memres<R>(n + 1);                         // allocate the space for the pattern
  alg::prefix_sum(row_nz,rC,n);                        // compute the row pointer
  nz = rC[n];                                          // nz of the pattern
  auto* cC = memres<C>(nz);                            // column indices
  auto* vC = memres<V>(nz);                            // values

  auto sz = find_htable_sz(hsz * 1.07);                // take 7% extra

  // compute the pattern indices
  profiler.start("numeric");
  MxM_compute(n,sz,A->r,A->c,A->v,
    B->r,B->c,B->v,
    rC,cC,vC);
  profiler.stop();

  free(row_nz);

  auto CC = std::make_shared<CSR<R,C,V>>(n,B->m,nz,rC,cC,vC);

  return CC;
}