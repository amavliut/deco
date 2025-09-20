/*
    gather the local systems into the full_A (the start of each system is stored in cumsize array)
    gather rhsL and rhsU vectors (the start of each vector is stored in sizerhs array)
    A is the matrix A
    F is the lower preconditioner matrix

    This is an experimental way to gather local systems via hash table and it seems to be working well
    but it still can be better optimized.
    The classical way based on binary search was the starting point, however, it is constly O(log(m))
    where m is the size of the row. In particular, binary search was used for insertion of the corresponding
    values into dense format. With hashtable the insertion is much faster  
*/

template <typename R,typename C,typename V >
void gather(shared_ptr<CSR<R,C,V>> A,double* full_A,shared_ptr<CSR<R,C,V>> F,int nrows,int* cumsize,int* sizerhs,double* rhsL,double* rhsU,
  int* pos) {

  auto* rA = A->r;
  auto* cA = A->c;
  auto* vA = A->v;

  auto* rF = F->r;
  auto* cF = F->c;
  auto* vF = F->v;

  int offset = cumsize[0];
  int offrhs = sizerhs[0];

  // auto hash_sz = find_htable_sz(static_cast<int>(sizerhs[1] * 1.07));
  // std::cout << " hash_sz = " << find_htable_sz(sizerhs[1] * 1.07) << "\n";
  // std::cout << " hash_sz = " << ceil(sizerhs[1] * 1.07) << "\n";
  // int hash_sz = 16;
#pragma omp parallel 
  {
    int hash_sz = find_htable_sz(sizerhs[1] * 1.07);

    int hashtb_key[hash_sz];
    int hashtb_ind[hash_sz];

    int ind;

#pragma omp for //schedule(dynamic, 1)
    for (C i = 0; i < nrows; ++i) {
      auto iaF = rF[pos[i]];
      auto ibF = rF[pos[i] + 1] - 1; // -1 to exclude the diagonal terms
      auto rowsize = ibF - iaF;
      if (rowsize <= 0) continue;

      // initialize the hash table: column indices for keys, position of column indices - ind
      memset(hashtb_key,-1,sizeof(hashtb_key));
      for (auto j = iaF; j < ibF; ++j) {
        hashmap_insert(hashtb_key,hashtb_ind,cF[j],j,hash_sz-1);
      }

      // loop over each column index of F pattern 
      for (auto j = iaF; j < ibF; ++j) {
        auto rowid = j - iaF;
        auto colF = cF[j];     // convert the loop over col ind into the loop over the rows of mat A
        auto jaA = rA[colF];
        auto jbA = rA[colF + 1];
        for (auto s = jaA; s < jbA; ++s) {
          auto key = cA[s];
          ind = hashmap_check(hashtb_key,key,hash_sz - 1);
          if (hashtb_key[ind] == key){
            full_A[cumsize[i] + rowid * (rowsize)-offset + hashtb_ind[ind] - iaF] = vA[s];
          }
        }
        // gather -rhsU array
        ind = core::binarysearch(cA,jaA,jbA,cF[ibF]);
        if (cA[ind] == cF[ibF]){
          rhsU[sizerhs[i] + rowid - offrhs] = -vA[ind];
        }
      }
      auto colF = cF[ibF];
      auto jaA = rA[colF];
      auto jbA = rA[colF + 1];

      // gather rhsL array
      for (auto s = jaA; s < jbA; s++) {
        auto key = cA[s];
        auto ind = hashmap_check(hashtb_key,key,hash_sz - 1);
        if (hashtb_key[ind] == key){
          rhsL[sizerhs[i] - offrhs + hashtb_ind[ind] - iaF] = -vA[s];
        }
      }
    }
  }
}