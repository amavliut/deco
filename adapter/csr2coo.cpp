namespace AD {
  template<typename R,typename C=R>
  C* csr2coo(R n,R nz,R*r) {
    R i;
    C* l = CHECK((C*)malloc((nz) * sizeof(C)));
    #pragma omp parallel for private(i)
    for (C i = 0; i < n; i++) {
      R ra = r[i];
      R rb = r[i+1];
      for (R j = ra; j < rb; j++) {
        l[j] = i;
      }
    }
    return l;
  }
  
  // reorder the cells of A (csr format) and b
  template<typename R, typename C, typename V>
  void block_reorder(shared_ptr<CSR<R,C,V>> A, V*& b, const boost::property_tree::ptree param) {
    
    auto n  = A->n;
    auto nz = A->nz;
    auto c  = A->c;
    R i;

    int bs = param.get<int>("prec.block_size");
    std::string row_ord = param.get<std::string>("prec.row_order");
    std::string col_ord = param.get<std::string>("prec.col_order");
    
    // Compute the inverse permutation
    int reord_row[bs]; int reord_col[bs]; int ord_row[bs];
    for (int i = 0; i < bs; ++i) {
        int row = int(row_ord[i] - '0') - 1; // convert to zero base
        int col = int(col_ord[i] - '0') - 1;
        reord_row[row] = i;
        reord_col[col] = i;
        ord_row[i] = row;
    }

    if      (bs == 1) return;
    else if (bs == 2) if (reord_row[0] == 0 && reord_row[1] == 1 && 
                          reord_col[0] == 0 && reord_col[1] == 1) return;
    else if (bs == 3) if (reord_row[0] == 0 && reord_row[1] == 1 && reord_row[2] == 2 &&
                          reord_col[0] == 0 && reord_col[1] == 1 && reord_col[2] == 2) return;
    else {
      std::cerr << "Error: unknown block size, bs = " << bs << std::endl;
      exit(EXIT_FAILURE);
    }

    // convert to coordinate format
    auto* l = csr2coo<R,C>(n,nz,A->r);
    
    // reorder the rows of A stored in coo in-place
    #pragma omp parallel for private(i)
    for (i = 0; i < nz; ++i){
      int rest = l[i] % bs; 
      l[i] = l[i] - rest + reord_row[rest];
    }

    // reorder the cols of A stored in coo in-place
    #pragma omp parallel for private(i)
    for (i = 0; i < nz; ++i){
      int rest = c[i] % bs;  
      c[i] = c[i] -rest + reord_col[rest];
    }

    // sort the rows using key-value1-value2 pairs
    // sort once, make a comparison based on if row==row then compare col
    // not optimized
    alg::sortByKey(nz, l, A->c, A->v);

    // find the row pointer (beware of empty rows)
    IO::find_row_ptr(A->r, A->nz, l);
    
    V scratch[bs];
    auto sn = n/bs;
    #pragma omp parallel for private(scratch)
    for (int s = 0; s < sn; ++s){
      for (int p = 0; p < bs; ++p){
        scratch[p] = b[s*bs+p];
      }
      for (int p = 0; p < bs; ++p){
        b[s*bs + p] = scratch[ord_row[p]];
      }
    }

    free(l);
  }


  // reorder the cells of A (coo format) and b
  template<typename R, typename C, typename V>
  shared_ptr<CSR<R,C,V>> block_reorder_coo(C &n,C &m,R &nz,C*&l,C*&c,V*&v,V*&b,const boost::property_tree::ptree param){

    R *r = (R*)malloc((n+1) * sizeof(R));
    R i;

    int bs = param.get<int>("prec.block_size");
    std::string row_ord = param.get<std::string>("prec.row_order");
    std::string col_ord = param.get<std::string>("prec.col_order");
    
    // Compute the inverse permutation
    int reord_row[bs]; int reord_col[bs]; int ord_row[bs];
    for (int i = 0; i < bs; ++i) {
        int row = int(row_ord[i] - '0') - 1; // convert to zero base
        int col = int(col_ord[i] - '0') - 1;
        reord_row[row] = i;
        reord_col[col] = i;
        ord_row[i] = row;
    }

    // reorder the rows of A stored in coo format in-place
    #pragma omp parallel for private(i)
    for (i = 0; i < nz; ++i){
      int rest = l[i] % bs; 
      l[i] = l[i] - rest + reord_row[rest];
    }

    // reorder the cols of A stored in coo format in-place
    #pragma omp parallel for private(i)
    for (i = 0; i < nz; ++i){
      int rest = c[i] % bs;  
      c[i] = c[i] -rest + reord_col[rest];
    }

    // sort the rows using key-value1-value2 pairs
    // sort once, make a comparison based on if row==row then compare col
    alg::sortByKey(nz, l, c, v);

    // find the row pointer (beware of empty rows)
    IO::find_row_ptr(r, nz, l);

    // reorder b array
    V scratch[bs];
    auto sn = n/bs;
    #pragma omp parallel for private(scratch)
    for (int s = 0; s < sn; ++s){
      for (int p = 0; p < bs; ++p){
        scratch[p] = b[s*bs+p];
      }
      for (int p = 0; p < bs; ++p){
        b[s*bs + p] = scratch[ord_row[p]];
      }
    }

    auto A = std::make_shared<CSR<R,C,V>>(n,m,nz,r,c,v);

    return A;
  }

  // reorder the cells of A (coo format), b, and x
  template<typename R, typename C, typename V>
  shared_ptr<CSR<R,C,V>> block_reorder_coo(C &n,C &m,R &nz,C*&l,C*&c,V*&v,V*&b,V*&x,const boost::property_tree::ptree param){

    R *r = (R*)malloc((n+1) * sizeof(R));
    R i;

    int bs = param.get<int>("prec.block_size");
    std::string row_ord = param.get<std::string>("prec.row_order");
    std::string col_ord = param.get<std::string>("prec.col_order");
    
    // Compute the inverse permutation
    int reord_row[bs]; int reord_col[bs]; int ord_row[bs];
    for (int i = 0; i < bs; ++i) {
        int row = int(row_ord[i] - '0') - 1; // convert to zero base
        int col = int(col_ord[i] - '0') - 1;
        reord_row[row] = i;
        reord_col[col] = i;
        ord_row[i] = row;
    }

    // reorder the rows of A stored in coo format in-place
    #pragma omp parallel for private(i)
    for (i = 0; i < nz; ++i){
      int rest = l[i] % bs; 
      l[i] = l[i] - rest + reord_row[rest];
    }

    // reorder the cols of A stored in coo format in-place
    #pragma omp parallel for private(i)
    for (i = 0; i < nz; ++i){
      int rest = c[i] % bs;  
      c[i] = c[i] -rest + reord_col[rest];
    }

    // sort the rows using key-value1-value2 pairs
    // sort once, make a comparison based on if row==row then compare col
    alg::sortByKey(nz, l, c, v);

    // find the row pointer
    IO::find_row_ptr(r, nz, l);

    // reorder b & x arrays
    V scratch[bs];
    V scratchx[bs];
    auto sn = n/bs;
    #pragma omp parallel for private(scratch)
    for (int s = 0; s < sn; ++s){
      for (int p = 0; p < bs; ++p){
        scratch [p] = b[s*bs+p];
        scratchx[p] = x[s*bs+p];
      }
      for (int p = 0; p < bs; ++p){
        b[s*bs + p] = scratch [ord_row[p]];
        x[s*bs + p] = scratchx[ord_row[p]];
      }
    }

    auto A = std::make_shared<CSR<R,C,V>>(n,m,nz,r,c,v);

    return A;
  }


  // column reorder of x according to the original ordering
  template <typename C, typename V>
  void inorder(const C n, V *&x, const boost::property_tree::ptree param){

    int bs = param.get<int>("prec.block_size");
    std::string col_ord = param.get<std::string>("prec.col_order");

    int ord_col[bs];
    for (int i = 0; i < bs; ++i) {
        int col = int(col_ord[i] - '0') - 1; // convert to zero base
        ord_col[i] = col;
    }

    V scratch[bs];
    auto sn = n/bs;
    #pragma omp parallel for private(scratch)
    for (int s = 0; s < sn; ++s){
      for (int p = 0; p < bs; ++p){
        scratch[p] = x[s*bs+p];
      }
      for (int p = 0; p < bs; ++p){
        x[s*bs + p] = scratch[ord_col[p]];
      }
    }
  }

}// close AD