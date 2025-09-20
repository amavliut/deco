// namespace alg {
//   template <typename R, typename C, typename V>
template <typename R = int,typename C = int,typename V = double>
class CSR : public std::enable_shared_from_this<CSR<R,C,V>> {
public:
  bool owns_data = 1;  // True if this instance owns the memory and should free it
  C n;                          // # rows
  C m;                          // # cols
  R nz;                         // # nnz

  R* r;                         // row ptr
  C* c;                         // col ptr
  V* v;                         // val ptr

  // ------------ MPI
  bool owns_MPI = false;
  int np;
  int mid;
  int* idpool = nullptr;
  C  n_glo = 0;
  C  m_glo = 0;
  C  shift_nr = 0;
  C* displs_nr = nullptr;
  C* counts_nr = nullptr;

  R shift_nz = 0;
  R* displs_nz = nullptr;
  R* counts_nz = nullptr;

  // ------------ MPI spmv
  V* xspmv = nullptr;
  C** xcol = nullptr;
  V** xval = nullptr;
  int* source_ranks = nullptr;
  int* destin_ranks = nullptr;
  int count_src = 0;
  int count_dst = 0;
  std::vector<MPI_Request> send_requests;
  std::vector<MPI_Request> recv_requests;

  //--------------------------------------------------------------------------------

  CSR() : n(0),m(0),nz(0),r(nullptr),c(nullptr),v(nullptr) {}

  CSR(C n_,C m_,R nz_) : n(n_),m(m_),nz(nz_) {
    if (nz_ != 0) {
      r = memres<R>(n_ + 1);
      c = memres<C>(nz_);
      v = memres<V>(nz_);
    } else {
      r = nullptr;
      c = nullptr;
      v = nullptr;
    }
  }

  CSR(C n_,C m_,R nz_,
    R* r_,C* c_,V* v_) :
    n(n_),m(m_),nz(nz_),r(r_),c(c_),v(v_) {}

  CSR(C n_,C m_,R nz_,
    R* r_,C* c_,V* v_,
    int np_,int mid_,C n_glo_,
    C* displs_nr_,C* counts_nr_) :
    n(n_),m(m_),nz(nz_),r(r_),c(c_),v(v_),
    np(np_),mid(mid_),n_glo(n_glo_),
    displs_nr(displs_nr_),counts_nr(counts_nr_) {}

  CSR(C n_,C m_,R nz_,
    R* r_,C* c_,V* v_,
    int np_,int mid_,C n_glo_,C shift_nr_,
    C* displs_nr_,C* counts_nr_) :
    n(n_),m(m_),nz(nz_),r(r_),c(c_),v(v_),
    np(np_),mid(mid_),n_glo(n_glo_),shift_nr(shift_nr_),
    displs_nr(displs_nr_),counts_nr(counts_nr_) {}

  void copyMPI(std::shared_ptr<CSR<R,C,V>> A) {
    np = A->np;
    mid = A->mid;
    idpool = A->idpool;
    n_glo = A->n_glo;
    m_glo = A->m_glo;
    shift_nr = A->shift_nr;
    displs_nr = A->displs_nr;
    counts_nr = A->counts_nr;

    shift_nz = A->shift_nz;
    displs_nz = A->displs_nz;
    counts_nz = A->counts_nz;
    xspmv = A->xspmv;
  }

  void setup_spmv() {
    xspmv = memres<V>(n_glo);
  }

  void setup_spmv_v2() {
    _MPI_ENV;
    if (np > 1) {
      xspmv = memres<V>(n_glo);
      source_ranks = memres<int>(np);
      destin_ranks = memres<int>(np);
      C left = INT_MAX,right = 0;
      for (C i = 0; i < n; ++i) {
        left = MIN(left,c[r[i]]);
        right = MAX(right,c[r[i + 1] - 1]);
      }

      int source_ranks_glob[np * np];

      for (int i = 0; i < np; i++) source_ranks[i] = -1;

      alg::findIndicesWithinBounds(displs_nr,np,left,right,source_ranks,&count_src);

      CHECK_MPI(MPI_Allgather(source_ranks,np,get_mpi_type<int>(),
        source_ranks_glob,np,get_mpi_type<int>(),deco_comm));

      for (int i = 0; i < np; ++i) {
        for (int j = 0; j < np; ++j) {
          int ind = i * np + j;
          if (source_ranks_glob[ind] == mid) {
            destin_ranks[count_dst] = i;
            count_dst++;
          }
        }
      }
      send_requests.resize(count_dst);
      recv_requests.resize(count_src);
    }
  }

  void apply(const V* x,V*& b,const int bs = 1) {
    
    alg::MPI_spmv(this->shared_from_this(),x,b,bs); 
    
  }

  std::shared_ptr<CSR<R,C,V>> copy() {

    auto A = std::make_shared<CSR<R,C,V>>(n,m,nz);

    memcopy(A->r,r,n + 1);
    memcopy(A->c,c,nz);
    memcopy(A->v,v,nz);

    return A;
  }

  // old code, should be revisited
  // Sparse transposition with coefficients from CSR matrix A to CSC matrix AT 
  void trans_mat_full() {

    int i,j,iaa,iab,k,jp;
    int* IA,* JA,* IAT,* JAT;
    double* AN,* ANT;

    IA = r;
    JA = c;
    AN = v;

    IAT = memres<R>(m + 1);
    JAT = memres<C>(nz);
    ANT = memres<V>(nz);

#pragma omp parallel for
    for (i = 0; i < nz; i++) {
      JAT[i] = 0;
      ANT[i] = 0.0;
      if (i < n + 1) {
        IAT[i] = 0;
      }
    }

    // count the number of elements in each column(row)
// #pragma omp parallel for schedule(dynamic,1)
    for (i = 0; i < nz; i++) {
      j = JA[i] + 2;
      if (j > n) continue;
// #pragma omp atomic
      IAT[j]++;
    }

    // compute IAT with a shift of 1: for row j=2... iat[j] = number of elements in row j-1 (in the previous row)
    // the shift is needed so that we do not use other arrays for storing data
    // when you write in parallel you have to use another array. So the shift is not needed.
    IAT[0] = 0;
    IAT[1] = 0;
    for (i = 2; i < n + 1; i++) {
      IAT[i] = IAT[i] + IAT[i - 1];
    }
    // when IAT is computed we only need to copy the elements from JA to JAT and AN to ANT
    // #pragma omp parallel for private(i,iaa,iab,jp,j,k) //reduction(+:)
    for (i = 0; i < n; i++) {
      iaa = IA[i];
      iab = IA[i + 1];
      // #pragma omp parallel for private(jp,j,k)
      for (jp = iaa; jp < iab; jp++) {
        j = JA[jp] + 1;
        k = IAT[j];
        JAT[k] = i;
        ANT[k] = AN[jp];
        IAT[j]++;
      }
    }

    free(v);

    r = IAT;
    c = JAT;
    v = ANT;
  }


  ~CSR() {
    if (owns_data) {
      free_memory(r);
      free_memory(c);
      free_memory(v);
    }
    free_memory(xspmv);
  }
  void destroy() {
    free_memory(r);
    free_memory(c);
    free_memory(v);
  }





  ////////////////////////////////////////////////////////////////////////////
  void dense_print(std::string name = "NULL") {
    if (name != "NULL") {
      std::cout << "\n varname: " << name << "\n";
    }
    // Allocate memory for the dense matrix
    V** denseMatrix = (V**)malloc(n * sizeof(V*));
    for (int i = 0; i < n; ++i) {
      denseMatrix[i] = (V*)calloc(m,sizeof(V));
    }

    // Fill the dense matrix using CSR format
    for (int i = 0; i < n; ++i) {
      for (int j = r[i]; j < r[i + 1]; ++j) {
        denseMatrix[i][c[j]] = v[j];
      }
    }

    // Print
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        // printf("%6.2f ", (float)denseMatrix[i][j]);
        printf("%10.6f ",(float)denseMatrix[i][j]);
      }
      printf("\n");
    }
    printf("\n");

    // Free allocated memory
    for (int i = 0; i < n; ++i) {
      free(denseMatrix[i]);
    }
    free(denseMatrix);
  }


  void dense_print_pattern(std::string name = "NULL") {
    if (name != "NULL") {
      std::cout << "\n varname: " << name << "\n";
    }
    // Allocate memory for the dense matrix
    V** denseMatrix = (V**)malloc(n * sizeof(V*));
    for (int i = 0; i < n; ++i) {
      denseMatrix[i] = (V*)calloc(m,sizeof(V));
    }

    // Fill the dense matrix using CSR format
    for (int i = 0; i < n; ++i) {
      for (int j = r[i]; j < r[i + 1]; ++j) {
        denseMatrix[i][c[j]] = 1.0;
      }
    }

    // Print
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        if (i == j) printf("%5.f ",0.0);
        else printf("%5.f ",(float)denseMatrix[i][j]);
      }
      printf("\n");
    }
    printf("\n");

    // Free allocated memory
    for (int i = 0; i < n; ++i) {
      free(denseMatrix[i]);
    }
    free(denseMatrix);
  }

  void sparse_print(std::string name = "NULL",int N = 0) {
    if (name != "NULL") {
      std::cout << "\n varname: " << name << "\n";
    }
    printf("\t%5d %5d %5d\n",(int)n,(int)m,(int)nz);
    int nn = n;
    if (N > 0) nn = N;
    for (int i = 0; i < nn; i++) {
      for (int j = r[i]; j < r[i + 1]; j++) {
        printf("%5d \t\t %10d %10d %10.4f\n",j,i,(int)c[j],v[j]);
      }
    }
  }


  // convert csr matrix to dense format (debugging)
  double* csr2dense() {
    double* dense_matrix = (double*)malloc(n * m * sizeof(double));

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        dense_matrix[i * m + j] = 0.0;
      }
    }
    
    for (int i = 0; i < n; i++) {
      for (int j = r[i]; j < r[i + 1]; j++) {
        int col = c[j];
        dense_matrix[i * m + col] = v[j];
      }
    }
    return dense_matrix;
  }

  // p is the local matrix (pressure) entry; bs is the block size
  // rescale using pressure weights (eliminate one sided IMPES scaling to preserve symmetry)
  std::shared_ptr<CSR<R,C,V>> local_matrix(const int bs = 3,const int p = 0,const V* weights = nullptr) {

    _MPI_ENV;
    if (n % bs != 0 || n % bs != 0) {
      std::cerr << "n or m are not divisible by bs,\n \t n = " << n << ", m = " << m << ", bs = " << bs << std::endl;
      exit(EXIT_FAILURE);
    }
    
    C nl = n / bs; 
    C ml = m / bs;

    auto rl = memres<R>(nl + 1);

    // symbolic
#pragma omp parallel for 
    for (C i = 0; i < nl; ++i) {
      auto ia = r[bs * i + p];
      auto ib = r[bs * i + p + 1];
      R nnz = 0;
      for (auto j = ia; j < ib; ++j) {
        if (c[j] % bs == p) ++nnz;
      }
      rl[i + 1] = nnz;
    }

    // prefix sum
    rl[0] = 0;
    for (C i = 1; i < nl + 1; i++) {
      rl[i] = rl[i] + rl[i - 1];
    }
    R nzl = rl[nl];

    auto cl = memres<C>(nzl);
    auto vl = memres<V>(nzl);

    // numeric
    if (weights == nullptr) {
#pragma omp parallel for 
      for (C i = 0; i < nl; ++i) {
        auto ia = r[bs * i + p];
        auto ib = r[bs * i + p + 1];
        auto col = cl + rl[i];
        auto val = vl + rl[i];
        R nnz = 0;
        for (auto j = ia; j < ib; ++j) {
          if (c[j] % bs == p) {
            col[nnz] = c[j] / bs;
            val[nnz] = v[j];
            nnz++;
          }
        }
      }
    } else {
#pragma omp parallel for
      for (C i = 0; i < nl; ++i) { 
        auto ia = r[bs * i + p];
        auto ib = r[bs * i + p + 1];
        auto col = cl + rl[i];
        auto val = vl + rl[i];
        auto scale = weights[bs * i + p];
        R nnz = 0;
        for (auto j = ia; j < ib; ++j) {
          if (c[j] % bs == p) {
            col[nnz] = c[j] / bs;
            val[nnz] = v[j] / scale; // unscale
            nnz++;
          }
        }
      }
    }

    auto* displs_nrl = memres<C>(np + 1);
    auto* counts_nrl = memres<C>(np + 1);
    
    for (int i = 0; i <= np; ++i) {
      displs_nrl[i] = displs_nr[i] / bs;
      counts_nrl[i] = counts_nr[i] / bs;
    }

    return std::make_shared<CSR<R,C,V>>(nl,ml,nzl,rl,cl,vl,
      np,mid,n_glo / bs,displs_nrl[mid],
      displs_nrl,counts_nrl);
  }

  // filter with absolute value (used for removing 'almost' zero entries)
  void filter(const double tol) {

    auto* __restrict__ rp = memres<R>(n + 1);
    auto* __restrict__ cp = memres<C>(nz);
    auto* __restrict__ vp = memres<V>(nz);

    auto* __restrict__ rA = this->r;
    auto* __restrict__ cA = this->c;
    auto* __restrict__ vA = this->v;

#pragma omp parallel for
    for (C i = 0; i < n; ++i) {
      const R ia = rA[i];
      const R ib = rA[i + 1];
      int nnz = 0;
      for (R j = ia; j < ib; ++j) {
        if (abs(vA[j]) > tol || cA[j] == i + shift_nr) nnz++;
      }
      rp[i + 1] = nnz;
    }
    // prefix sum, to parallelize
    rp[0] = 0;
    for (C i = 1; i <= n; ++i) {
      rp[i] += rp[i - 1];
    }
    this->nz = rp[n];

#pragma omp parallel for
    for (C i = 0; i < n; ++i) {
      const R ia = rA[i];
      const R ib = rA[i + 1];
      const R off = rp[i];
      int nnz = 0;
      for (R j = ia; j < ib; ++j) {
        if (abs(vA[j]) > tol || cA[j] == i + shift_nr) {
          cp[off + nnz] = cA[j];
          vp[off + nnz] = vA[j];
          nnz++;
        }
      }
    }

    memcopy(this->r,rp,n + 1);
    memcopy(this->c,cp,nz);
    memcopy(this->v,vp,nz);

    free(rp);
    free(cp);
    free(vp);
    
  }

  // filter relative to the diagonal value
  void filter_proper(double tol) {

    auto* rp = memres<R>(n + 1);
    auto* cp = memres<C>(nz);
    auto* vp = memres<V>(nz);

#pragma omp parallel for
    for (C i = 0; i < n; ++i) {
      R ia = r[i];
      R ib = r[i + 1];
      auto j = core::binarysearch(c,ia,ib,i + shift_nr); // find the diagonal value
      V D = abs(v[j] * tol);
      int nnz = 0;
      for (R j = ia; j < ib; ++j) {
        if (abs(v[j]) > D || c[j] == i + shift_nr) nnz++;
      }
      rp[i + 1] = nnz;
    }
    // prefix sum, parallelize
    rp[0] = 0;
    for (C i = 1; i <= n; ++i) {
      rp[i] += rp[i - 1];
    }
    this->nz = rp[n];

#pragma omp parallel for
    for (C i = 0; i < n; ++i) {
      R ia = r[i];
      R ib = r[i + 1];
      R off = rp[i];
      int nnz = 0;
      auto j = core::binarysearch(c,ia,ib,i + shift_nr); // find the diagonal value
      V D = abs(v[j] * tol);
      for (R j = ia; j < ib; ++j) {
        if (abs(v[j]) > D || c[j] == i + shift_nr) {
          cp[off + nnz] = c[j];
          vp[off + nnz] = v[j];
          nnz++;
        }
      }
    }

    memcopy(this->r,rp,n + 1);
    memcopy(this->c,cp,nz);
    memcopy(this->v,vp,nz);

    free(rp);
    free(cp);
    free(vp);
  }


private:

};
