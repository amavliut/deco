#include "../adapter/array_adapter.cpp"

template<typename V>
class CPRBase {
public:
    virtual ~CPRBase() = default;
    std::function<void(V* __restrict__, V* & __restrict__ )> apply;
};


template <typename R = int,typename C = int,typename V = double, 
template <typename, typename, typename> class Mglobal = Jacobi,
template <typename, typename, typename> class  Mlocal = Jacobi>
class CPR : public CPRBase<V> {
public:

  std::shared_ptr<CSR <R,C,V>> A;                    // system matrix
  std::shared_ptr<CSR <R,C,V>> Ap;                   // pressure matrix
   
  std::shared_ptr<Mglobal<R,C,V>> Fglo;              // global preconditioner
  std::shared_ptr<Mlocal <R,C,V>> Floc;              // local  preconditioner
  std::shared_ptr<Jacobi<R,C,V>> FgloJ;              // global preconditioner
   
  std::function<void(V*, V*&)> precond_glo;          // ptr to apply global precond
  std::function<void(V*, V*&)> precond_loc;          // ptr to apply local  precond
  std::function<void(V*, V*&)> precond_jac;          // ptr to apply jacobi  precond
   
  R   n;                                             // full system size
  R   sn;                                            // number of cells (n/bs)
  int bs;                                            // block size
  int p;                                             // pressure index
  V*  w = nullptr;                                   // weights
  bool* w_drs = nullptr;                             // drs_weights
  bool* check_diag = nullptr;

  V* xp = nullptr; V* rt = nullptr;                  // aux variables for apply prec
  V* J = nullptr;

  const boost::property_tree::ptree& param;
  // ===========================================================================

  CPR(const boost::property_tree::ptree& pt_,std::shared_ptr<CSR<R,C,V>> A_, V*& b)
    : param(pt_), A(A_) {

    _MPI_ENV;

    profiler.start("decoupling");
    
    bs = param.get<int>("block_size");
    p  = param.get<int>("pressure_id");
    
    if      (param.get<int>("seq") == 0) this->apply = apply_seq0();
    else if (param.get<int>("seq") == 1) this->apply = apply_seq1();
    else if (param.get<int>("seq") == 2) this->apply = apply_seq2();
    else if (param.get<int>("seq") == 3) this->apply = apply_seq3();
    else {
      std::cerr << "error: unknown sequence, seq =  " << param.get<int>("seq") << std::endl;
      exit(EXIT_FAILURE);
    }

    auto wtype = param.get<std::string>("weights");
    auto use_CPD = param.get<int>("CPD");
    auto use_drs = param.get<int>("use_drs");
    auto e_dd = param.get<double>("drs.e_dd");
    auto e_ps = param.get<double>("drs.e_ps");

    n  = A->n;
    sn = n / bs;
    if (n % bs != 0) {
      std::cerr << "error: n is not divisible by bs, n = " << n << ", bs = " << bs << std::endl;
      exit(EXIT_FAILURE);
    }
    xp = memres<V>(sn);
    rt = memres<V>(n);

    check_diag = fix_diag(A, b, 1e-20, 1e-20);
    
    profiler.start("quasiimpes weights");
    if       (wtype == "quasiimpes") {
      w = quasi_IMPES(A, bs, p);
    }else if (wtype == "quasiimpes_GE"){
      w = quasi_IMPES_GE(A, bs, p);
    } else {
      if (wtype != "none"){
        throw std::invalid_argument("Unknown weights type " + wtype);
      }
    }
    
    if (wtype != "none"){
      // apply quasiimpes weights
      alg::diag_scale(n,w,A->r,A->v);
      alg::diag_scale(n,w,b);
    }
    profiler.stop();

    if (use_drs){
      profiler.start("DRS weights");
      w_drs = drs_weights(bs, e_dd, e_ps);
      if (wtype != "none") unscale_drs_weights(A, b, w_drs, w);
      profiler.stop();
    }

    profiler.start("IMPES");
    IMPES(A,w_drs);
    IMPES(b,w_drs);
    profiler.stop();

    if (use_CPD){
      profiler.start("CPD decoup");
      J = CPD_build(A,bs);
      profiler.stop();
      profiler.start("CPD apply");
      CPD_apply(J,A,bs);
      CPD_apply(J,b,bs);
      profiler.stop();
    }

    double tol_zero = param.get<double>("tol_zero");
    if (tol_zero != 0.0) {
      profiler.start("filter A");
      A->filter(tol_zero); // remove zero entries
      profiler.stop();
    }

    profiler.start("get local matrix");
    // Ap = A->local_matrix(bs,p,w);
    Ap = A->local_matrix(bs,p,nullptr);
    profiler.stop();

    double tol_Ap = param.get<double>("prec_loc.tol_rel");
    if (tol_Ap > 1e-20){
      profiler.start("filter Ap");
      Ap->filter_proper(tol_Ap);
      profiler.stop();
    // Ap->filter(1e-13); // absolute filtering    
    }
    
    profiler.stop();

    MPI_Barrier(deco_comm);
    profiler.start("Global precond");
    Fglo = std::make_shared<Mglobal<R,C,V>>(param.get_child("prec_glo"), A);
    MPI_Barrier(deco_comm);
    profiler.stop();

    MPI_Barrier(deco_comm);
    profiler.start("Local  precond");
    Floc = std::make_shared<Mlocal <R,C,V>>(param.get_child("prec_loc"), Ap);
    MPI_Barrier(deco_comm);
    profiler.stop();

    profiler.start("decoupling");
    
    if      (param.get<int>("seq") == 2){ 
      if (Fglo->name == "Jacobi"){
        precond_jac = Fglo->apply;
      } else{
        profiler.start("Jacobi precond");
        FgloJ = std::make_shared<Jacobi<R,C,V>>(param.get_child("prec_glo"), A);
        precond_jac = FgloJ->apply;
        profiler.stop();
      }
    }

    precond_loc = Floc->apply;
    precond_glo = Fglo->apply;

    // undo the fix of diagonal
    fix_diag(A, check_diag);  // you should fix Ap as well if it is changed!
    profiler.stop();

  }
  
private:
  // first local prec then global prec
  std::function<void(V*,V*&)> apply_seq1() {
    return [this](V* r,V*& x) {
      #pragma omp parallel for schedule(static,bs)
      for (int i = 0; i < sn; ++i){
        rt[i] = r[i*bs+p];// / w[i*bs+p];
      }
      precond_loc(rt,xp);                                // xp = precond_loc(rt);

      // works with multiple MPI ranks                   // rt = r - A*x;
      std::memset(x, 0.0, (n)*sizeof(V));
      #pragma omp parallel for schedule(static,bs)
      for (int i = 0; i < sn; ++i){
        x[i*bs+p] = xp[i];
      }

      A->apply(x,rt);                                    // rt = r - A*x;
      #pragma omp parallel for
      for (int i = 0; i < n; ++i){
        rt[i] = r[i] - rt[i];
      }

      precond_glo(rt,x);                                 // x = precond_glob(rt);
      alg::axpby(sn,xp,adapter::block_array<V>(p,bs,x)); // x(p:bs:end) += xp;
    };
  }

  std::function<void(V*,V*&)> apply_seq2() {
    return [this](V* r,V*& x) {
      // precond_glo(r,x);                               // x = precond_glob(r);
      precond_jac(r,x);

      A->apply(x,rt,bs);                                 // rt = r - A*x;

      #pragma omp parallel for schedule(static,bs)
      for(int i=0;i<sn;++i){
        rt[i] = (r[i*bs+p] - rt[i]);// / w[i*bs+p];      // for quasi-IMPES_GE
      }
      
      precond_loc(rt,xp);                                // xp = precond_loc(rt);
      #pragma omp parallel for schedule(static,bs)
      for(int i=0;i<sn;++i){
        x[i*bs+p] += xp[i];
      }

      A->apply(x,rt);                                    // rt = r - A*x;
      #pragma omp parallel for
      for (int i = 0; i < n; ++i){
        rt[i] = r[i] - rt[i];
      }

      precond_glo(rt,rt);

      #pragma omp parallel for 
      for (C i = 0; i < n; ++i) {
        x[i] += rt[i];
      }
    };
  }

  std::function<void(V* __restrict__,V*& __restrict__)> apply_seq3() {
    return [this](V* __restrict__ r,V*& __restrict__ x) {
      
      precond_glo(r,x);                                 // x = precond_glob(r);
      // precond_jac(r,x);
      profiler.start("CPR_rest");
      A->apply(x,rt,bs);                                // rt = r - A*x;

      #pragma omp parallel for schedule(static,bs)
      for(int i=0;i<sn;++i){
        rt[i] = (r[i*bs+p] - rt[i]);// / w[i*bs+p];     // for quasi-IMPES_GE
      }
      profiler.stop();

      
      precond_loc(rt,xp);                               // xp = precond_loc(rt);

      profiler.start("CPR_rest");
      #pragma omp parallel for schedule(static,bs)
      for(int i=0;i<sn;++i){
        x[i*bs+p] += xp[i];
      }

      A->apply(x,rt);                                   // rt = r - A*x;
      #pragma omp parallel for
      for (int i = 0; i < n; ++i){
        rt[i] = r[i] - rt[i];
      }
      profiler.stop();

      precond_glo(rt,rt);

      profiler.start("CPR_rest");
      #pragma omp parallel for 
      for (C i = 0; i < n; ++i) {
        x[i] += rt[i];
      }
      profiler.stop();
    };
  }


  std::function<void(V* __restrict__,V*& __restrict__)> apply_seq0() {
    return [this](V* __restrict__ r,V*& __restrict__ x) {
      precond_glo(r,x);                                 // x = precond_glob(r);

      profiler.start("CPR_rest");
      A->apply(x,rt,bs);                                // rt = r - A*x;

      #pragma omp parallel for schedule(static,bs)
      for(int i=0;i<sn;++i){
        rt[i] = (r[i*bs+p] - rt[i]);// / w[i*bs+p];     // for quasi-IMPES_GE
      }
      profiler.stop();
      
      precond_loc(rt,xp);                               // xp = precond_loc(rt);
      profiler.start("CPR_rest");
      #pragma omp parallel for schedule(static,bs)
      for(int i=0;i<sn;++i){
        x[i*bs+p] += xp[i];
      }
      profiler.stop();
      
    };
  }

    std::function<void(const V*,V*&)> dummy() {
    return [this](const V* x,V*& b) {
      memcopy(b, x, this->n);
    };
  }

  // ===========================================================================
public:
  ~CPR() {
    free_memory(w);
    free_memory(w_drs);
    free_memory(xp);
    free_memory(rt);
    free_memory(check_diag);
    free_memory(J);
  }

private:
  void setup(shared_ptr<CSR<R,C,V>> A);

// compute quasi IMPES weights for a *sorted* CSR matrix A with block size bs and the pressure entry p_id
  V* quasi_IMPES(shared_ptr<CSR<R,C,V>> A, int bs,int p_id){
    
    _MPI_ENV;
    auto  n = A->n; R sn = n / bs; // (# of cells) 
    auto* r = A->r;
    auto* c = A->c;
    auto* v = A->v;

    auto shift_nr = A->shift_nr;

    auto* rhs = memput<V>(n,0.0);

#pragma omp parallel for
    for (C s = 0; s < sn; ++s){
      rhs[s*bs+p_id] = 1.0;
    }
#pragma omp parallel
    {
      int info;
      int ipiv[bs]; // permutations array
      V Ap[bs*bs]; // local block diagonal matrix
#pragma omp for // partition cells continously
      for (C s = 0; s < sn; ++s){ // loop over cells
        std::memset(Ap, 0, sizeof(V) * bs * bs);
        for (int p = 0; p < bs; ++p){ // loop the cell's entries
          C i = s*bs+p; // global A row index
          auto ia = r[i];
          auto ib = r[i+1];
          // assuming that c is sorted!
          auto j = core::binarysearch(c,ia,ib,i+shift_nr);
          if (c[j] == i+shift_nr){ // find the diagonal (this check can be skipped since there is always a diagonal entry)
            for (int k = 0; k < bs; ++k){ // loop over block columns
              // if(c[j+k-p] == i+k-p) { // check if other columns are nonzero
              if ( (c[j + k - p]-shift_nr) / bs == s){ // check if col entry is in the diagonal cell
                if (j+k-p < ib && j+k-p >= ia){ // check if you are still in the same row
                  Ap[p*bs + (c[j+k-p]-shift_nr)%bs] = v[j+k-p];
                } 
              }
            }
          }
        }
        // matrix must be transposed by definition -> COL_MAJOR
        info = CHECK(LAPACKE_dgesv(LAPACK_COL_MAJOR,bs,1,Ap,bs,ipiv,rhs + s*bs,bs));

      }
    }
    return rhs;
  }

  // compute quasi IMPES GE weights for a *sorted* CSR matrix A with block size bs and the pressure entry p_id
  V* quasi_IMPES_GE(const shared_ptr<CSR<R,C,V>> A, const int bs,const int p_id){
    
    _MPI_ENV;
    const auto  n = A->n; const R sn = n / bs; // (# of cells) 
    const auto* r = A->r;
    const auto* c = A->c;
    const auto* v = A->v;

    const int bs1 = bs-1;

    const auto shift_nr = A->shift_nr;

    auto* rhs = memres<V>(n);

#pragma omp parallel for
    for (C s = 0; s < sn; ++s){
      rhs[s*bs+p_id] = 1.0;
    }
#pragma omp parallel
    {
      V Ap[bs1*bs1]; // local block diagonal matrix
      V Ainv[bs1*bs1];
      V bp[bs1];
#pragma omp for // partition cells continously
      for (C s = 0; s < sn; ++s){ // loop over cells
        for (int p = 0; p < bs; ++p){ // loop the cell's entries
          C i = s*bs+p; // global A row index
          auto ia = r[i];
          auto ib = r[i+1];
          // assuming that c is sorted!
          auto j = core::binarysearch(c,ia,ib,i+shift_nr);
          #pragma unroll
          for (int k = 1; k < bs; ++k){ // loop over block columns, skip the pressure entry
            if (p == p_id) {
              bp[k-1] = -v[j+k-p];
            }else{
              Ap[(p-1)*bs1 + (c[j+k-p]-shift_nr)%bs-1] = v[j+k-p];
            }
          }
        }

        if (bs1 == 2){
          const V det = Ap[0]*Ap[3] - Ap[1]*Ap[2];

          if (det == 0.0) {
              std::cerr << "determinant is zero in CPD decoupling" << std::endl;
              std::exit(EXIT_FAILURE);
          }

          const V invdet = 1.0 / det;

          Ainv[0] =  Ap[3] * invdet;
          Ainv[1] = -Ap[1] * invdet;

          Ainv[2] = -Ap[2] * invdet;
          Ainv[3] =  Ap[0] * invdet;

          rhs[s*bs+1] = Ainv[0] * bp[0] + Ainv[2] * bp[1];
          rhs[s*bs+2] = Ainv[1] * bp[0] + Ainv[3] * bp[1];
        }else if (bs1==1){
          if (Ap[0] == 0.0) {
            std::cerr << "determinant is zero in CPD decoupling" << std::endl;
            std::exit(EXIT_FAILURE);
          }
          rhs[s*bs+1] = bp[0] / Ap[0];
        } else {
          std::cerr << "unknown block size CPD decoupling" << std::endl;
          std::exit(EXIT_FAILURE);
        }
        
        // matrix must be transposed by definition -> COL_MAJOR
        // info = CHECK(LAPACKE_dgesv(LAPACK_COL_MAJOR,bs1,1,Ap,bs1,ipiv,rhs + s*bs+1,bs1));
      }
    }
    return rhs;
  }

// summation of pressure aligned parts
void IMPES(V* x, bool* w = nullptr, int p = 0) {

  if (w == nullptr){
#pragma omp parallel for
    for (C s = 0; s < sn; ++s) {
      V sum = 0.0;
      for (int j = 0; j < bs; ++j) {
        sum += x[s*bs + j];
      }
      x[s*bs + p] = sum;
    }
  } else {
#pragma omp parallel for
    for (C s = 0; s < sn; ++s) {
      V sum = 0.0;
      for (int j = 0; j < bs; ++j) {
        sum += w[s*bs + j] * x[s*bs + j];
      }
      x[s*bs + p] = sum;
    }
  }
}


// scaled summation of pressure aligned parts
void IMPES(shared_ptr<CSR<R,C,V>> A, bool* weights = nullptr, int p = 0) {

  auto* r = A->r;
  auto* c = A->c;
  auto* v = A->v;

  if (weights == nullptr){
#pragma omp parallel for
    for (C s = 0; s < sn; ++s){
      V* table = v + r[s*bs+p];
      for (int k = 0; k < bs; ++k) {
        if (k == p) continue;
        R i = s*bs + k;
        auto ia = r[i];
        auto ib = r[i+1];
        for (auto j = ia; j < ib; ++j){
          table[j-ia] += v[j];
        }
      }
    }
  } else {
#pragma omp parallel for
    for (C s = 0; s < sn; ++s){
      V* table = v + r[s*bs+p];
      for (int k = 0; k < bs; ++k) {
        R i = s*bs + k;
        auto ia = r[i];
        auto ib = r[i+1];
        auto w = weights[i];
        if (k == p){
          for (auto j = ia; j < ib; ++j){
            v[j]*=w;
          }
        } else {
          for (auto j = ia; j < ib; ++j){
            table[j-ia] += w*v[j];
          }
        }
      }
    }
  }
}

bool* drs_weights(const int bs, const V eps_dd = 0.2, const V eps_ps = 0.0){

  // A matrix
  const auto* __restrict__ ra = A->r;
  const auto* __restrict__ ca = A->c;
  const auto* __restrict__ va = A->v;

  const auto shift_nr = A->shift_nr;

  auto* __restrict__ w = memres<bool>(n);
  
  // a_{x,1}^{ii} < eps_dd * sum_{j=1:sn,j!=i}|a_{x,1}^{ij}|  
#pragma omp parallel for
  for(C s=0;s<sn;++s){
    #pragma unroll
    for(int p=0;p<bs;++p){
      const C i = s*bs+p;
      const auto ia = ra[i];
      const auto ib = ra[i+1];
      // pressure column diagonal entry
      const auto diag = va[core::binarysearch(ca,ia,ib,i+shift_nr-p)]; // -p to account for the 0th column only for each cell

      // sum off-diagonal pressure column entries (only 0th column)
      auto off_diag = -abs(diag);
      for (auto j=ia;j<ib;++j){
        off_diag += (ca[j]%bs==0) ? abs(va[j]) : 0.;
      }
      w[i] = (diag < eps_dd * off_diag) ? false : true;
    }
  }
  
  if (eps_ps != 0.0){
    // sum_{j=1:sn}|a_{1,x}^{ij}| < eps_ps * a_{1,1}^{ii}
#pragma omp parallel for
    for(C s=0;s<sn;++s){
      C i = s*bs;
      auto ia = ra[i];
      auto ib = ra[i+1];
      // pressure pp diagonal entry
      auto j_diag = core::binarysearch(ca,ia,ib,i+shift_nr);
      auto diag = abs(va[j_diag]);

      // sum off-diagonal pressure column entries (only 0th column)
      V __restrict__ off_diag[bs] = {0.0};
      #pragma unroll
      for(int p=0;p<bs;++p) off_diag[p] = 0.0;

      for (auto j=ia;j<ib;++j){
        if (j == j_diag) continue;
        int p = ca[j]%bs; 
        off_diag[p] += abs(va[j]);
      }

      #pragma unroll
      for(int p=0;p<bs;++p){
        bool ok_ps = (off_diag[p] < eps_ps * diag) ? false : true;
        w[i+p] &= ok_ps; 
      }
    }
  }

  for(C s=0;s<sn;++s){
    w[s*bs] = true;
  }

  return w;
}

void unscale_drs_weights(shared_ptr<CSR<R,C,V>> A, V *b, bool *w_drs, V *w){
#pragma omp parallel for
  for (C i = 0; i < n; ++i){
    if (!w_drs[i]){
      auto ia = A->r[i];
      auto ib = A->r[i+1];
      auto d  = w[i];
      for (R j = ia; j < ib; ++j) {
        A->v[j] /= d;
      }
      b[i] /= d;
      w[i] = 1.;
    } 
  }
}

  // Constrained pressure decoupling
  inline V* CPD_build(shared_ptr<CSR<R,C,V>> A, const int bs){

    auto* vj = memres<V>(sn*bs*bs);

    // A matrix
    const auto* r = A->r;
    const auto* c = A->c;
    const auto* v = A->v;
    const auto shift_nr = A->shift_nr;

    // compute the values
    #pragma omp parallel
      {
        V Ap[bs * bs]; // local block diagonal matrix
        V Dp[bs];     // local diagonal matrix  
    #pragma omp for // partition cells continously
        for (C s = 0; s < sn; ++s) { // loop over cells
          #pragma unroll
          for (int p = 0; p < bs; ++p) { // loop the cell's entries
            const R i = s * bs + p; // global A row index
            const auto ia = r[i];
            const auto ib = r[i + 1];
            // assuming that c is sorted!
            const auto j = core::binarysearch(c,ia,ib,i+shift_nr);
            Dp[p] = v[j];
            #pragma unroll
            for (int k = 0; k < bs; ++k) { // loop over block columns
              Ap[p * bs + (c[j + k - p]-shift_nr)%bs] = v[j + k - p];
            }  
          }

          V* vj_loc = vj + s*bs*bs;

          if (bs == 3){
            // compute determinant
            const V det =
              Ap[0]*(Ap[4]*Ap[8] - Ap[5]*Ap[7])
            - Ap[3]*(Ap[1]*Ap[8] - Ap[2]*Ap[7])
            + Ap[6]*(Ap[1]*Ap[5] - Ap[2]*Ap[4]);

              if (det == 0.0) {
                  std::cerr << "determinant is zero in CPD decoupling" << std::endl;
                  std::exit(EXIT_FAILURE);
              }

              const V invdet = 1.0 / det;

              // scatter and diag scale the local inverse dense matrix into the global sparse matrix
              vj_loc[0] =  (Ap[4]*Ap[8] - Ap[5]*Ap[7]) * invdet * Dp[0];
              vj_loc[1] = -(Ap[1]*Ap[8] - Ap[2]*Ap[7]) * invdet * Dp[0];
              vj_loc[2] =  (Ap[1]*Ap[5] - Ap[2]*Ap[4]) * invdet * Dp[0];

              vj_loc[3] = -(Ap[3]*Ap[8] - Ap[5]*Ap[6]) * invdet * Dp[1];
              vj_loc[4] =  (Ap[0]*Ap[8] - Ap[2]*Ap[6]) * invdet * Dp[1];
              vj_loc[5] = -(Ap[0]*Ap[5] - Ap[2]*Ap[3]) * invdet * Dp[1];

              vj_loc[6] =  (Ap[3]*Ap[7] - Ap[4]*Ap[6]) * invdet * Dp[2];
              vj_loc[7] = -(Ap[0]*Ap[7] - Ap[1]*Ap[6]) * invdet * Dp[2];
              vj_loc[8] =  (Ap[0]*Ap[4] - Ap[1]*Ap[3]) * invdet * Dp[2];
          } else if (bs == 2){
            const V det = Ap[0]*Ap[3] - Ap[1]*Ap[2];

            if (det == 0.0) {
                std::cerr << "determinant is zero in CPD decoupling" << std::endl;
                std::exit(EXIT_FAILURE);
            }

            const V invdet = 1.0 / det;

            vj_loc[0] =  Ap[3] * invdet * Dp[0];
            vj_loc[1] = -Ap[1] * invdet * Dp[0];

            vj_loc[2] = -Ap[2] * invdet * Dp[1];
            vj_loc[3] =  Ap[0] * invdet * Dp[1];

          } else {
            std::cerr << "unknown block size CPD decoupling" << std::endl;
            std::exit(EXIT_FAILURE);
          }
        }
      }
    return vj;
}

// apply CPD to A csr matrix
inline void CPD_apply(const V* __restrict__ vj, shared_ptr<CSR<R,C,V>> A, const int bs) {
  
  auto* __restrict__ scr = memres<V>(A->nz);
  memcopy(scr, A->v, A->nz);

  const auto* __restrict__ ra = A->r;
  const auto* __restrict__ ca = A->c;
  auto* __restrict__ va = A->v;

  #pragma omp parallel for
  for (C s = 0; s < sn; ++s){
    #pragma unroll
    for (int p = 0; p < bs; ++p) {
      const C i = s*bs + p;
      const R ind = s*bs*bs+p*bs;
      V val = vj[ind+p];
      const auto ia = ra[i];
      const auto ib = ra[i+1];
      V* __restrict__ table = va + ia;
      for (auto j = ia; j < ib; ++j){
        va[j]*=val; 
      }
      #pragma unroll
      for (int k = 0; k < bs; ++k) {
        if (k==p) continue;
        V val = vj[ind+k];
        const C i = s*bs + k;
        const auto ia = ra[i];
        const auto ib = ra[i+1];
        for (auto j = ia; j < ib; ++j){
          table[j-ia] += scr[j]*val; 
        }
        
      }
    }
  }
  free(scr);
  
}


// apply CPD to right hand side b dense vector
void CPD_apply(V* vJ, V*& x, const int bs) {
  
#pragma omp parallel for
  for (C s = 0; s < sn; ++s) {
    V sum[bs];
    #pragma unroll
    for (int p = 0; p < bs; ++p) { 
      R i = s * bs + p; // global vJ row index
      sum[p] = 0.0;
      #pragma unroll
      for (int j = 0; j < bs; ++j) { 
        sum[p] += vJ[i * bs + j] * x[s*bs+j];
      }
    }
    #pragma unroll
    for (int p = 0; p < bs; ++p) { 
       x[s * bs + p] = sum[p];
    }
  }

}

// fix diagonal with reordering to strengthen the diagonal part
bool* fix_diag(shared_ptr<CSR<R,C,V>> A, V* b, V small_tol = 1e-20, V big_tol = 1e+12, int p_id = 0) {
  auto* check_diag_ = memput<bool>(A->n,false);
  auto* Ar = A->r;
  auto* Ac = A->c;
  auto* Av = A->v;
  auto  Ashift = A->shift_nr;
  int count = 0;
  int count2 = 0;

  V scratch[5000]; ///!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  #pragma omp parallel for private(scratch)
  for (int s = 0; s < sn; ++s){
    int i = s*bs;
    auto j0 = core::binarysearch(Ac,Ar[i],Ar[i+1],i+Ashift);
    if (abs(Av[j0]) < small_tol) {
      printf("there is zero pressure entry!\n");
      Av[j0] = (Av[j0] > 0) ? small_tol : -small_tol; // with sign
      check_diag_[i] = true;
      count++;
    }
    if (bs ==3){
      auto j1 = core::binarysearch(Ac,Ar[i+1],Ar[i+1+1],i+1+Ashift);
      auto j2 = core::binarysearch(Ac,Ar[i+2],Ar[i+2+1],i+2+Ashift);
      if (abs(Av[j1]) < small_tol || abs(Av[j2]) < small_tol ) {
        if (abs(Av[j1+1]) > small_tol || abs(Av[j2-1]) > small_tol ) {
      
          // permute rows of A and b
          auto ia1 = Ar[i+1];
          auto ib1 = Ar[i+1+1];
          auto ib2 = Ar[i+2+1];
          for (int j = ia1,k=0; j < ib1; ++j,++k){
            scratch[k] = Av[j]; 
          }
          for (int j = ib1,k=ia1; j < ib2, k < ib1; ++j,++k){
            Av[k] = Av[j]; 
          }
          for (int j = ib1,k=0; j < ib2; ++j,++k){
            Av[j] = scratch[k]; 
          }
          *scratch = b[i+1];
          b[i+1] = b[i+2];
          b[i+2] = *scratch;
        } else{
          if (abs(Av[j1]) < small_tol) {
            Av[j1] = (Av[j1] > 0) ? big_tol : -big_tol; // with sign
            check_diag_[i+1] = true;
            count++;
          }
          if (abs(Av[j2]) < small_tol) {
            Av[j2] = (Av[j2] > 0) ? big_tol : -big_tol; // with sign
            check_diag_[i+2] = true;
            count++;
          }
        }
      } 
      // else if (abs(Av[j1]) < abs(Av[j2-1]) && abs(Av[j2]) < abs(Av[j1+1]) ) {
      //   if (abs(Av[j1+1]) > small_tol || abs(Av[j2-1]) > small_tol ) {
      //     count2++;
      //     auto ia1 = Ar[i+1];
      //     auto ib1 = Ar[i+1+1];
      //     auto ib2 = Ar[i+2+1];
      //     for (int j = ia1,k=0; j < ib1; ++j,++k){
      //       scratch[k] = Av[j]; 
      //     }
      //     for (int j = ib1,k=ia1; j < ib2, k < ib1; ++j,++k){
      //       Av[k] = Av[j]; 
      //     }
      //     for (int j = ib1,k=0; j < ib2; ++j,++k){
      //       Av[j] = scratch[k]; 
      //     }
      //     *scratch = b[i+1];
      //     b[i+1] = b[i+2];
      //     b[i+2] = *scratch;
      //   }
      // }
    } else if (bs == 2){
      auto j1 = core::binarysearch(Ac,Ar[i+1],Ar[i+1+1],i+1+Ashift);
      if (abs(Av[j1]) < small_tol) {
        Av[j1] = (Av[j1] > 0) ? big_tol : -big_tol; // with sign
        check_diag_[i+1] = true;
        count++;
      }
      
    }
  }
  
  return check_diag_;
}

void fix_diag(shared_ptr<CSR<R,C,V>> A, bool *check_diag) {
  if (check_diag == nullptr) return;
  auto* Ar = A->r;
  auto* Ac = A->c;
  auto* Av = A->v;
  auto  Ashift = A->shift_nr;
#pragma omp parallel for
  for (int i = 0; i < A->n; ++i){
    if (check_diag[i] == true ) {
      auto j = core::binarysearch(Ac,Ar[i],Ar[i+1],i+Ashift);
      Av[j] = 0.;
    }
  }
}


};
// #include "CPR.tpp"