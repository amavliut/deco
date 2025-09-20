
template <typename R = int, typename C = int, typename V = double>
class Jacobi {
public:
    std::string name = "Jacobi";
    std::shared_ptr<CSR <R,C,V>> A;                   // system matrix
    V* J = nullptr;
    const boost::property_tree::ptree& param;
    int bs; // block size
    int sn; // number of cells 
    int n; // sn*bs 
    std::function<void(const V* __restrict__, V*& __restrict__)> apply;
    // ===========================================================================

    Jacobi(const boost::property_tree::ptree& pt_, std::shared_ptr<CSR<R, C, V>>& A_)
      : param(pt_), A(A_) {

        bs = param.get<int>("block_size");
        
        if (bs==1) this->classic(A);
        else this->blocked(A,bs);
        
        apply = (bs == 1) ? precond_classic() : precond_dense();
        
      }
    
    // b = J(x)
    std::function<void(const V* __restrict__, V*& __restrict__)> precond() {
      return [this](const V* __restrict__ x, V*& __restrict__ b) {
        precond_dense(x,b);
      };
    }

    std::function<void(const V*,V*&)> dummy() {
      return [this](const V* x,V*& b) {
        memcopy(b, x, this->n);
      };
    }

    std::function<void(const V*, V*&)> precond_classic() {
      return [this](const V* x, V*& b) {
        #pragma omp parallel for 
          for (C i = 0; i < n; ++i) {
            b[i] = J[i] * x[i];
          }
        };
      }
    
    std::function<void(const V* __restrict__, V*& __restrict__)> precond_dense() {
      return [this](const V* __restrict__ x, V*& __restrict__ b) {
          #pragma omp parallel
          {
            V __restrict__ sum[bs];
            #pragma omp for
            for (C s = 0; s < sn; ++s) {
              V* __restrict__ Ap = J + s*bs*bs;
              C sbs = s*bs;
              for (int p = 0; p < bs; ++p) { 
                sum[p] = Ap[p*bs] * x[sbs];
                for (int k = 1; k < bs; ++k) { 
                  sum[p] += Ap[p*bs+k] * x[sbs+k];
                }
                b[sbs+p] = sum[p];
              }
            }
          }
        };
      }

    // ===========================================================================
    ~Jacobi(){
      free_memory(J);
    }

private:
    void classic(shared_ptr<CSR<R,C,V>> A);
    void blocked(shared_ptr<CSR<R,C,V>> A, int bs);
};
#include "Jacobi.tpp"