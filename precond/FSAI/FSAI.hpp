
template <typename R = int,typename C = int,typename V = double>
class FSAI {
public:
  std::string name = "FSAI";
  std::shared_ptr<CSR <R,C,V>> FL;
  std::shared_ptr<CSR <R,C,V>> FU;
  V* temp; // auxiliary array to apply preconditioner

  const boost::property_tree::ptree& param;
  std::function<void(const V* __restrict__, V*& __restrict__)> apply;
  // ===========================================================================

  FSAI(const boost::property_tree::ptree& pt_,std::shared_ptr<CSR<R,C,V>> A_)
    : param(pt_){
    
    std::shared_ptr<CSR <R,C,V>> A;

    double tol_rel = param.get<double>("tol_rel");
    double tol_abs = param.get<double>("tol_abs");

    if (tol_rel != 0.0 || tol_abs != 0.0){
      profiler.start("filter A");
      A = A_->copy();
      if (tol_rel!=0.0) A->filter_proper(tol_rel);
      if (tol_abs!=0.0) A->filter(tol_abs);
      profiler.stop();
    } else A = A_;

    FL = std::make_shared<CSR<R,C,V>>(A->n,A->m,0,nullptr,nullptr,nullptr);
    FU = std::make_shared<CSR<R,C,V>>(A->n,A->m,0,nullptr,nullptr,nullptr);

    temp = memres<V>(A->n);
    this->setup(A,param.get<int>("power"));
    apply = precond();

  }

  // b = P(x)
  std::function<void(const V* __restrict__, V*& __restrict__)> precond() {
    return [this](const V* __restrict__ x, V*& __restrict__ b) {
      profiler.start("FSAI apply");
      alg::spmv(FL->n,FL->r,FL->c,FL->v,x,temp);
      alg::spmv(FU->n,FU->r,FU->c,FU->v,temp,b);
      profiler.stop();
      };
  }

  // ===========================================================================
  ~FSAI() {
    free(temp);
  }

private:
  void setup(shared_ptr<CSR<R,C,V>> A,int k);
};
#include "FSAI.tpp"