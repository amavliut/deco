template <typename R = int,typename C = int,typename V = double>
class SOL {
public:

  const boost::property_tree::ptree& param;
  std::shared_ptr<CSR <R,C,V>> A;              // system matrix
  V* b;                                        // right hand side
  V* x;                                        // solution
  const std::function<void(V*,V*&)>& precond;  // pointer to apply preconditioner: precond(x,b) is b = precond(x)
  int iter;
  double res_abs;
  double res_rel;
  int flag;

  std::shared_ptr<CSR <R,C,V>> Aorg; // = nullptr;              // system matrix
  V* borg; //= nullptr;                                         // right hand side
  // ============================================================================================================

  SOL(const boost::property_tree::ptree& param_,
    std::shared_ptr<CSR <R,C,V>> A_,
    const std::function<void(V*,V*&)>& precond_,
    V* b_,V* x_ = nullptr,std::shared_ptr<CSR <R,C,V>> Aorg_ = nullptr,V* borg_ = nullptr)
    : param(param_),A(A_),b(b_),x(x_),precond(precond_),Aorg(Aorg_),borg(borg_) {

    auto type = param.get<std::string>("type");
    if (type == "rgmres") {             // right preconditioned gmres
      this->rgmres();
    } else if (type == "lgmres") {      // left preconditioned gmres
      this->lgmres();
    } else if (type == "bicgstab") {    // right preconditioned bicgstab
      this->bicgstab();
    } else if (type == "lbicgstab") {   // left preconditioned bicgstab
      this->lbicgstab();
    } else if (type == "pcg") {
      this->pcg();
    } else {
      throw std::invalid_argument("Invalid solver type: " + type);
    }
  }

  V* get_result() const {
    return x;
  }

  // ============================================================================================================
  ~SOL() {
    free(x);
    free(b);
  }

private:
  void pcg();
  void bicgstab();
  void lbicgstab();
  void rgmres();
  void lgmres();
};
#include "pcg.tpp"
#include "bicgstab.tpp"
#include "lbicgstab.tpp"
#include "rgmres.tpp"
#include "lgmres.tpp"