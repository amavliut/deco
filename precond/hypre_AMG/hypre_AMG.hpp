template <typename R = int, typename C = int, typename V = double>
class hypre_AMG {
public:
    const boost::property_tree::ptree& param;
    R n;  // number of rows (local)
    HYPRE_Solver P;
    HYPRE_IJMatrix A_ij;
    HYPRE_IJVector b_ij, x_ij;
    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_ParVector par_b, par_x;

    std::function<void(const V*, V*&)> apply;
    std::function<void(const V*, V*&)> Axb;
    // ================================================================================
    
    hypre_AMG(const boost::property_tree::ptree& pt_, std::shared_ptr<CSR<R, C, V>>& A)
      : param(pt_),n(A->n) {
        
        this->setup(A,param);
        apply = precond();
      }

    // b = P(x)
    // x and b are inverted with respect to hypre notation
    std::function<void(const V*,V*&)> precond() {
      return [this](const V* x,V*& b) {
        profiler.start("AMG apply");
        memcal(b,this->n);
        HYPRE_IJVectorSetValues(this->x_ij, this->n, NULL, x); 
        HYPRE_IJVectorSetValues(this->b_ij, this->n, NULL, b);
        HYPRE_IJVectorAssemble (this->x_ij);
        HYPRE_IJVectorAssemble (this->b_ij);
        HYPRE_BoomerAMGSolve(this->P, this->parcsr_A, this->par_x, this->par_b);
        HYPRE_IJVectorGetValues(this->b_ij, this->n, NULL, b);
        profiler.stop();
      };
    }

    std::function<void(const V*,V*&)> Axb_() {
      return [this](const V* x,V*& b) {
        memcal(b,this->n);
        HYPRE_IJVectorSetValues(this->x_ij, this->n, NULL, x);
        HYPRE_IJVectorSetValues(this->b_ij, this->n, NULL, b);
        HYPRE_ParCSRMatrixMatvec(1.0, this->parcsr_A, this->par_x, 0.0, this->par_b);
        HYPRE_IJVectorGetValues(this->b_ij, this->n, NULL, b);
      };
    }

    // no preconditioner is applied
    std::function<void(const V*,V*&)> dummy() {
      return [this](const V* x,V*& b) {
        memcopy(b, x, this->n);
      };
    }
    
    // ================================================================================
    ~hypre_AMG(){
      HYPRE_IJMatrixDestroy (this->A_ij);
      HYPRE_IJVectorDestroy (this->b_ij);
      HYPRE_IJVectorDestroy (this->x_ij);
      HYPRE_BoomerAMGDestroy(this->P);
      HYPRE_Finalize();
    }

private:
    void setup(shared_ptr<CSR<R,C,V>> A, const boost::property_tree::ptree param);
};
#include "hypre_AMG.tpp"