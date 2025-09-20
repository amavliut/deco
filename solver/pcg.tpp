// sequential over MPI version
template <typename R,typename C,typename V>
inline void SOL<R,C,V>::pcg() {
  using namespace alg;

  auto n = A->n;
  V rho1,rho2,alpha,beta;                      // scalars
  auto rvec = memres<V>(n);                    // residual  vector
  auto pvec = memres<V>(n);                    // direction vector
  auto zvec = memres<V>(n);                    // auxiliary
  auto gvec = memres<V>(n);                    // auxiliary
  
  // input parameters
  V reltol = param.get< V >("reltol");         // exit tolerance
  V norm_b = l2norm(n,b);                      // norm of rhs
  V tol = norm_b * reltol;                     // relative residual
  int itmax = param.get<int>("itmax");         // max iterations
  int it = 0;                                  // iteration count
  
  // initilize values
  if (x == nullptr) {
    memcal(x,n);                               // solution vector
    memcpy(rvec,b,n*sizeof(V));
  } else {
    A->apply(x,rvec);                             // r = b - Ax 
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      rvec[i] = b[i] - rvec[i];
    }
  }
  precond(rvec,pvec);                          // p = P·r
  rho1 = dot(n,rvec,pvec);                     // rho = r·p
  V res_norm = l2norm(n,rvec);                 // L2 norm of residual
  
  // iterate untill convergence or max iteration count
  while (res_norm > tol && it < itmax) {
    A->apply(pvec,zvec);                       // z = A·p
    alpha = rho1 / dot(n,zvec,pvec);           // alpha = (p·r)/(z·p)
    axpby(n, alpha,pvec,(V)1.0,x);             // x = x + alpha p
    axpby(n,-alpha,zvec,(V)1.0,rvec);          // r = r - alpha z
    precond(rvec,gvec);                        // g = P·r
    rho2 = dot(n,rvec,gvec);                   // rho2 = r·g
    beta = rho2 / rho1;                        // beta = rho2 / rho1
    rho1 = rho2;                               // reset rho
    axpby(n,(V)1.0,gvec,beta,pvec);            // p = g + beta pvec
    
    res_norm = l2norm(n,rvec);
    ++it;
  }

  if (param.get<int>("verbosity") > 0){
    printf("\n--------------------------------\n");
    printf("Profile of PCG:\n");
    printf("iteration count =  %d\n",it);
    printf("residual = %e\n",res_norm/norm_b);
  }

  free(rvec);
  free(pvec);
  free(zvec);
  free(gvec);
}
