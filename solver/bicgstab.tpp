template <typename R,typename C,typename V>
inline void SOL<R,C,V>::bicgstab() {
  using namespace alg;
  _MPI_ENV;
  
  auto n = A->n;
  V rho_old,rho = 1,alpha = 1,beta,delta,gamma,omega = 1,kappa; // scalars
  V* r = memres<V>(n);                            // residual  vector (s = r)
  V* rt = memres<V>(n);                           // initial residual
  V* p = memres<V>(n);                            // direction vector
  V* y = memres<V>(n);                            // auxiliary
  V* v = memres<V>(n);                            // auxiliary
  V* t = memres<V>(n);                            // auxiliary

  // input parameters
  V reltol = param.get< V >("reltol");            // exit tolerance
  V norm_b = MPI_l2norm(n,b,nprocs);              // norm of rhs
  V tol = norm_b * reltol;                        // relative residual
  int itmax = param.get<int>("itmax");            // max iterations
  float it = 0.0;                                 // iteration count
  flag = 1;
  V res_norm_old;

  // initilize values
  if (x == nullptr) {
    memcal(x,n);
    memcopy(r,b,n);
  } else {
    A->apply(x,r);                                                              // r = b - A*x
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      r[i] = b[i] - r[i];
    }
    printf("initial solution is nonzero: norm r = %e, norm b = %e, norm x = %e\n",MPI_l2norm(n,r,nprocs),MPI_l2norm(n,b,nprocs),MPI_l2norm(n,x,nprocs));
  }
  memcopy(rt,r,n);                                                              // rt = r

  V res_norm = MPI_l2norm(n,r,nprocs);

  // printf(" Iter           Defect\n");
  // printf("%8d %17.4f %17.4f\n",(int)it,res_norm,res_norm/norm_b);
  for (it = 0.5; it < itmax; it += .5) {

    rho_old = rho;
    rho = MPI_dot(n,r,rt,nprocs);                                               // rho_old = r·rr

    if (it < 1) {
      memcopy(p,r,n);                                                           // p = r
    } else {
      beta = (abs(res_norm) < 1e-40) ? .0 : (rho / rho_old) * (alpha / omega);  // beta = (rho/rho_old)*(alpha/omega)
      axpby(n,(V)1.0,r,-beta * omega,v,beta,p);                                 // p = r + beta*(p - omega*v); 
    }
    // --------------- BiCG step -----------------
    precond(p,y);
    A->apply(y,v);                                                              // v = A·P·p
    alpha = (abs(res_norm) < 1e-40) ? .0 : rho / MPI_dot(n,v,rt,nprocs);        // alpha = rho / v'*rt;
    axpby(n,-alpha,v,(V)1.0,r);                                                 // r = r - alpha*v
    axpby(n,alpha,y,(V)1.0,x);                                                  // x = x + alpha*y
    // -------------------------------------------------
    res_norm = MPI_l2norm(n,r,nprocs);
    if (res_norm <= tol) {
      flag = 0;
      break;
    }
    it += .5;
    // --------------- stabilization step --------------
    precond(r,y);
    A->apply(y,t);
    omega = (abs(res_norm) < 1e-40) ? .0 : MPI_dot(n,t,r,nprocs) / MPI_dot(n,t,t,nprocs); // omega = < t, r > / < t, t >
    axpby(n,-omega,t,(V)1.0,r);                                                           // r = r - omega*t;
    axpby(n,omega,y,(V)1.0,x);                                                            // x = x + omega*y;

    res_norm = MPI_l2norm(n,r,nprocs);
    if (res_norm <= tol) {
      flag = 0;
      break;
    }
    // printf("\n%8d %17.4f %17.4f\n",(int)it,res_norm,res_norm/norm_b);
  }

  iter = it;
  res_abs = res_norm;
  res_rel = res_norm / norm_b;
  // printf("\n%8d %17.4f %17.4f\n",(int)it,res_norm,res_norm/norm_b);

  if (param.get<int>("verbosity") > 0 && mid == 0) {
    printf("\n--------------------------------\n");
    printf("Profile of BICGstab:\n");
    printf("iteration count =  %.1f\n",it);
    printf("residual = %.2e\n",res_norm / norm_b);
  }

  free(r);
  free(rt);
  free(p);
  free(y);
  free(v);
  free(t);
}
