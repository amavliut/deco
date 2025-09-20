template <typename R,typename C,typename V>
inline void SOL<R,C,V>::lgmres() {
  using namespace alg;
  _MPI_ENV;
  // Input parameters
  V tol = param.get<V>("reltol");             // Relative tolerance
  int maxRestart = param.get<int>("restart"); // Maximum number of restarts
  int maxIter = param.get<int>("itmax");      // Maximum iterations
  int verbosity = param.get<int>("verbosity");

  auto* rhs = b;
  auto n = A->n;

  double bnorm,absTol,rnorm;

  // Allocate memory and initialize pointers
  double* r,* z,* w,* ksp_,* H_,* g,* c,* s;
  V* resvec = memres<double>(maxIter);
  allocate_memory(n,maxRestart,&r,&z,&w,&ksp_,&H_,&g,&c,&s);

  // initialize dense matrices for easier handling
  adapter::matrix<V> H((maxRestart + 1),maxRestart,H_);
  adapter::matrix<V> ksp(n,(maxRestart + 1),ksp_);

  flag = 1,iter = 0;

  // Initialize solution
  if (x == nullptr) {
    memcal(x,n);
    precond(b,r);
    // printf("norm of b = %e, norm of r = %e\n",MPI_l2norm(n,b,nprocs),MPI_l2norm(n,r,nprocs));
  } else {
    A->apply(x,z);                               // r = b - A*x
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      z[i] = b[i] - z[i];
    }
    precond(z,r);                                // apply preconditioner
  }

  bnorm = MPI_l2norm(n,r,nprocs);
  absTol = tol * bnorm;                          // relative tolerance

  while (iter < maxIter) {
    // initialize the subspace to zero
    meminit(H_,(maxRestart + 1) * (maxRestart + 3));

    g[0] = MPI_l2norm(n,r,nprocs);
    axpby(n,1 / g[0],r,&(ksp(0,0)));

    int j = 0;
    while (j < maxRestart && iter < maxIter) {

      rnorm = fabs(g[j]);
      resvec[iter] = rnorm / bnorm;

      // inner check convergence
      if (rnorm < absTol) {
        flag = 0; break;
      }

      A->apply(&ksp(0,j),z);                     // first compute new z
      precond(z,w);                              // then precondition

      // Orthogonalize w against Krylov subspace
      for (int i = 0; i <= j; ++i) {
        H(i,j) = MPI_dot(n,&ksp(0,i),w,nprocs);    // H(i,j) = ksp(:,i)' * w
        axpby(n,-H(i,j),&ksp(0,i),1.0,w);          // w = w - H(i,j)*ksp(:,i)
      }
      H(j + 1,j) = MPI_l2norm(n,w,nprocs);
      axpby(n,1 / H(j + 1,j),w,&ksp(0,j + 1));

      // Apply the previous Givens rotation to the new j column
      for (int i = 0; i <= j - 1; ++i) {
        applyGivensRotation(c[i],s[i],H(i,j),H(i + 1,j));
      }

      // Compute the new Givens rotations
      comptGivensRotation(H(j,j),H(j + 1,j),c[j],s[j]);

      // eliminate the jth column subdiagonal 
      applyGivensRotation(c[j],s[j],H(j,j),H(j + 1,j));
      applyGivensRotation(c[j],s[j],g[j],g[j + 1]);

      ++j; ++iter;
    }

    // backward substitution to solve g(1:j) = H(1:j,1:j)\g(1:j);
    for (int i = j - 1; i >= 0; --i) {
      if (H(i,i) == 0.0) {
        fprintf(stderr,"Error in rgmres: Zero diagonal element in upper triangular matrix.\n");
        std::exit(EXIT_FAILURE);
      }
      g[i] /= H(i,i);
      for (int k = 0; k < i; ++k) {
        g[k] -= H(k,i) * g[i];
      }
    }

    axpby(n,g[0],&ksp(0,0),w);
    for (int i = 1; i < j; ++i) {
      axpby(n,g[i],&ksp(0,i),1.0,w);
    }

    axpby(n,w,x);

    A->apply(x,z);                               // r = b - A*x
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      z[i] = b[i] - z[i];
    }

    precond(z,r);                                // apply preconditioner

    // outer convergence check
    rnorm = MPI_l2norm(n,r,nprocs);
    if (rnorm < absTol) {
      flag = 0; break;
    }
  }

  res_abs = rnorm;
  res_rel = rnorm / bnorm;

  if (verbosity > 0 && mid == 0) {
    printf("\n--------------------------------\n");
    printf("Profile of lGMRES:\n");
    printf("iteration count =  %d\n",iter);
    printf("residual = %.2e\n",rnorm / bnorm);
  }

  free(r);
  free(resvec);
}