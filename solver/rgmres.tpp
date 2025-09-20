#define EPSILON 2.220446049250312e-16 // Small threshold for comparisons
void applyGivensRotation(const double c,const double s,double& dx,double& dy);
void comptGivensRotation(const double x,const double y,double& c,double& s);
void allocate_memory(int n,int maxRestart,
  double** r,double** z,double** w,
  double** ksp_,double** H_,double** g,double** c,double** s);

template <typename R,typename C,typename V>
inline void SOL<R,C,V>::rgmres() {
  using namespace alg;
  _MPI_ENV;
  // Input parameters
  V tol = param.get<V>("reltol");             // Relative tolerance
  int maxRestart = param.get<int>("restart"); // Maximum number of restarts
  int maxIter = param.get<int>("itmax");      // Maximum iterations
  int verbosity = param.get<int>("verbosity");

  auto* rhs = b;
  auto n = A->n;

  // Allocate memory and initialize pointers
  double* r,* z,* w,* ksp_,* H_,* g,* c,* s;
  V* resvec = memres<double>(maxIter);
  allocate_memory(n,maxRestart,&r,&z,&w,&ksp_,&H_,&g,&c,&s);

  // initialize dense matrices for easier handling
  adapter::matrix<V> H((maxRestart + 1),maxRestart,H_);
  adapter::matrix<V> ksp(n,(maxRestart + 1),ksp_);

  flag = 1,iter = 0;

  // Initial calculations
  double bnorm = MPI_l2norm(n,rhs,nprocs);
  double absTol = tol * bnorm;                  // relative tolerance
  double rnorm;
  double val;

  // Initialize solution
  if (x == nullptr) {
    memcal(x,n);
    memcopy(r,b,n);
  } else {
    A->apply(x,r);                               // r = b - A*x
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      r[i] = b[i] - r[i];
    }
  }

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

      precond(&ksp(0,j),z);                             // apply preconditioner
      A->apply(z,w);                                    // compute new w

      // Orthogonalize w against Krylov subspace
      for (int i = 0; i <= j; ++i) {
        H(i,j) = MPI_dot(n,&ksp(0,i),w,nprocs);         // H(i,j) = ksp(:,i)' * w
        axpby(n,-H(i,j),&ksp(0,i),1.0,w);               // w = w - H(i,j)*ksp(:,i)
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

    precond(w,z);                                // apply preconditioner
    axpby(n,z,x);                                // compute new x

    A->apply(x,r);                               // r = b - A*x
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      r[i] = b[i] - r[i];
    }

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
    printf("Profile of rGMRES:\n");
    printf("iteration count =  %d\n",iter);
    printf("residual = %.2e\n",rnorm / bnorm);
  }

  free(r);
  free(resvec);
}

void comptGivensRotation(const double x,const double y,double& c,double& s) {
  if (fabs(y) < EPSILON) {
    c = 1.0;
    s = 0.0;
  } else if (fabs(y) > fabs(x)) {
    double nu = x / y;
    s = 1.0 / sqrt(1.0 + nu * nu);
    c = nu * s;
  } else {
    double nu = y / x;
    c = 1.0 / sqrt(1.0 + nu * nu);
    s = nu * c;
  }
}

void applyGivensRotation(const double c,const double s,double& dx,double& dy) {
  double temp = c * dx + s * dy;
  dy = -s * dx + c * dy;
  dx = temp;
}


void allocate_memory(int n,int maxRestart,
  double** r,double** z,double** w,
  double** ksp_,double** H_,double** g,double** c,double** s) {
  // Calculate total memory size required
  size_t total_size =
    n * 3 +      // r, z, w
    n * (maxRestart + 1) + // ksp_
    (maxRestart + 1) * maxRestart + // H_
    (maxRestart + 1) * 3; // g, c, s

  // Allocate all memory at once
  double* base = memres<double>(total_size);
  if (!base) {
    fprintf(stderr,"Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }

  // Assign pointers to respective sections
  *r = base;
  base += n;

  *z = base;
  base += n;

  *w = base;
  base += n;

  *ksp_ = base;
  base += n * (maxRestart + 1);

  *H_ = base;
  base += (maxRestart + 1) * maxRestart;

  *g = base;
  base += (maxRestart + 1);

  *c = base;
  base += (maxRestart + 1);

  *s = base;
  // No need to increment base further as it's the last pointer
}