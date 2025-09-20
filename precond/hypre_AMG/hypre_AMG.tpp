// hypre set-up (see ex5.c)
template <typename R, typename C, typename V>
inline void hypre_AMG<R, C, V>::setup(std::shared_ptr<CSR<R, C, V>> A, const boost::property_tree::ptree param) {
    _MPI_ENV;
    MPI_Barrier(deco_comm);
    profiler.start("Hypre_CSR2IJMatrix");
    HYPRE_Init();

    // Convert CSR to Hypre IJMatrix
    HYPRE_IJMatrix A_ij = CSR2IJMatrix(A);

    // Determine local row bounds
    int ilower,iupper;
    if (nprocs == 1){
        ilower = 0;
        iupper = A->n - 1;
    }else{
      ilower = A->displs_nr[mid];
      iupper = A->displs_nr[mid + 1] - 1;
    }

    // Create vectors
    HYPRE_IJVector b_ij, x_ij;
    HYPRE_IJVectorCreate(deco_comm, ilower, iupper, &b_ij);
    HYPRE_IJVectorCreate(deco_comm, ilower, iupper, &x_ij);

    HYPRE_IJVectorSetObjectType(b_ij, HYPRE_PARCSR);
    HYPRE_IJVectorSetObjectType(x_ij, HYPRE_PARCSR);

    HYPRE_IJVectorInitialize(b_ij);
    HYPRE_IJVectorInitialize(x_ij);

    // Convert IJMatrix and IJVectors to ParCSR objects
    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_ParVector par_b, par_x;

    HYPRE_IJMatrixGetObject(A_ij, (void**)&parcsr_A);
    HYPRE_IJVectorGetObject(b_ij, (void**)&par_b);
    HYPRE_IJVectorGetObject(x_ij, (void**)&par_x);

    MPI_Barrier(deco_comm);
    profiler.stop();

    // Setup AMG preconditioner
    HYPRE_Solver P;
    HYPRE_BoomerAMGCreate(&P);
    // HYPRE_BoomerAMGSetPrintLevel(P, 1);  // Print AMG setup info
    HYPRE_BoomerAMGSetDebugFlag(P, param.get<int>("verbosity"));   // debug
    HYPRE_BoomerAMGSetCoarsenType(P, param.get<int>("amg_coarse_type")); // HMIS (10) -- default, PMIS (8) is less accurate but more scalable
    // HYPRE_BoomerAMGSetInterpType(P, 6);  // EXTI+I interpolation (6 by default)
    int relax = param.get<int>("amg_relax_type");
    if (relax > 0) HYPRE_BoomerAMGSetRelaxType(P, relax);  // FCF-Jacobi relaxation (17 - by default, 0 - Jacobi)

    // aggresive coarsening for larger models
    HYPRE_BoomerAMGSetAggNumLevels(P,param.get<int>("amg_agg_levs")); 
    HYPRE_BoomerAMGSetNumPaths(P, param.get<int>("amg_agg_path"));
    HYPRE_BoomerAMGSetCycleType(P,1); // V-cycle

    HYPRE_BoomerAMGSetADropTol(P,param.get<double>("amg_A_drop_tol")); // drop tolerance  (at most 1e-4 for sleipner)
    HYPRE_BoomerAMGSetTruncFactor(P,param.get<double>("amg_prol_trunc_fac")); // used 1e-3
    // HYPRE_BoomerAMGSetMaxLevels(P,10);

    // HYPRE_BoomerAMGSetRelaxOrder(P, 1);
    HYPRE_BoomerAMGSetMaxCoarseSize(P, 300);
    HYPRE_BoomerAMGSetNumSweeps(P, 1);
    HYPRE_BoomerAMGSetStrongThreshold(P, param.get<double>("amg_SoC_alpha")); // 0.4-0.6
    HYPRE_BoomerAMGSetMaxIter(P, 1);     // do only one iteration

    HYPRE_BoomerAMGSetup(P, parcsr_A, par_x, par_b);
    // HYPRE_BoomerAMGSetup(P, parcsr_A, NULL, NULL);

    // Store in the class
    this->x_ij = x_ij;
    this->b_ij = b_ij;
    this->A_ij = A_ij;
    this->par_x = par_x;
    this->par_b = par_b;
    this->parcsr_A = parcsr_A;
    this->P = P;
}


template <typename R, typename C, typename V>
HYPRE_IJMatrix CSR2IJMatrix(std::shared_ptr<CSR<R, C, V>> A_csr) {
    _MPI_ENV;
    HYPRE_IJMatrix A_ij;

    // Determine local row bounds
    int ilower,iupper;
    if (nprocs == 1){
        ilower = 0;
        iupper = A_csr->n - 1;
    }else{
      ilower = A_csr->displs_nr[mid];
      iupper = A_csr->displs_nr[mid + 1] - 1;
    }

    // Create the IJMatrix
    HYPRE_IJMatrixCreate(deco_comm, ilower, iupper, ilower, iupper, &A_ij);
    HYPRE_IJMatrixSetObjectType(A_ij, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A_ij);

    // Populate the matrix
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < A_csr->n; i++) {
      int ii = i + ilower;
      int offset = static_cast<int>(A_csr->r[i]);
      int nnz = static_cast<int>(A_csr->r[i + 1] - A_csr->r[i]);
      int* cols = A_csr->c + offset;
      double* vals = A_csr->v + offset;
      HYPRE_IJMatrixSetValues(A_ij, 1, &nnz, &ii, cols, vals);
    }

    // Assemble the IJMatrix
    HYPRE_IJMatrixAssemble(A_ij);

    return A_ij;
}