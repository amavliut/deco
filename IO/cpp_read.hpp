using namespace std;

namespace IO {

  template <typename R,typename C>
  void find_row_ptr(R* r,const R nz,const C* p,const C shift = 0) {
    R row_old = p[0];
    r[0] = 0;
    for (R i = 0; i < nz; ) {
      R row = p[i];
      // fill empty rows
      while (row_old < row) {
        r[row_old - shift + 1] = i;
        row_old++;
      }
      // cumulative sum
      while (p[i] == row) {
        i++;
        if (i == nz) break; // for last value
      }
      r[row - shift + 1] = i;
      row_old = row; // reset the old row
    }
  }

  long find_length(std::ifstream& file) {
    file.seekg(0,std::ios::end); // Move the file pointer to the end
    std::streampos fileSize = file.tellg();
    file.seekg(0,std::ios::beg); // Move back to the start of the file
    long lineCount = 0;
    char c;
    for (long i = 0; i < fileSize; ++i) {
      file.get(c); // Read character by character
      if (c == '\n') {
        ++lineCount;
      }
    }
    return lineCount;
  }

  
  //////////////////////////////////////////////////////
  /*                     ascii IO                     */
  //////////////////////////////////////////////////////


  template <typename R,typename C,typename V>
  shared_ptr<CSR<R,C,V>> cpp_read_ascii_matrix_coo(const std::string filename) {

    R n,m,nz;

    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << filename << std::endl;
      exit(EXIT_FAILURE);
    }

    // Read the dimensions and number of non-zero elements
    file >> n >> m >> nz;
    std::cout << "matrix size:\t";
    std::cout << "n = " << n << ", m = " << m << ", nz = " << nz << "\n";

    // allocate memory for the sparse matrix 
    C* p = CHECK((C*)malloc((nz) * sizeof(C)));
    C* c = CHECK((C*)malloc((nz) * sizeof(C)));
    V* v = CHECK((V*)malloc((nz) * sizeof(V)));

    // Read the non-zero elements
    for (R i = 0; i < nz; ++i) {
      file >> p[i] >> c[i] >> v[i];
      p[i] -= 1;
      c[i] -= 1;
    }
    file.close();

    auto A = std::make_shared<CSR<R,C,V>>(n,m,nz,p,c,v);

    return A;
  }

  template <typename R,typename C,typename V>
  shared_ptr<CSR<R,C,V>> cpp_read_ascii_matrix(const std::string filename) {

    R n,m,nz;

    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << filename << std::endl;
      exit(EXIT_FAILURE);
    }

    // Read the dimensions and number of non-zero elements
    std::string line;
    int ind = 0;
    while (std::getline(file,line) && ind < 33) {
      if (!line.empty() && line[0] == '%') {
        line = line.substr(1);  // Remove the '%' symbol
      }
      std::istringstream dataStream(line);
      if (dataStream >> n >> m >> nz) {
        break;
      } else ind++;
    }
    if (ind == 33) {
      std::cerr << "Error: Failed to read matrix dimensions." << std::endl;
      exit(EXIT_FAILURE);
    }

    std::cout << "matrix size:\t";
    std::cout << "n = " << n << ", m = " << m << ", nz = " << nz << "\n";

    // allocate memory for the sparse matrix 
    auto* r = memres<R>(n + 1);
    auto* p = memres<C>(nz);
    auto* c = memres<C>(nz);
    auto* v = memres<V>(nz);

    // Read the non-zero elements
    for (R i = 0; i < nz; ++i) {
      file >> p[i] >> c[i] >> v[i];
      --p[i]; // p[i] -= 1;
      --c[i]; // c[i] -= 1;
    }
    file.close();

    alg::sortByKey(nz,p,c,v);

    // find the row pointer for the csr matrix
    find_row_ptr(r,nz,p,p[0]);

    auto A = std::make_shared<CSR<R,C,V>>(n,m,nz,r,c,v);
    free(p);

    return A;
  }


  template <typename R,typename C,typename V>
  void cpp_write_ascii_matrix(const std::string filename,const shared_ptr<CSR<R,C,V>>& A) {

    std::ofstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << filename << std::endl;
      exit(EXIT_FAILURE);
    }
    // file << std::fixed << std::setprecision(9) << std::setw(12);
    file << A->n << " " << A->m << " " << A->nz << "\n";

    for (C i = 0; i < A->n; ++i) {
      for (R j = A->r[i]; j < A->r[i + 1]; ++j) {
        file << (i + 1) << " "
          << (A->c[j] + 1) << " "
          << A->v[j] << "\n";
      }
    }
    file.close();
  }

  template <typename R,typename C,typename V>
  void cpp_write_ascii_matrix_ISTL(const std::string& filename,const std::shared_ptr<CSR<R,C,V>>& A,C block_rows = 1,C block_cols = 1) {
    std::ofstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << filename << std::endl;
      exit(EXIT_FAILURE);
    }

    // Write Matrix Market header
    file << "%%MatrixMarket matrix coordinate real general\n";
    file << "% ISTL_STRUCT blocked " << block_rows << " " << block_cols << "\n";

    // Write matrix dimensions and number of nonzeros
    file << A->n << " " << A->m << " " << A->nz << "\n";

    // Set output format to scientific notation
    file << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10);

    // Write matrix entries
    for (C i = 0; i < A->n; ++i) {
      for (R j = A->r[i]; j < A->r[i + 1]; ++j) {
        file << (i + 1) << " " << (A->c[j] + 1) << " " << A->v[j] << "\n";
      }
    }

    file.close();
  }

  template <typename C,typename V>
  void cpp_write_ascii_array_ISTL(const std::string filename,const C n,const V* v,C block_rows = 1) {

    std::ofstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << filename << std::endl;
      exit(EXIT_FAILURE);
    }
    // Write Matrix Market header
    file << "%%MatrixMarket matrix array real general\n";
    file << "% ISTL_STRUCT blocked " << block_rows << " " << 1 << "\n";

    // Write matrix array dimensions 
    file << n << " " << 1 << "\n";

    // Set output format to scientific notation
    file << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10);

    // file << std::fixed << std::setprecision(9) << std::setw(12);
    for (C i = 0; i < n; ++i) {
      file << v[i] << "\n";
    }
    file.close();
  }

  template <typename C,typename V>
  void cpp_write_ascii_array(const std::string filename,const C n,const V* v) {

    std::ofstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << filename << std::endl;
      exit(EXIT_FAILURE);
    }
    // file << std::fixed << std::setprecision(9) << std::setw(12);
    for (C i = 0; i < n; ++i) {
      file << v[i] << "\n";
    }
    file.close();
  }

  template <typename R = int,typename V = double>
  V* cpp_read_ascii_array(std::string filename,R n = 0) {

    if (filename == "unity" || filename == "unitary" || filename == "one" || filename == "ones") {
      auto* v = memput<V>(n,1.0);
      return v;
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << filename << std::endl;
      exit(EXIT_FAILURE);
    }

    if (n == 0) {
      n = find_length(file);
    }

    std::cout << "rhs size:\t" << n << "\n";

    V* v = CHECK((V*)malloc(n * sizeof(V)));

    // Read the non-zero elements
    for (R i = 0; i < n; ++i) {
      file >> v[i];
    }
    file.close();

    return v;
  }

  template <typename R = int,typename C = int,typename V = double>
  V* cpp_read_ascii_rhs(std::string filename,shared_ptr<CSR<R,C,V>> A) {

    auto n = A->n;
    C m;
    if (filename == "unity" || filename == "unitary" || filename == "uno" || filename == "one" || filename == "ones") {
      auto* v = memres<V>(n);
      auto* x = memput<V>(n,1.0);
      alg::spmv(A->n,A->r,A->c,A->v,x,v);
      free(x);
      return v;
    } else if (filename == "zero") {
      auto* v = memres<V>(n);
      auto* x = memput<V>(n,0.0);
      alg::spmv(A->n,A->r,A->c,A->v,x,v);
      free(x);
      return v;
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << filename << std::endl;
      exit(EXIT_FAILURE);
    }

    std::string line;
    int ind = 0;
    while (std::getline(file,line) && ind < 33) {
      if (!line.empty() && line[0] == '%') {
        line = line.substr(1);  // Remove the '%' symbol
      }
      std::istringstream dataStream(line);
      if (dataStream >> n >> m) {
        break;
      } else ind++;
    }

    if (ind == 33) {
      std::cerr << "Error: Failed to read matrix dimensions." << std::endl;
      exit(EXIT_FAILURE);
    }

    if (n == 0) {
      n = find_length(file);
    }

    std::cout << "rhs size:\t" << n << "\n";

    V* v = CHECK((V*)malloc(n * sizeof(V)));

    // Read the non-zero elements
    for (R i = 0; i < n; ++i) {
      file >> v[i];
    }
    file.close();

    return v;
  }








  //////////////////////////////////////////////////////
  /*                     binary IO                    */
  //////////////////////////////////////////////////////



  template <typename R,typename C,typename V>
  void cpp_write_binary_matrix(const std::string& filename,const std::shared_ptr<CSR<R,C,V>>& A) {
    std::ofstream file(filename,std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
      exit(EXIT_FAILURE);
    }

    // Write the matrix dimensions and number of non-zero elements
    file.write(reinterpret_cast<const char*>(&A->n),sizeof(C));
    file.write(reinterpret_cast<const char*>(&A->m),sizeof(C));
    file.write(reinterpret_cast<const char*>(&A->nz),sizeof(R));

    // Write the CSR data arrays (row pointers, column indices, and values)
    file.write(reinterpret_cast<const char*>(A->r),sizeof(R) * (A->n + 1));
    file.write(reinterpret_cast<const char*>(A->c),sizeof(C) * A->nz);
    file.write(reinterpret_cast<const char*>(A->v),sizeof(V) * A->nz);

    file.close();
  }


  template <typename R,typename C,typename V>
  std::shared_ptr<CSR<R,C,V>> cpp_read_binary_matrix(const std::string& filename) {
    R n,m,nz;

    std::ifstream file(filename,std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << filename << std::endl;
      exit(EXIT_FAILURE);
    }

    // Read the matrix dimensions and number of non-zero elements
    file.read(reinterpret_cast<char*>(&n),sizeof(C));
    file.read(reinterpret_cast<char*>(&m),sizeof(C));
    file.read(reinterpret_cast<char*>(&nz),sizeof(R));

    std::cout << "matrix size:\t";
    std::cout << "n = " << n << ", m = " << m << ", nz = " << nz << "\n";

    // Allocate memory for the sparse matrix
    auto* r = memres<R>(n + 1);
    auto* c = memres<C>(nz);
    auto* v = memres<V>(nz);

    // Read the CSR data arrays (row pointers, column indices, and values)
    file.read(reinterpret_cast<char*>(r),sizeof(R) * (n + 1));
    file.read(reinterpret_cast<char*>(c),sizeof(C) * nz);
    file.read(reinterpret_cast<char*>(v),sizeof(V) * nz);

    file.close();

    // Create the CSR matrix
    auto A = std::make_shared<CSR<R,C,V>>(n,m,nz,r,c,v);
    return A;
  }


  template <typename C,typename V>
  void cpp_write_binary_array(const std::string& filename,const C n,const V* v) {
    std::ofstream file(filename,std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
      exit(EXIT_FAILURE);
    }

    // Write the size of the array (n)
    file.write(reinterpret_cast<const char*>(&n),sizeof(C));

    // Write the array elements
    file.write(reinterpret_cast<const char*>(v),sizeof(V) * n);

    file.close();
  }


  template <typename R = int,typename C = int,typename V = double>
  V* cpp_read_binary_rhs(const std::string& filename,shared_ptr<CSR<R,C,V>> A) {

    auto n = A->n;
    if (filename == "unity" || filename == "unitary" || filename == "uno" || filename == "one" || filename == "ones") {
      auto* v = memres<V>(n);
      auto* x = memput<V>(n,1.0);
      alg::spmv(A->n,A->r,A->c,A->v,x,v);
      free(x);
      return v;
    } else if (filename == "zero") {
      auto* v = memres<V>(n);
      auto* x = memput<V>(n,0.0);
      alg::spmv(A->n,A->r,A->c,A->v,x,v);
      free(x);
      return v;
    }

    std::ifstream file(filename,std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << filename << std::endl;
      exit(EXIT_FAILURE);
    }

    file.read(reinterpret_cast<char*>(&n),sizeof(C));

    std::cout << "rhs size:\t" << n << "\n";

    // Allocate memory for the array
    auto* v = memres<V>(n);

    // Read the array elements
    file.read(reinterpret_cast<char*>(v),sizeof(V) * n);
    file.close();

    return v;
  }





  //////////////////////////////////////////////////////
  /*                   MPI routines                   */
  //////////////////////////////////////////////////////


  template<typename C>
  C split_rows(const C n,const int mid,const  int np,const int bs = 1) {
    C nloc = n / np;
    if (nloc % bs > 0) nloc = (nloc / bs) * bs;
    if (mid == (np - 1)) nloc = n - (nloc * (np - 1)); // compute the number of rows for the last process
    return nloc;
  }

  template<typename R,typename C>
  void apply_shift(R* row,const R shift,const C n) {
#pragma omp parallel for 
    for (C i = 0; i < n; ++i) {
      row[i] -= shift;
    }
  }

  // template <typename R, typename C, typename V>
  template <template <typename,typename,typename> class TCSR,typename R,typename C,typename V>
  shared_ptr<TCSR<R,C,V>> distribute_csr_matrix(shared_ptr<TCSR<R,C,V>> A,int mid,int np,int bs) {

    if (np == 1) {
      A->np = np;
      A->mid = mid;

      A->n_glo = A->n;
      A->m_glo = A->m;

      A->idpool = memres<int>(np);
      A->idpool[0] = 0;

      A->displs_nr = memres<int>(np + 1);
      A->displs_nz = memres<int>(np + 1);
      A->counts_nr = memres<int>(np + 1);
      A->counts_nz = memres<int>(np + 1);
      for (int i = 0; i <= np; ++i) {
        A->displs_nr[i] = i * A->n;
        A->displs_nz[i] = i * A->nz;
        A->counts_nr[i] = A->n;
        A->counts_nz[i] = A->nz;
      }

      A->shift_nz = A->displs_nz[mid];
      A->shift_nr = A->displs_nr[mid];

      A->owns_MPI = true;

      return A;
    }

    R nzloc,shift;
    C n = A->n;
    C m = A->m;
    R nz = A->nz;

    CHECK_MPI(MPI_Bcast(&n,sizeof(C),MPI_BYTE,0,deco_comm)); // broadcast global sizes
    CHECK_MPI(MPI_Bcast(&m,sizeof(C),MPI_BYTE,0,deco_comm));
    CHECK_MPI(MPI_Bcast(&nz,sizeof(R),MPI_BYTE,0,deco_comm));

    auto nloc = split_rows(n,mid,np,bs); // partition the rows by np s.t. divisible by bs

    C counts_nr[np + 1]; // number of elements per processor for r buffer
    C displs_nr[np + 1]; // displacements for r buffer (exclusive sum of counts_nr)

    R counts_nz[np + 1]; // number of elements per processor for c/v buffer
    R displs_nz[np + 1]; // displacements for c/v buffer (exclusive sum of counts_nz)

    if (mid == 0) {
      displs_nz[0] = 0;
      displs_nr[0] = 0;
      for (C i = 1; i < np; ++i) {
        counts_nr[i - 1] = nloc + 1;
        displs_nr[i] = displs_nr[i - 1] + nloc;

        counts_nz[i - 1] = A->r[i * nloc] - A->r[(i - 1) * nloc];
        displs_nz[i] = displs_nz[i - 1] + counts_nz[i - 1];
      }
      counts_nr[np - 1] = n - (np - 1) * nloc + 1;
      counts_nz[np - 1] = A->r[A->n] - A->r[(np - 1) * nloc];
      displs_nr[np] = n;
      displs_nz[np] = A->nz;
    }

    CHECK_MPI(MPI_Scatter((int*)counts_nz,1,get_mpi_type<R>(),&nzloc,1,get_mpi_type<R>(),0,deco_comm)); // scatter the local nz count
    CHECK_MPI(MPI_Scatter((int*)displs_nz,1,get_mpi_type<R>(),&shift,1,get_mpi_type<R>(),0,deco_comm)); // scatter the shift (global nz count)

    auto Aloc = std::make_shared<CSR<R,C,V>>(nloc,m,nzloc);

    CHECK_MPI(MPI_Scatterv(A->r,(int*)counts_nr,(int*)displs_nr,get_mpi_type<R>(),Aloc->r,nloc + 1,get_mpi_type<R>(),0,deco_comm));
    CHECK_MPI(MPI_Scatterv(A->c,(int*)counts_nz,(int*)displs_nz,get_mpi_type<C>(),Aloc->c,nzloc,get_mpi_type<C>(),0,deco_comm));
    CHECK_MPI(MPI_Scatterv(A->v,(int*)counts_nz,(int*)displs_nz,get_mpi_type<V>(),Aloc->v,nzloc,get_mpi_type<V>(),0,deco_comm));

    // broadcast MPI data
    CHECK_MPI(MPI_Bcast(displs_nr,np + 1,get_mpi_type<C>(),0,deco_comm));
    CHECK_MPI(MPI_Bcast(displs_nz,np + 1,get_mpi_type<R>(),0,deco_comm));
    CHECK_MPI(MPI_Bcast(counts_nr,np + 1,get_mpi_type<C>(),0,deco_comm));
    CHECK_MPI(MPI_Bcast(counts_nz,np + 1,get_mpi_type<R>(),0,deco_comm));

    apply_shift(Aloc->r,shift,nloc + 1);

    // setup MPI data
    Aloc->np = np;
    Aloc->mid = mid;

    Aloc->n_glo = n;
    Aloc->m_glo = m;

    Aloc->idpool = memres<int>(np);
    for (int i = 0; i < np; ++i) Aloc->idpool[i] = i;

    Aloc->displs_nr = memres<int>(np + 1);
    Aloc->displs_nz = memres<int>(np + 1);
    Aloc->counts_nr = memres<int>(np + 1);
    Aloc->counts_nz = memres<int>(np + 1);
    for (int i = 0; i <= np; ++i) {
      Aloc->displs_nr[i] = displs_nr[i];
      Aloc->displs_nz[i] = displs_nz[i];
      Aloc->counts_nr[i] = counts_nr[i] - 1;
      Aloc->counts_nz[i] = counts_nz[i];
    }

    Aloc->shift_nz = displs_nz[mid];
    Aloc->shift_nr = displs_nr[mid];

    return Aloc;
  }

}