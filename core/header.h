// Template function to map C++ types to MPI datatypes
template <typename T>
MPI_Datatype get_mpi_type();

template <>
MPI_Datatype get_mpi_type<int>() {
    return MPI_INT;
}

template <>
MPI_Datatype get_mpi_type<long>() {
    return MPI_LONG;
}

template <>
MPI_Datatype get_mpi_type<float>() {
    return MPI_FLOAT;
}

template <>
MPI_Datatype get_mpi_type<double>() {
    return MPI_DOUBLE;
}

template <>
MPI_Datatype get_mpi_type<char>() {
    return MPI_CHAR;
}

template <>
MPI_Datatype get_mpi_type<bool>() {
    return MPI_CHAR;
}


#define ISMASTER (mid==0)

#define _MPI_ENV int mid, nprocs; MPI_Comm_rank(deco_comm, &mid); MPI_Comm_size(deco_comm,&nprocs)

inline void check_mpi(int mpi_err_code, const char* file, int line) {
    if (mpi_err_code != MPI_SUCCESS) {
        char mpi_err_string[MPI_MAX_ERROR_STRING];
        int mpi_err_len;
        MPI_Error_string(mpi_err_code, mpi_err_string, &mpi_err_len);
        
        std::cerr << "[MPI ERROR] " << file << ":" << line << std::endl
                  << "  MPI Error Code: " << mpi_err_code << std::endl
                  << "  MPI Error: " << mpi_err_string << std::endl;
        
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_MPI(X) check_mpi((X), __FILE__, __LINE__)

// ##########################################################################
    
void start_MPI(int *id, int *np, int *argc, char ***argv) {
  int myid, nprocs;
  MPI_Init(argc, argv);
  // hypre_MPI_Init(&argc, &argv);
  MPI_Comm_rank(deco_comm,&myid);
  MPI_Comm_size(deco_comm,&nprocs);
  *id=myid;
  *np=nprocs;
}

void init_MPI(int *id, int *np, int *argc = NULL, char ***argv = NULL, int np_ = 1) {

  int myid, nprocs;
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    deco_comm = MPI_COMM_WORLD;
  } else{
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int color = world_rank / np_; // Groups processes by their rank
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &deco_comm);
    MPI_Comm_rank(deco_comm, &myid);
    MPI_Comm_size(deco_comm, &nprocs);
  }
  *id=myid;
  *np=nprocs;
}

void finish_MPI(){
  int finalized;
  MPI_Finalized(&finalized);
  if (!finalized) {
      MPI_Finalize();
  } else {
    MPI_Comm_free(&deco_comm);
  }
}


void checkOMP(){
#pragma omp parallel
  {
    printf("Thread %d of %d\n", omp_get_thread_num(), omp_get_max_threads());
  }
}

// print the name of a variable
#define PRINT_VAR_NAME(name) print_var_name(#name)
void print_var_name(std::string name) {
    std::cout << "varname: " << name << std::endl;
}


enum class memtype { C,Cpp,MKL };


#define CHECK(ptr) check((ptr), __func__, __FILE__, __LINE__)
template<typename T>
static inline T* check(T* ptr,const char* func,const char* file,int line) {
  if (ptr == NULL) {
    fprintf(stderr,"Error: Null pointer detected in function %s, file %s, line %d\n",func,file,line);
    std::exit(EXIT_FAILURE);
  }
  return ptr;
}
template<typename T>
static inline T check(T value, const char* func, const char* file, int line) {
    if (value) { // Check if value is zero or false
        std::cerr << "Error: flag is not zero, flag = "<<value << ", in function " 
                  << func << ", file " << file << ", line " << line << '\n';
        // std::exit(EXIT_FAILURE);
    }
    return value;
}


// reserve memory
template<typename T>
T* memres(size_t n, const std::string& type = "C") {
  T* ptr = nullptr;
  if (type == "C") {
    // ptr = static_cast<T*>(aligned_alloc(16, n * sizeof(T)));
    // auto flag = posix_memalign((void**)&ptr, 64, n * sizeof(T)); 
    ptr = static_cast<T*>(malloc(n * sizeof(T)));
  } else if (type == "Cpp") {
    ptr = new T[n];
  } else if (type == "MKL") {
    // ptr = static_cast<T*>(mkl_malloc(n * sizeof(T), 64));
  }
  return CHECK(ptr);
}

template<typename T>
T* memrealloc(T*& ptr, size_t n, const std::string& type = "C") {
  T* temp = nullptr;
  if (type == "C") {
    temp = static_cast<T*>(realloc(ptr, n * sizeof(T)));
  } else if (type == "Cpp") {
    ptr = new T[n];
  } else if (type == "MKL") {
    // ptr = static_cast<T*>(mkl_malloc(n * sizeof(T), 64));
  }
  ptr = CHECK(temp);
  return ptr;
}

template<typename V, typename T>
void memcopy(V* __restrict__ dest, const V* __restrict__ src, const T n) {
  // void memcopy(V* dest, const V* src, const T n) {
// #pragma omp parallel 
//   {
//     int tid = omp_get_thread_num();
//     int np  = omp_get_max_threads();
//     T nloc  = n / np;
//     T start =    tid  * nloc;
//     // T end   = (tid+1) * nloc;
//     T end   = (tid == np - 1) ? n : start + nloc;

//     memcpy(dest+start,src+start,(end-start)*sizeof(V));  
//   }
  // 
  // #pragma omp simd
  #pragma omp parallel for
  for (T i=0;i<n;++i){
    dest[i]=src[i];
  }   
}

template<typename V, typename T>
void meminit(V*& arr, const T n) {
  #pragma omp parallel 
  {
      int tid = omp_get_thread_num();
      int np  = omp_get_num_threads();

      T nloc  = n / np;
      T start = tid * nloc;
      T end   = (tid == np - 1) ? n : start + nloc;

      std::memset(arr+start, 0, (end-start)*sizeof(V));
  }
}

void parallel_memset(void* buffer, int value, size_t size) {
    unsigned char* byteBuffer = static_cast<unsigned char*>(buffer);

    #pragma omp parallel
    {
        size_t tid = omp_get_thread_num();
        size_t np = omp_get_num_threads();

        // Calculate start and end indices for each thread
        size_t chunk_size = size / np;
        size_t start = tid * chunk_size;
        size_t end = (tid == np - 1) ? size : start + chunk_size;

        // Perform memset for the thread's chunk
        std::memset(byteBuffer + start, value, end - start);
    }
}

template<typename D, typename S, typename N>
void arrcopy(D dest, const S src, N n) {
#pragma omp parallel for
  for (N i = 0; i < n; ++i){
    dest[i] = src[i];
  }
}

template<typename D, typename S, typename W, typename N>
void arrcp_scale(D dest, const S src, const W scale, N n) {
#pragma omp parallel for
  for (N i = 0; i < n; ++i){
    dest[i] = src[i] / scale[i];
  }
}



template<typename T>
T* memput(size_t n, T val = 0.0, const std::string& type = "C") {
  T* ptr = nullptr;
  if (type == "C") {
    if (val == 0.0){
      if (ptr == nullptr) ptr = static_cast<T*>(calloc(n, sizeof(T)));
      else {
#pragma omp parallel for
        for (size_t i = 0; i < n; ++i){
          ptr[i] = val;
        }
      }
    } else{
      if (ptr == nullptr) ptr = static_cast<T*>(malloc(n * sizeof(T)));
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i){
        ptr[i] = val;
      }
    }
  } else if (type == "Cpp") {
    
  } else if (type == "MKL") {
    
  }

  return CHECK(ptr);
}

template<typename T>
void memcal(T* &ptr, size_t n, T val = 0.0, const std::string& type = "C") {
  if (type == "C") {
    if (val == 0.0){
      if (ptr == nullptr) ptr = CHECK( static_cast<T*>(calloc(n, sizeof(T))) );
      else {
#pragma omp parallel for
        for (size_t i = 0; i < n; ++i){
          ptr[i] = val;
        }
      }
    } else{
      if (ptr == nullptr) ptr = CHECK( static_cast<T*>(malloc(n * sizeof(T))) );
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i){
        ptr[i] = val;
      }
    }
  } else if (type == "Cpp") {
    
  } else if (type == "MKL") {
    
  }
}

template <typename T>
void free_memory(T*& ptr) {
    if (ptr != nullptr) {
        free(ptr);
        ptr = nullptr;  // Null the pointer after freeing memory
    }
}