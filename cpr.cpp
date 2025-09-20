#include "core/pch.h"

// define data types (hypre IJMatrix works with integers)
#define DTR int
#define DTC int 
#define DTV double

int main(int argc, char* argv[]) {

  // start MPI
  int mid, nprocs;
  init_MPI(&mid, &nprocs, &argc, &argv);
  MPI_Comm_set_errhandler(deco_comm, MPI_ERRORS_RETURN); // return errors instead of aborting
  // -------------------------------------

  { // create local scope for a proper memory release
  boost::property_tree::ptree param = IO::read_ptree(argc, argv); // initialize input parameters  

  // int np = param.get<int>("parallel.np");
  // omp_set_num_threads(np);
  // checkOMP();
  
  auto A = std::make_shared<CSR<DTR,DTC,DTV>>(); // system matrix
  DTV* b = nullptr; // right hand side
  DTV* x = nullptr; // initial solution
  auto prec_glo_type = param.get<std::string>("prec.prec_glo.type");
  if (mid == 0){
    profiler.start("IO");
    auto format = param.get<std::string>("IO.format");
    if (format == "ascii") {
      A = IO::cpp_read_ascii_matrix<DTR,DTC,DTV>(param.get<std::string>("IO.file_A"));
      b = IO::cpp_read_ascii_rhs   <DTR,DTC,DTV>(param.get<std::string>("IO.file_b"),A);
    } else if (format == "bin" || format == "binary"){
      A = IO::cpp_read_binary_matrix<DTR,DTC,DTV>(param.get<std::string>("IO.file_A"));
      b = IO::cpp_read_binary_rhs   <DTR,DTC,DTV>(param.get<std::string>("IO.file_b"),A);
    } else {
       throw std::invalid_argument("Unknown global IO::format: " + format);
    }
    profiler.stop();

    AD::block_reorder(A, b, param); // reordering is done for simplicity of academic code production, only pressure eqn & var are needed for production usecase
  }
  
  MPI_Barrier(deco_comm);

  // standard 1D row splitting
  auto Aloc = IO::distribute_csr_matrix(A, mid, nprocs, param.get<int>("prec.block_size"));

  auto* b_loc = memres<DTV>(Aloc->n);
  DTV* x_loc = nullptr;
  
  MPI_Scatterv(b, Aloc->counts_nr, Aloc->displs_nr, get_mpi_type<DTV>(),
               b_loc, Aloc->n, get_mpi_type<DTV>(),0,deco_comm);

  // Time measurements start when the input system is split among MPI ranks, since in practice the linear solver will be called at that point
  MPI_Barrier(deco_comm);
  profiler.start("setup & solve");
  profiler.start("Prec");

  Aloc->setup_spmv_v2(); // setup MPI version of spmv

  std::shared_ptr<CPRBase<DTV>> Ploc; // define the base class

  if (prec_glo_type == "Jacobi")
    Ploc = std::make_shared<CPR<DTR, DTC, DTV,Jacobi,hypre_AMG>>(param.get_child("prec"), Aloc, b_loc);
  else if (prec_glo_type == "FSAI")
    Ploc = std::make_shared<CPR<DTR, DTC, DTV,FSAI,hypre_AMG>>(param.get_child("prec"), Aloc, b_loc);
  else 
    throw std::invalid_argument("Unknown global prec.prec_glo.type: " + prec_glo_type);

  MPI_Barrier(deco_comm);
  profiler.stop();

  MPI_Barrier(deco_comm);
  profiler.start("Solve");
  auto S = std::make_shared<SOL <DTR, DTC, DTV>>(param.get_child("sol") , Aloc, Ploc->apply, b_loc, x_loc);
  MPI_Barrier(deco_comm);
  profiler.stop();
  profiler.stop();

  free(b);
  free(x);

  profiler.post_process();

  if (mid==0) profiler.report();   
  }
  MPI_Finalize();

  return 0;
}
