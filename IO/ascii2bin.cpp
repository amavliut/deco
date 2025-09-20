#include "../core/pch.h"

#include <iostream>
#include <algorithm> 
#include <mpi.h>
#include <omp.h>
#include<unistd.h>

#define DTR int
#define DTC int 
#define DTV double


// read write routine
int main(int argc, char* argv[]) {
  
  boost::property_tree::ptree param = IO::read_ptree(argc, argv); // initialize input parameters
  
  auto A = std::make_shared<CSR<DTR,DTC,DTV>>();
  DTV* b = nullptr;

  profiler.start("IO");
    // READ
    profiler.start("read");
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

    // WRITE
    profiler.start("write");
    IO::cpp_write_binary_matrix(param.get<std::string>("IO.wfile_A"), A);
    IO::cpp_write_binary_array(param.get<std::string> ("IO.wfile_b"), A->n, b);
    profiler.stop();
  profiler.stop();
  
  profiler.report();   

  return 0;
}
