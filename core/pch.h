// C
#include <stdlib.h>      // Standard library for memory allocation, process control, conversions, etc.
#include <stdio.h>       // Standard I/O library for C-style input/output functions (e.g., printf).
#include <math.h>        // C-style math functions. Could be replaced by <cmath> for C++ code.
#include <limits.h>      // INT_MAX

// C++
#include <memory>        // Smart pointers (e.g., std::shared_ptr, std::shared_ptr) for automatic memory management.
#include <string>        // C++ standard string class (e.g., std::string).
#include <iostream>      // C++ standard I/O stream library (e.g., std::cout). Potential redundancy with <stdio>.
#include <fstream>       // C++ file stream library (e.g., file reading/writing with std::ifstream, std::ofstream).

// profiler
#include <chrono>        // C++ standard library for time-based operations (e.g., measuring execution time).
#include <unordered_map> // C++ standard unordered map (hash map) for key-value pairs.
#include <stack>         // C++ standard stack (LIFO data structure).
#include <vector>        // C++ standard dynamic array (vector).
#include <iomanip>       // C++ standard library for input/output manipulation (e.g., setting precision, formatting).

// HPC
#include <mpi.h>         // MPI library for parallel processing.
#include <omp.h>         // OpenMP for parallel processing on shared memory systems.
MPI_Comm deco_comm = MPI_COMM_WORLD;

// ptree
#include <boost/property_tree/ptree.hpp>        // Boost property tree for hierarchical data structures.
#include <boost/property_tree/json_parser.hpp>  // JSON parser from Boost property tree.

// lapack
#include <cblas.h>           // CBLAS library for basic linear algebra operations.
#include <lapacke.h>         // LAPACKe library for advanced linear algebra operations (e.g., solving systems, decompositions).

// hypre release v2.31.0
#include "HYPRE.h"           // HYPRE library for scalable linear solvers.
#include "HYPRE_parcsr_ls.h" // HYPRE's parallel structured solver routines.
#include <HYPRE_config.h>    // HYPRE configuration settings.

// deco
#include "../core/header.h"
#include "../core/profiler.hpp"
#include "../core/alg.hpp"
#include "../core/CSR.hpp"
#include "../precond/FSAI/core.tpp"
#include "../core/MxM.cpp"
#include "../IO/cpp_read.hpp"
#include "../IO/ptree.hpp"
#include "../adapter/csr2coo.cpp"
#include "../precond/FSAI/FSAI.hpp"
#include "../precond/hypre_AMG/hypre_AMG.hpp"
#include "../precond/Jacobi/Jacobi.hpp"
#include "../precond/CPR.hpp"
#include "../solver/SOL.hpp"