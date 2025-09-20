# deco Linear Solver Library

A C++ library for **decoupling, preconditioning, and solving linear systems** arising from **multi-phase flow problems**.

---

## Features

- **CPR Preconditioner**: Based on AMG from [Hypre](https://github.com/hypre-space/hypre) and in-house FSAI Preconditioner. For benchmarking in distributed MPI framework, FSAI preconditioner from [Chronos](https://www.m3eweb.it/chronos/) was used.
- **Parallelization**:
  - Most of the library is parallelized with both **OpenMP** and **MPI**.  
  - in-house FSAI parallelization is limited to OpenMP.

---

## Installation


1. **Install Hypre**  
   Download and install [Hypre](https://github.com/hypre-space/hypre).  
   Optionally, compile it with **MPI support** if you plan to use distributed parallelization.

2. **Compile the library**  
   Run the compile script:
   ```bash
   ./compile.sh 
3. **Run tests**  
   Three sample linear systems are provided, originating from **Norne**, **SPE11C**, and **Sleipner** simulations.  
   Use the `run.sh` script to execute the tests, specifying the input `.json` file inside

   ```bash
   ./run.sh
