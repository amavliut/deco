cmake_minimum_required(VERSION 3.10)
project(deco VERSION 0.1.0 LANGUAGES C CXX)

# Find MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# Find OpenMP
find_package(OpenMP REQUIRED)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)

# Set LAPACK
set(LAPACKE_LIBRARIES /usr/lib/x86_64-linux-gnu/liblapacke.so)

# Hypre
set(HYPRE_DIR "path2hypre")
include_directories(${HYPRE_DIR}/include)
link_directories(${HYPRE_DIR}/lib)
set(HYPRE_LIBRARY ${HYPRE_DIR}/lib/libHYPRE.a)

# OpenMP compile options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# main file
add_executable(cpr cpr.cpp)
target_link_libraries(cpr PRIVATE ${MY_COMPILER_OPTIONS} ${MPI_CXX_LIBRARIES} ${LAPACKE_LIBRARIES} HYPRE OpenMP::OpenMP_CXX)

# routine to convert ascii data to binary data
add_executable(ascii2bin IO/ascii2bin.cpp)
target_link_libraries(ascii2bin PRIVATE -g ${MPI_CXX_LIBRARIES})