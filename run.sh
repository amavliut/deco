# run with 4 openMP threads
./compile.sh && export OMP_NUM_THREADS=4 && ./build/cpr param_norne.json

# run with 4 MPI ranks
# ./compile.sh && export OMP_NUM_THREADS=1 && mpirun -np 4 ./build/cpr param_norne.json