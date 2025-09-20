mkdir -p build && \
cmake -S . -B build/ -DCMAKE_BUILD_TYPE=Release && \
  cd build && \
  make cpr -j8
  cd --
