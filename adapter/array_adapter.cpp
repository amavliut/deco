#include <iostream>

namespace adapter {
  template <typename R = int,typename V = double>
  class block_array_left {
  public:
    const int bs;       // block size
    const R   sn;       // reduced size
    const R    n;       // original size
    const V* arr;       // original array
    // ===========================================================================
    block_array_left(int bs_,R n_,V* arr_) : bs(bs_),n(n_),sn(n_ / bs_),arr(arr_) {
      if (n % bs != 0) {
        std::cerr << "n is not divisible by bs, n =  " << n << ", bs = " << bs << ", sn = " << sn << std::endl;
        exit(EXIT_FAILURE);
      }
    }

    // Overloaded operator[] to adjust the index
    V operator[](R index) const {
      return arr[bs * index];
    }
  private:

  };

  // dense matrix adapter
  template <typename V = double>
  class matrix {
  public:
    const int    n;       // number of rows
    const int    m;       // number of columns
    V* arr;               // array in column major order
    // ===========================================================================
    matrix(int n_,int m_,V* arr_) : n(n_),m(m_),arr(arr_) {}

    // return value at [i,j] position
    V& operator()(const int i,const int j) {
      return arr[i + j * n];
    }

    // // return column j
    // const V& operator()(const int j) const {
    //   return arr[j*n];
    // }

    void print() {
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
          printf("%8.2f ",arr[i + j * n]);
        }
        printf("\n");
      }
      printf("\n");
    }
  };


  // dense matrix adapter
  template <typename R,typename C,typename V = double>
  class mpi_array {
  public:
    const C    n;       // local array size
    V** arr;            // double array for each mpi rank it stores its local part 
    R* displ;           // displacements of the local arrays with respect to the global one
    // ===========================================================================
    mpi_array(const C n_,const C displ_,V* arr_) : n(n_),displ(displ_),arr(arr_) {}

    V& operator[](const C ind) {
      int ind_rank = extract_rank(ind); // use find_rank
      return arr[ind_rank][ind - displ[ind_rank]];
    }
  };

  template <typename R = int,typename V = double>
  class block_array_right {
  public:
    const int bs;       // block size
    const R   sn;       // reduced size
    const R    n;       // original size
    const V* arr;       // original array
    // ===========================================================================
    block_array_right(int bs_,R n_,V* arr_) : bs(bs_),n(n_),sn(n_ / bs_),arr(arr_) {
      if (n % bs != 0) {
        std::cerr << "n is not divisible by bs, n =  " << n << ", bs = " << bs << ", sn = " << sn << std::endl;
        exit(EXIT_FAILURE);
      }
    }

    // Overloaded operator[] to adjust the index
    V operator[](R index) const {
      return arr[index / bs];
    }
  private:

  };


  template <typename V = double>
  class block_array {
  public:
    const int p;        // pressure entry
    const int bs;       // block size
    V* arr;             // array
    // ===========================================================================
    block_array(int p_,int bs_,V* arr_) : p(p_),bs(bs_),arr(arr_) {}
    // read
    template <typename T>
    const V& operator[](const T index) const {
      return arr[bs * index + p];
    }
    // write
    template <typename T>
    V& operator[](const T index) {
      return arr[bs * index + p];
    }
  };
}