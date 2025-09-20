
/*
  Linear algebra kernels
*/

template <typename R,typename C,typename V>
class CSR;

template <typename R,typename C,typename V>
class Jacobi;


namespace core {

  template<typename T>
  T binarysearch(const T* __restrict__ arr,const T lower,const T upper,const T target) {
    T min = lower;
    T max = upper;
    T mid = (lower + upper) >> 1;
    while (max - min > 1) {
      arr[mid] > target ? max = mid : min = mid;
      mid = (min + max) >> 1;
    }
    return mid;
  }

  template<typename T>
  T binarysearch(const T* __restrict__ arr,const T upper,const T target) {
    T min = 0;
    T max = upper;
    T mid = upper >> 1;
    while (max - min > 1) {
      arr[mid] > target ? max = mid : min = mid;
      mid = (min + max) >> 1;
    }
    return mid;
  }
}

namespace alg {
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define HASH_SCAL 107

  /*
   * Compute sparsae matrix by vector product
   * b = A*x
   */
   // spmv_kernel
  template <typename R,typename C,typename V>
  void spmv(const R n,const R* __restrict r,const C* __restrict c,const V* __restrict v,const V* __restrict x,V* __restrict b,const int bs = 1) {

    // profiler.start("spmv");
#pragma omp parallel for
    for (R i = 0; i < n; i += bs) {
      R ra = r[i];
      R rb = r[i + 1];
      V sum = 0.;
      for (R j = ra; j < rb; ++j) {
        sum += v[j] * x[c[j]];
      }
      b[i / bs] = sum;
    }
  }


  void findIndicesWithinBounds(int displacements[],int size,int left,int right,int indices[],int* count) {

    // two ranges [a,b] and [c,d] overlap: if and only if max(a,c)≤min(b,d) (since the range is excluded then strict sign)
    _MPI_ENV;
    for (int i = 0; i < size; i++) {
      if ((displacements[i] < right) && (left < displacements[i + 1])) {
        indices[*count] = i; // Store the index
        (*count)++;          // Increment the count
      }
    }
  }



  template <typename R,typename C,typename V>
  void MPI_spmv(std::shared_ptr<CSR<R,C,V>> A,const V* x_loc,V* b_loc,const int bs = 1) {

    if (A->xspmv == nullptr) {
      spmv(A->n,A->r,A->c,A->v,x_loc,b_loc,bs);
    } else {

      _MPI_ENV;

      auto n = A->n;
      auto* r = A->r;
      auto* c = A->c;
      auto* v = A->v;
      auto* displs_nr = A->displs_nr;
      auto* source_ranks = A->source_ranks;
      auto* destin_ranks = A->destin_ranks;
      auto count_src = A->count_src;
      auto count_dst = A->count_dst;
      std::vector<MPI_Request> send_requests = A->send_requests;
      std::vector<MPI_Request> recv_requests = A->recv_requests;

      // Non-blocking send loop
      // std::vector<MPI_Request> send_requests(count_dst);
      for (int j = 0; j < count_dst; ++j) {
        MPI_Isend(x_loc,A->n,get_mpi_type<V>(),destin_ranks[j],0,deco_comm,&send_requests[j]);
      }

      // Non-blocking receive loop
      // std::vector<MPI_Request> recv_requests(count_src);
      for (int j = 0; j < count_src; ++j) {
        MPI_Irecv(A->xspmv + A->displs_nr[source_ranks[j]],
          A->counts_nr[source_ranks[j]],
          get_mpi_type<V>(),
          source_ranks[j],
          0,
          deco_comm,
          &recv_requests[j]);
      }

      // Perform computation while communication progresses
      bool all_sends_done = false,all_recvs_done = false;

      while (!all_sends_done || !all_recvs_done) {
        // Check progress of sends
        if (!all_sends_done) {
          int flag;
          MPI_Testall(count_dst,send_requests.data(),&flag,MPI_STATUSES_IGNORE);
          all_sends_done = flag; // True if all sends are completed
        }

        // Check progress of receives
        if (!all_recvs_done) {
          int flag;
          MPI_Testall(count_src,recv_requests.data(),&flag,MPI_STATUSES_IGNORE);
          all_recvs_done = flag; // True if all receives are completed
        }

        // Perform some independent computation here
      }

      spmv(A->n,A->r,A->c,A->v,A->xspmv,b_loc,bs);

    }
  }



  /*
   * Scale CSR matrix A by diagonal matrix D (stored as array)
   * A_ = D*A;
   */
  template <typename R,typename C,typename V>
  void diag_scale(C n,V* D,R* r,V* v) {

#pragma omp parallel for 
    for (C i = 0; i < n; ++i) {
      auto ia = r[i];
      auto ib = r[i + 1];
      auto d = D[i];
      for (R j = ia; j < ib; ++j) {
        v[j] *= d;
      }
    }
  }

  /*
   * Scale vector b by diagonal matrix D (stored as array)
   * b_ = D*b;
   */
  template <typename C,typename V>
  void diag_scale(C n,V* D,V* b) {

#pragma omp parallel for 
    for (C i = 0; i < n; ++i) {
      b[i] *= D[i];
    }
  }


  /*
   * Sum of of two vectors
   * y = b*y + a*x
   */
  template <typename R,typename V>
  void axpby(R n,V a,V* x,V b,V* y) {
#pragma omp parallel for
    for (R i = 0; i < n; ++i) {
      y[i] = b * y[i] + a * x[i];
    }
  }

  /*
   * Sum of of two vectors
   * y = z + a*x
   */
  template <typename R,typename V>
  void axpby(R n,V a,V* x,V* z,V* y) {
#pragma omp parallel for
    for (R i = 0; i < n; ++i) {
      y[i] = z[i] + a * x[i];
    }
  }

  /*
   * Sum of of two vectors
   * y = z + x
   */
  template <typename R,typename V>
  void axpby(R n,V* x,V* z,V* y) {
#pragma omp parallel for
    for (R i = 0; i < n; ++i) {
      y[i] = z[i] + x[i];
    }
  }

  /*
   * Sum of of two vectors
   * y = y + x
   */
  template <typename R,typename V>
  void axpby(R n,V* x,V* y) {
#pragma omp parallel for
    for (R i = 0; i < n; ++i) {
      y[i] += x[i];
    }
  }

  /*
   * Sum of of two vectors
   * y = a * x
   */
  template <typename R,typename V>
  void axpby(R n,V a,V* x,V* y) {
#pragma omp parallel for
    for (R i = 0; i < n; ++i) {
      y[i] = a * x[i];
    }
  }

  /*
  * General data types (adapters)
  * Sum of of two vectors
  * y = y + x
  */
  template <typename R,typename S,typename D>
  void axpby(R n,S x,D y) {
#pragma omp parallel for
    for (R i = 0; i < n; ++i) {
      y[i] += x[i];
    }
  }

  /*
   * Sum of of 3 vectors
   * y = b*y + a*x + d*z
   */
  template <typename R,typename V>
  void axpby(R n,V d,V* z,V a,V* x,V b,V* y) {
#pragma omp parallel for
    for (R i = 0; i < n; ++i) {
      y[i] = b * y[i] + a * x[i] + d * z[i];
    }
  }

  template <typename R,typename V>
  void axpby(R n,V d,V* z,V a,V* x,V* y) {
#pragma omp parallel for
    for (R i = 0; i < n; ++i) {
      y[i] = a * x[i] + d * z[i];
    }
  }

  /*
   * Scalar product of two vectors
   * s = a · b
   */
  template <typename R,typename V>
  V dot(R n,V* a,V* b) {
    V s = 0.0;
#pragma omp parallel
    {
      V loc_sum = 0.0;
#pragma omp for
      for (R i = 0; i < n; ++i) {
        loc_sum += a[i] * b[i];
      }
#pragma omp atomic
      s += loc_sum;
    }
    return s;
  }

  template <typename R,typename V>
  V dot(const R n,const V* a,const V* b) {
    V s = 0.0;
#pragma omp parallel
    {
      V loc_sum = 0.0;
#pragma omp for
      for (R i = 0; i < n; ++i) {
        loc_sum += a[i] * b[i];
      }
#pragma omp atomic
      s += loc_sum;
    }
    return s;
  }

  template <typename R,typename V>
  V MPI_dot(R n,V* a,V* b) {
    V glo_sum = 0.0;
    V loc_sum = dot(n,a,b);
    MPI_Allreduce(&loc_sum,&glo_sum,1,get_mpi_type<V>(),MPI_SUM,deco_comm);
    return glo_sum;
  }

  template <typename R,typename V>
  V MPI_dot(const R n,const V* a,const V* b,const int np) {
    V loc_sum = dot(n,a,b);
    if (np == 1) {
      return loc_sum;
    }

    V glo_sum = 0.0;
    MPI_Allreduce(&loc_sum,&glo_sum,1,get_mpi_type<V>(),MPI_SUM,deco_comm);
    return glo_sum;
  }

  /*
   * L2norm
   * s = sqrt(sum(v_i^2))
   */
  template <typename R,typename V>
  double l2norm(R n,V* v) {

    double s = 0.0;

#pragma omp parallel
    {
      double loc_sum = 0.0;

#pragma omp for
      for (R i = 0; i < n; ++i) {
        loc_sum += v[i] * v[i];
      }

#pragma omp atomic
      s += loc_sum;
    }

    return sqrt(s);
  }

  template <typename R,typename V>
  double l2norm(R n,V v) {

    double s = 0.0;

#pragma omp parallel
    {
      double loc_sum = 0.0;

#pragma omp for
      for (R i = 0; i < n; ++i) {
        loc_sum += v[i] * v[i];
      }

#pragma omp atomic
      s += loc_sum;
    }

    return sqrt(s);
  }

  /*
 * L2norm
 * s = sqrt(sum(v_i^2))
 */
  template <typename R,typename V>
  V MPI_l2norm(R n,V* v) {
    V glo_sum = 0.0;
    V loc_sum = l2norm(n,v);
    loc_sum *= loc_sum;
    MPI_Allreduce(&loc_sum,&glo_sum,1,get_mpi_type<V>(),MPI_SUM,deco_comm);
    return sqrt(glo_sum);
  }

  template <typename R,typename V>
  V MPI_l2norm(const R n,const V* v,const int np) {
    V glo_sum = 0.0;
    V loc_sum = l2norm(n,v);
    if (np == 1) return loc_sum;
    loc_sum *= loc_sum;
    MPI_Allreduce(&loc_sum,&glo_sum,1,get_mpi_type<V>(),MPI_SUM,deco_comm);
    return sqrt(glo_sum);
  }

  /*
   * L1norm^2
   * s = sum(abs(v_i))
   */
  template <typename T,typename V>
  double l1norm(T n,V* v) {

    double s = 0.0;

#pragma omp parallel
    {
      double loc_sum = 0.0;

#pragma omp for
      for (T i = 0; i < n; ++i) {
        loc_sum += abs(v[i]);
      }

#pragma omp atomic
      s += loc_sum;
    }

    return s;
  }

  /*
   * L2norm
   * s = sum(v_i)
   */
  template <typename R,typename V>
  V MPI_l1norm(R n,V* v) {
    V glo_sum = 0.0;
    V loc_sum = l1norm(n,v);
    MPI_Allreduce(&loc_sum,&glo_sum,1,get_mpi_type<V>(),MPI_SUM,deco_comm);
    return glo_sum;
  }


  /*
   * Total sum of vector's values
   */
  template <typename R,typename T>
  T int_sum(R n,T* a) {
    T s = 0;
#pragma omp parallel
    {
      T loc_sum = 0;
#pragma omp for
      for (R i = 0; i < n; ++i) {
        loc_sum += a[i];
      }
#pragma omp atomic
      s += loc_sum;
    }
    return s;
  }

  /*
   * Total max of vector's values
   */
  template <typename R,typename T>
  T int_max(R n,T* a) {
    T s = 0;
#pragma omp parallel
    {
      T loc = 0;
#pragma omp for
      for (R i = 0; i < n; ++i) {
        loc = MAX(loc,a[i]);
      }
      // #pragma omp atomic // can't use here, atomic works with simple instruction only (assignment, addition, subtraction)
#pragma omp critical
      s = MAX(s,loc);
    }
    return MIN(s,n);
  }

  /*
  * copy x vector to y
  * y = x
  */
  template <typename R,typename V>
  void copy(R n,V* x,V* y) {

#pragma omp parallel for
    for (R i = 0; i < n; ++i) {
      y[i] = x[i];
    }
  }


  /*
  * cummulative sum of array
  */
  template <typename R>
  void prefix_sum(const R* arr,R* result,const R n) {
    result[0] = 0;
#pragma omp parallel for
    for (R i = 1; i <= n; ++i) {
      result[i] = arr[i - 1];
    }
    for (R i = 1; i <= n; ++i) {
      result[i] += result[i - 1];
    }
  }


}

#include "sorting.hpp"