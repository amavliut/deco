// Different sorting routines (basic) used in deco 
// f.e. chaning AoS to SoA data structures or using some library
// There are some redundant function (to be organized)

namespace alg {

#define RADIX 10 // Base for decimal system

  // Counting sort function used by radix sort
  void countingSort(int arr[],int size,int exp) {
    int output[size]; // Output array
    int count[RADIX] = { 0 }; // Count array

    // Store count of occurrences of each digit
    for (int i = 0; i < size; i++) {
      count[(arr[i] / exp) % RADIX]++;
    }

    // Change count[i] so that it now contains actual position of this digit in output[]
    for (int i = 1; i < RADIX; i++) {
      count[i] += count[i - 1];
    }

    // Build the output array
    for (int i = size - 1; i >= 0; i--) {
      output[count[(arr[i] / exp) % RADIX] - 1] = arr[i];
      count[(arr[i] / exp) % RADIX]--;
    }

    // Copy the output array to arr[], so that arr[] contains sorted numbers
    for (int i = 0; i < size; i++) {
      arr[i] = output[i];
    }
  }

  // Function to perform radix sort
  void radixSort(int arr[],int size) {
    // Find the maximum number to figure out the number of digits
    int max = arr[0];
    for (int i = 1; i < size; i++) {
      if (arr[i] > max) {
        max = arr[i];
      }
    }

    // Do counting sort for every digit. exp is 10^i where i is the current digit number
    for (int exp = 1; max / exp > 0; exp *= RADIX) {
      countingSort(arr,size,exp);
    }
  }


  // Structure to store the triple pair
  template<typename R,typename C,typename V>
  struct Tripple {
    R key;
    C val1;
    V val2;
  };

  template<typename R,typename T>
  struct Pair {
    R key;
    T val1;
  };

  template<typename Tuple>
  int compare(const void* a,const void* b) {

    const Tuple& t1 = *static_cast<const Tuple*>(a);
    const Tuple& t2 = *static_cast<const Tuple*>(b);

    // Compare key first, and if they are equal, compare val1
    return (t1.key != t2.key) ? (t1.key - t2.key) : (t1.val1 - t2.val1);
  }


  template<typename R,typename C,typename V>
  void sortByKey(R n,R* array_a,C* array_b,V* array_c = nullptr) {

    int i;
    if (array_c == nullptr) {
      auto* data = memres<Pair<R,C>>(n);

#pragma omp parallel for private(i)
      for (i = 0; i < n; i++) {
        (data[i]).key = array_a[i];
        (data[i]).val1 = array_b[i];
      }
      size_t size = sizeof(data) / sizeof(*data);

      std::cout << "size for qsort = " << size << "\n";

      qsort(data,size,sizeof(*data),compare<Pair<R,C>>);

#pragma omp parallel for private(i)
      for (i = 0; i < n; i++) {
        array_a[i] = (data[i]).key;
        array_b[i] = (data[i]).val1;
      }
      free(data);
    } else {

      auto* data = memres<Tripple<R,C,V>>(n);

#pragma omp parallel for private(i)
      for (i = 0; i < n; i++) {
        (data[i]).key = array_a[i];
        (data[i]).val1 = array_b[i];
        (data[i]).val2 = array_c[i];
      }

      qsort(data,n,sizeof(*data),compare<Tripple<R,C,V>>);

#pragma omp parallel for private(i)
      for (i = 0; i < n; i++) {
        array_a[i] = (data[i]).key;
        array_b[i] = (data[i]).val1;
        array_c[i] = (data[i]).val2;
      }
      free(data);
    }
  }


  template<typename R,typename C,typename V>
  void sortCSR(R n,R nz,R* r,C* c,V* v) {

    int i;
    auto* data = memres<Pair<C,V>>(nz);

#pragma omp parallel for private(i)
    for (i = 0; i < nz; i++) {
      (data[i]).key = c[i];
      (data[i]).val1 = v[i];
    }

#pragma omp parallel for private(i)
    for (i = 0; i < n; ++i) {
      auto ra = r[i];
      auto rb = r[i + 1];
      qsort(data + ra,rb - ra,sizeof(*data),compare<Pair<C,V>>);
    }

#pragma omp parallel for private(i)
    for (i = 0; i < nz; i++) {
      c[i] = (data[i]).key;
      v[i] = (data[i]).val1;
    }
    free(data);

  }

  // sort array_val with array_key as its pair
  template<typename R,typename C>
  void sortByKey_abs(int n,R* array_val,C* array_key,double tol = 1e-25) {

    auto* data = memres<Pair<R,C>>(n);

    for (int i = 0; i < n; i++) {
      (data[i]).key = abs(array_val[i]);
      (data[i]).val1 = array_key[i];
    }
    size_t size = sizeof(data) / sizeof(*data);

    qsort(data,size,sizeof(*data),compare<Pair<R,C>>);

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
      array_val[i] = (data[i]).key;
      array_key[i] = (data[i]).val1;
    }
    free(data);
  }

  void sortByValue(int n,int* Ap_id,double* Ap) {
    for (int i = 1; i < n; ++i) {
      double value = abs(Ap[i]);
      int key = Ap_id[i];
      int j = i - 1;
      while (j >= 0 && abs(Ap[j]) > value) {
        Ap[j + 1] = abs(Ap[j]);
        Ap_id[j + 1] = Ap_id[j];
        j--;
      }
      Ap[j + 1] = value;
      Ap_id[j + 1] = key;
    }
  }



  // SORTING FOR FSAI


  /*
    simple key-value sorting based on standard qsort (not optimized)
  */
  struct my_data {
    int a;
    int b;
  };

  static int data_cmp(const void* a,const void* b) {
    struct my_data* da = (struct my_data*)a;
    struct my_data* db = (struct my_data*)b;

    da->a = *(int*)a;
    db->a = *(int*)b;

    return db->a - da->a;
  }

  void sortByKey(int* array_a,int* array_b,int nrows,int* Ns = nullptr) {

    int i;
    // struct my_data data[nrows];
    struct my_data* data;
    data = (struct my_data*)malloc(sizeof(struct my_data) * nrows);

#pragma omp parallel for private(i)
    for (i = 0; i < nrows; i++) {
      (data[i]).a = array_a[i];
      (data[i]).b = array_b[i];
    }

    // #pragma omp barrier 
    // qsort(data, sizeof(data) / sizeof(*data), sizeof(*data), data_cmp);
    qsort(data,sizeof(struct my_data) * nrows / sizeof(*data),sizeof(*data),data_cmp);

#pragma omp parallel for private(i)
    for (i = 0; i < nrows; i++) {
      array_a[i] = (data[i]).a;
      array_b[i] = (data[i]).b;
    }

    free(data);

  }






///////////////////////////////////////////////////////////////// BITONIC
// Comparator to sort in ascending order by key
bool compareAsc(const Pair<int, double>& a, const Pair<int, double>& b) {
    return a.key < b.key;
}

// Comparator to sort in descending order by key
bool compareDesc(const Pair<int, double>& a, const Pair<int, double>& b) {
    return a.key > b.key;
}

// Function to swap elements
template<typename T>
void swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

// Bitonic merge function
template<typename T>
void bitonicMerge(T* data, int start, int length, bool ascending) {
    if (length > 1) {
        int mid = length / 2;
        for (int i = start; i < start + mid; ++i) {
            if (ascending ? compareDesc(data[i], data[i + mid]) : compareAsc(data[i], data[i + mid])) {
                swap(data[i], data[i + mid]);
            }
        }
        bitonicMerge(data, start, mid, ascending);
        bitonicMerge(data, start + mid, mid, ascending);
    }
}

// Bitonic sort function
template<typename T>
void bitonicSort(T* data, int start, int length, bool ascending) {
    if (length > 1) {
        int mid = length / 2;
        // Sort first half in ascending order, second half in descending order
        bitonicSort(data, start, mid, true);
        bitonicSort(data, start + mid, mid, false);
        // Merge entire sequence in ascending or descending order
        bitonicMerge(data, start, length, ascending);
    }
}
//////////////////////////////////////////////////// BITONIC END


template<typename C, typename V>
struct Pair0 {
  C key;
  V val;
};

template<typename Tuple>
int cmpr(const void* a, const void* b) {
  const Tuple& t1 = *static_cast<const Tuple*>(a);
  const Tuple& t2 = *static_cast<const Tuple*>(b);
  return t1.key - t2.key;
}


// Utility function to get the digit at a specific place
int getDigit(int number, int place) {
    return (number / place) % 10;
}

// Counting sort based on the digit represented by 'place'
template<typename Tuple>
void countingSort(Tuple arr[], int n, int place) {
    Tuple *output = (Tuple*)malloc(n * sizeof(Tuple));
    int count[10] = {0};

    // Count occurrences of each digit
    for (int i = 0; i < n; i++) {
        int digit = getDigit(arr[i].key, place);
        count[digit]++;
    }

    // Compute the cumulative count
    for (int i = 1; i < 10; i++) {
        count[i] += count[i - 1];
    }

    // Place elements in sorted order based on the current digit
    for (int i = n - 1; i >= 0; i--) {
        int digit = getDigit(arr[i].key, place);
        output[count[digit] - 1] = arr[i];
        count[digit]--;
    }

    // Copy sorted elements back to original array
    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }

    free(output);
}

// Radix Sort function
template<typename Tuple>
void radixSort(Tuple arr[], int n) {
    // Find the maximum key to determine the number of digits
    int maxKey = arr[0].key;
    for (int i = 1; i < n; i++) {
        if (arr[i].key > maxKey) {
            maxKey = arr[i].key;
        }
    }

    // Perform counting sort for each digit (from least significant to most)
    for (int place = 1; maxKey / place > 0; place *= 10) {
        countingSort(arr, n, place);
    }
}



}