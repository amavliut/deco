/*
  Different core function and utilities to compute FSAI preconditioner
*/

// Calculate the power of 2 for n
int find_htable_sz(float n_) {
    unsigned int n = static_cast<unsigned int>(ceil(n_));
    unsigned int power = 1;
    while (power < n) power <<= 1;
    return static_cast<int>(power);
}

template <typename T,typename C>
inline void hashmap_symbolic(T& nz,T* hashtb,const C key,const int sz) {

  int hash = (key * HASH_SCAL) % sz;
  if (hashtb[hash] != key) {
    while (hashtb[hash] != key && hashtb[hash] != -1) hash = (hash + 1) % sz;
    (hashtb[hash] == key) ? (void()) : (hashtb[hash] = key, nz++, void());
  }
}

template <typename T,typename C>
inline void hashmap_symbolic_bit(T& nz,T* hashtb,const C key,const int sz_1) {

  int hash = (key * HASH_SCAL) & sz_1;
  if (hashtb[hash] != key) {
    while (hashtb[hash] != key && hashtb[hash] != -1) hash = (hash + 1) & sz_1;
    (hashtb[hash] == key) ? (void()) : (hashtb[hash] = key, nz++, void());
  }
}

template <typename T,typename C>
inline void hashmap_symbolic_safe(T& nz,T* hashtb,const C key,const int sz) {

  int hash = (key * HASH_SCAL) % sz;
  while (true) {
    T old = hashtb[hash];
    if (old == -1) {
      hashtb[hash] = key;
      nz++;
      return;
    }
    if (old == key) {
      return;
    }
    hash = (hash + 1) % sz;
  }
}


template <typename T,typename C>
inline void hashmap_pattern(T* hashtb,const C key,const int sz) {

  int hash = (key * HASH_SCAL) % sz;
  if (hashtb[hash] != key) {
    while (hashtb[hash] != key && hashtb[hash] != -1) hash = (hash + 1) % sz;
    (hashtb[hash] == key) ? (void()) : (hashtb[hash] = key, void());
  }
}

template <typename T,typename C>
inline void hashmap_pattern_bit(T* hashtb,const C key,const int sz_1) {

  int hash = (key * HASH_SCAL) & sz_1;
  if (hashtb[hash] != key) {
    while (hashtb[hash] != key && hashtb[hash] != -1) hash = (hash + 1) & sz_1;
    (hashtb[hash] == key) ? (void()) : (hashtb[hash] = key, void());
  }
}



// input:
// arr    -- array to search in
// right  -- array's length -1
// target -- value to search for
// mid    -- index position of the 'target' in the 'arr'
template<typename T>
T binary_search(const T* arr, T right, const T target){
   T left = 0;
   T mid = right>>1;
   while (right-left > 1){
      arr[mid] > target ? right = mid : left = mid;
      mid = (left+right)>>1;
   }

   if (arr[left] == target)       return left;
   else if (arr[right] == target) return right;
   else                           return -1;
}


template <typename T,typename C>
inline void hashmap_insert(T* hashtb_key, T* hashtb_ind,const C key, const C ind,const int sz_1) {

  int hash = (key * HASH_SCAL) & sz_1;
  while (hashtb_key[hash] >= 0) hash = (hash + 1) & sz_1;
  if (hashtb_key[hash] == -1) {
    hashtb_key[hash] = key;
    hashtb_ind[hash] = ind;
  }
}

template <typename T,typename C>
inline int hashmap_check(T* hashtb_key,const C key,const int sz_1) {

  int hash = (key * HASH_SCAL) & sz_1;
  if (hashtb_key[hash] != key) {
    while (hashtb_key[hash] != key && hashtb_key[hash] != -1) hash = (hash + 1) & sz_1;
  }
  return hash;
}




// WITH VALUES
template <typename T,typename C, typename V>
inline void hashmap(T hashtb,const C key, const V val,const int sz) {

  int hash = (key * HASH_SCAL) % sz;
  if (hashtb[hash].key != key) {
    while (hashtb[hash].key != key && hashtb[hash].key != -1) hash = (hash + 1) % sz;
    (hashtb[hash].key == key) ? (void()) : (hashtb[hash].key = key, void());
  }
  hashtb[hash].val += val;
}

// WITH VALUES
template <typename T,typename C, typename V>
inline void hashmap_bit(T hashtb,const C key, const V val,const int sz_1) {

  int hash = (key * HASH_SCAL) & sz_1;
  while (hashtb[hash].key != key && hashtb[hash].key != -1) {
    hash = (hash + 1) & sz_1;
  }

  hashtb[hash].key = key;
  hashtb[hash].val += val;
}

template <typename T,typename C, typename V>
inline void hashmap_bit2(T hashtb,const C key, const V val,const int sz_1) {

  int hash = (key * HASH_SCAL) & sz_1;
  for (; ; hash = (hash + 1) & sz_1){
    C kk = hashtb[hash].key;
    if (kk == key || kk == -1) {
      hashtb[hash].key = key;
      hashtb[hash].val += val;
    }
  }
}