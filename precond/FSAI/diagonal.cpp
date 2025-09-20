// find the scaling factor D = diag(FL A FU)
// FL and FU are both lower diagonal matrices (FU is yet to be transposed)
// For each diagonal value, loop over the columns of FL. At the corresponding FL columns,
// find the column array as a product of A*FU. Since FU is transposed, then you can traverse
// both A and FU at the same time over the shortest merge path
template <typename R,typename C,typename V >
void MxM_diag(shared_ptr<CSR<R,C,V>> FL,shared_ptr<CSR<R,C,V>> A,shared_ptr<CSR<R,C,V>> FU,double* D) {

  C   n = A->n;
  R* rA = A->r;
  C* cA = A->c;
  V* vA = A->v;

  R* rFL = FL->r;
  C* cFL = FL->c;
  V* vFL = FL->v;
  V* vFU = FU->v;

#pragma omp parallel for
  for (C i = 0; i < n; ++i) { // set up diag vals, FL & FU are stored as lower diagonal matrices
    R ind = rFL[i + 1] - 1;
    vFL[ind] = 1.0;
    vFU[ind] = 1.0;
  }

#pragma omp parallel for
  for (C i = 0; i < n; ++i) {
    R aFL = rFL[i];
    R bFL = rFL[i + 1];
    V sumFLAFU = 0.;
    for (R j = aFL; j < bFL; ++j) { // loop over FL
      V valFL = vFL[j];
      if (valFL != 0.) {
        C colFL = cFL[j];
        R aA = rA[colFL];
        R bA = rA[colFL + 1];
        R aFU = aFL;
        V sumAFU = 0.;
        while (aA < bA && aFU < bFL) { // common loop over A and FU
          if (cA[aA] == cFL[aFU]) {
            sumAFU += vA[aA] * vFU[aFU];
            aA++;
            aFU++;
          } else (cA[aA] < cFL[aFU]) ? aA++ : aFU++;
        }
        sumFLAFU += valFL * sumAFU;
      }
    }
    D[i] = (sumFLAFU < 0.) ? (-1. / sqrt(fabs(sumFLAFU))) : (1. / sqrt(sumFLAFU));
  }

}


// FL is signed scaled
// FU is unsigned scaled
void Jacobi_scale(int nrows,  double *D, int *iatFL, double *coefFL, double *coefFU){

    int i,j,iaa,iab;

    #pragma omp parallel for private(i,iaa,iab,j)
    for ( i = 0; i < nrows; ++i){
        iaa = iatFL[i];
        iab = iatFL[i+1];
        auto dd = D[i];
        auto dd_abs = fabs(dd);
        for ( j = iaa; j < iab; ++j){
            coefFL[j] = coefFL[j] * dd;
            coefFU[j] = coefFU[j] * dd_abs;
        }
        
    }

}