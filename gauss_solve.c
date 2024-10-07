#include "gauss_solve.h"
#include <math.h>

void gauss_solve_in_place(const int n, double A[n][n], double b[n])
{
  for(int k = 0; k < n; ++k) {
    for(int i = k+1; i < n; ++i) {
      /* Store the multiplier into A[i][k] as it would become 0 and be
	 useless */
      A[i][k] /= A[k][k];
      for( int j = k+1; j < n; ++j) {
	A[i][j] -= A[i][k] * A[k][j];
      }
      b[i] -= A[i][k] * b[k];
    }
  } /* End of Gaussian elimination, start back-substitution. */
  for(int i = n-1; i >= 0; --i) {
    for(int j = i+1; j<n; ++j) {
      b[i] -= A[i][j] * b[j];
    }
    b[i] /= A[i][i];
  } /* End of back-substitution. */
}

void lu_in_place(const int n, double A[n][n])
{
  for(int k = 0; k < n; ++k) {
    for(int i = k; i < n; ++i) {
      for(int j=0; j<k; ++j) {
	/* U[k][i] -= L[k][j] * U[j][i] */
	A[k][i] -=  A[k][j] * A[j][i]; 
      }
    }
    for(int i = k+1; i<n; ++i) {
      for(int j=0; j<k; ++j) {
	/* L[i][k] -= A[i][k] * U[j][k] */
	A[i][k] -= A[i][j]*A[j][k]; 
      }
      /* L[k][k] /= U[k][k] */
      A[i][k] /= A[k][k];	
    }
  }
}

void lu_in_place_reconstruct(int n, double A[n][n])
{
  for(int k = n-1; k >= 0; --k) {
    for(int i = k+1; i<n; ++i) {
      A[i][k] *= A[k][k];
      for(int j=0; j<k; ++j) {
	A[i][k] += A[i][j]*A[j][k];
      }
    }
    for(int i = k; i < n; ++i) {
      for(int j=0; j<k; ++j) {
	A[k][i] +=  A[k][j] * A[j][i];
      }
    }
  }
}

void plu(int n, double A[n][n], int P[n]) {
    for (int i = 0; i < n; i++) {
        P[i] = i;
    }
    
    double L[n][n];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            L[i][j] = 0.0;
        }
    }
    for (int k = 0; k < n; k++) {
        int max_row_index = k;
        double max_value = fabs(A[k][k]);

        for (int l = k + 1; l < n; l++) {
            if (fabs(A[l][k]) > max_value) {
                max_value = fabs(A[l][k]);
                max_row_index = l;
            }
        }
        if (max_row_index != k) {
            for (int j = 0; j < n; j++) {
                SWAP(A[k][j], A[max_row_index][j], double);
            }
            SWAP(P[k], P[max_row_index], int);
        }

        for (int i = k + 1; i < n; i++) {
            L[i][k] = A[i][k] / A[k][k];
            for (int j = k; j < n; j++) {
                A[i][j] -= L[i][k] * A[k][j];
            }
        }
    }
    
    for (int i = 0; i < n; i++) {
        L[i][i] = 1.0;
    }
}
