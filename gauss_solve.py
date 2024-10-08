import ctypes
import numpy as np
#from scipy.linalg import hilbert

gauss_library_path = './libgauss.so'

class NoImplementationInC(Exception):
    pass

def unpack(A):
    """ Extract L and U parts from A, fill with 0's and 1's """
    n = len(A)
    L = [[A[i][j] for j in range(i)] + [1] + [0 for j in range(i+1, n)]
         for i in range(n)]

    U = [[0 for j in range(i)] + [A[i][j] for j in range(i, n)]
         for i in range(n)]

    return L, U

def lu_c(A):
    """ Accepts a list of lists A of floats and
    it returns (L, U) - the LU-decomposition as a tuple.
    """
    # Load the shared library
    lib = ctypes.CDLL(gauss_library_path)

    # Create a 2D array in Python and flatten it
    n = len(A)
    flat_array_2d = [item for row in A for item in row]

    # Convert to a ctypes array
    c_array_2d = (ctypes.c_double * len(flat_array_2d))(*flat_array_2d)

    # Define the function signature
    lib.lu_in_place.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double))

    # Modify the array in C (e.g., add 10 to each element)
    lib.lu_in_place(n, c_array_2d)

    # Convert back to a 2D Python list of lists
    modified_array_2d = [
        [c_array_2d[i * n + j] for j in range(n)]
        for i in range(n)
    ]

    # Extract L and U parts from A, fill with 0's and 1's
    return unpack(modified_array_2d)

def lu_python(A):
    n = len(A)
    for k in range(n):
        for i in range(k,n):
            for j in range(k):
                A[k][i] -= A[k][j] * A[j][i]
        for i in range(k+1, n):
            for j in range(k):
                A[i][k] -= A[i][j] * A[j][k]
            A[i][k] /= A[k][k]

    return unpack(A)

# def plu_python(A):
#     n = len(A)
#     P = [[0 for _ in range(n)] for _ in range(n)]
    
#     # Place ones on the diagonal
#     for i in range(n):
#         P[i][i] = 1
    
#     for k in range(n-1):
#         max_row_index = k
#         max_value = abs(A[k][k])
#         for l in range(k+1, n):
#             if abs(A[l][k]) > max_value:
#                 max_value = abs(A[l][k])
#                 max_row_index = l    
#         if max_row_index != k:
#             A[k], A[max_row_index] = A[max_row_index], A[k]
#             P[k], P[max_row_index] = P[max_row_index], P[k]
#         for i in range(k,n):
#             for j in range(k):
#                 A[k][i] -= A[k][j] * A[j][i]
#         for i in range(k+1, n):
#             for j in range(k):
#                 A[i][k] -= A[i][j] * A[j][k]
#             A[i][k] /= A[k][k]
    
#     L, U = unpack(A)
    
#     return P, L, U

def plu_python(A):
    n = len(A)
    P = np.eye(n)
    U = np.array(A, dtype=float)
    L = np.zeros((n,n), dtype=float)
    
    for k in range(n-1):
        max_row_index = k
        max_value = abs(U[k][k])
        for l in range(k+1, n):
            if abs(U[l][k]) > max_value:
                max_value = abs(U[l][k])
                max_row_index = l    
        if max_row_index != k:
            U[[k, max_row_index]] = U[[max_row_index, k]]
            P[[k, max_row_index]] = P[[max_row_index, k]]
            L[[k, max_row_index], 0:k] = L[[max_row_index, k], 0:k] 
        for i in range(k+1, n):
            L[i][k] = U[i][k] / U[k][k]
            for j in range(k, n):
                U[i][j] -= L[i][k]*U[k][j]
    
    for m in range(n):
        L[m][m] = 1
    
    return P, L, U

def lu(A, use_c=False):
    if use_c:
        return lu_c(A)
    else:
        return lu_python(A)

def plu(A, use_c=False):
    if use_c:
        raise NoImplementationInC()
    else:
        return plu_python(A)

if __name__ == "__main__":

    def get_A():
        """ Make a test matrix """
        A = [[4.0, 9.0, 10.0],
              [14.0, 30.0, 34.0],
              [2.0, 3.0, 3.0]]
        return A

    A = get_A()

    P, L, U = plu(A, use_c = False)
    
    A = get_A()

    P, L, U = plu(A, use_c=True)
    
