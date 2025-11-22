# =================================  TESTY  ===================================
# Testy do tego pliku zostały podzielone na dwie kategorie:
#
#  1. `..._invalid_input`:
#     - Sprawdzające poprawną obsługę nieprawidłowych danych wejściowych.
#
#  2. `..._correct_solution`:
#     - Weryfikujące poprawność wyników dla prawidłowych danych wejściowych.
# =============================================================================
import numpy as np
import scipy as sp


def is_diagonally_dominant(A: np.ndarray | sp.sparse.csc_array) -> bool | None:

    if not isinstance(A, (np.ndarray, sp.sparse.csc_array)):
        return None
    
    if isinstance(A, sp.sparse.csc_array):
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return None
        A_dense = A.toarray()
    else:
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return None
        A_dense = A

    n = A_dense.shape[0]

    for i in range(n):
        diag = abs(A_dense[i, i])
        row_sum = np.sum(np.abs(A_dense[i, :])) - diag

        if diag <= row_sum:
            return False

    return True



def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float | None:
    """Funkcja obliczająca normę residuum dla równania postaci: Ax = b."""
    
    if not isinstance(A, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(b, np.ndarray):
        return None
    
    if A.ndim != 2 or x.ndim != 1 or b.ndim != 1:
        return None
    
    m, n = A.shape
    if x.shape[0] != n or b.shape[0] != m:
        return None

    r = A @ x - b
    
    return float(np.linalg.norm(r, ord=2))
