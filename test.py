import numpy as np
from scipy.io._fast_matrix_market import mmread
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import matplotlib.pyplot as plt
from sksparse.cholmod import cholesky
# Creiamo una matrice sparsa simmetrica con valori lontani dalla diagonale
# Visualizziamo la matrice originale
A =  mmread("MatriciNonFattibili/Flan_1565/Flan_1565.mtx")
A = sp.csc_matrix(A)
sparsity = 1.0 - (A.nnz / float(A.shape[0] * A.shape[1]))
(A != A.T).nnz
print(f"Sparsità: {sparsity:.5%}")
R = cholesky(A)
