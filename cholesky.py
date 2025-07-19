from scipy.sparse import  csc_matrix, lil_matrix,random,eye,csgraph,linalg,issparse
from scipy.io import mmread
import numpy as np
import os
import matplotlib.pyplot as plt
from sksparse.cholmod import cholesky
import time
from multiprocessing import process
'''def cholesky_sparse(A)-> tuple:
    iters = 0
    A_star =  A.tolil(copy=True)
    print(A_star.nnz)
    n = A.shape[0]
    R = lil_matrix((n,n),dtype=np.float64)
    print(n)
    for k in range(n):
        iters += 1
        R[k,k] =  np.sqrt(A_star[k,k])
        for j in range(k+1,n):
            R[k,j] = A_star[k,j] / R[k,k]
        p = 1.0 / A_star[k,k]

        for j in range(k+1,n):
            if A_star[j,k] !=0 :

                for i in range(k+1,n):
                    A_star[i,j] = A_star[i,j] - (p * A_star[i,k] * A_star[k,j])


    return R.tocsc(),iters

'''
'''def cholesky_sparse(A)-> tuple:
    iters = 0
    A_star =  A.tolil(copy=True)
    print(A_star.nnz)
    n = A.shape[0]
    R = lil_matrix((n,n),dtype=np.float64)
    print(n)
    for k in range(n):
        R[k,k] = np.sqrt(A_star[k,k])
        # Prendo solo gli indici j > k dove A_star[k,j] non zero
        js = A_star.rows[k]
        js = [j for j in js if j > k]

        for j in js:
            R[k,j] = A_star[k,j] / R[k,k]

        p = 1.0 / A_star[k,k]

        # per ogni riga i > k che ha valore non zero in colonna k
        is_ = A_star[:, k].nonzero()[0]
        is_ = [i for i in is_ if i > k]

        for j in js:
            for i in is_:
                # aggiornamento solo se A_star[i,j] è non zero
                if A_star[i,j] != 0:
                    A_star[i,j] -= p * A_star[i,k] * A_star[k,j]


    return R.tocsc(),iters

'''
# Matrice sparsa simmetrica definita positiva

os.getcwd()



folder = "../Matrici"
dense_matrices = {}
sparse_matrices = {}



#A = mmread("../Matrici/apache2/apache2.mtx").tocsc()



# plt.figure()
# plt.spy(A, markersize=10)
# plt.title('Matrice originale')
# plt.show()


#p = csgraph.reverse_cuthill_mckee(A)

#A = A[p,:][:,p]

# plt.figure()
# plt.spy(A, markersize=10)
# plt.title('Matrice permutat')
# plt.show()



folder = "Matrici"
results = []
for root, dirs, files in os.walk(folder):
    for file in files:
        if file.endswith(".mtx"):
            full_path = os.path.join(root, file)
            print(f"Processing {file} ...")
            A = mmread(full_path).tocsc()
            print(A.shape)
            if not issparse(A):
                print(f"Skipping {file}: not sparse")
                continue
            try:
                start = time.time()
                iters= 0
                R = cholesky(A).L()
                #R, iters = cholesky_sparse(A)
                end = time.time()
                elapsed = end - start
                print(f"Done {file}: time={elapsed:.4f}s, iterations={iters}")
                results.append((file, elapsed, iters))
            except Exception as e:
                print(f"Error processing {file}: {e}")


if results:
    files, times, iterations = zip(*results)
    # Print values
    print("\nBenchmark Results:")
    for f, t, it in zip(files, times, iterations):
          print(f"{f}: time = {t:.4f} s, iterations = {it}")
    # Plot times
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.bar(range(len(times)), times)
    plt.xticks(range(len(times)), files, rotation=90)
    plt.ylabel("Time (seconds)")
    plt.title("Cholesky Sparse: Execution Time")
    plt.tight_layout()

    # Plot iterations
    plt.subplot(1,2,2)
    plt.bar(range(len(iterations)), iterations)
    plt.xticks(range(len(iterations)), files, rotation=90)
    plt.ylabel("Iterations (outer loop count)")
    plt.title("Cholesky Sparse: Iterations")
    plt.tight_layout()


    plt.tight_layout()
    plt.savefig("benchmark_cholesky.png")
    print("Plot saved as benchmark_cholesky.png")
else:
    print("No results to plot.")
