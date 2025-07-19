import os
import numpy as np
import time
import scipy.linalg as lg
from scipy.io._fast_matrix_market import mmread
from sksparse.cholmod import cholesky
from memory_profiler import memory_usage


def run_cholesky(A):
    return cholesky(A)

def run_solver(R, b):
    return R(b)

def main():
    file_names =  os.listdir("Matrici/")
    output_filename =  'python_linux_output_result.csv' if os.name == 'posix' else "python_windows_file_result.csv"
    data_to_save = open(output_filename,'w')
    data_to_save.write("Matrix Name,Matrix size,Cholesky Time (ms),Cholesky Memory,Solution Time (ms),Solution Memory,Error\n")
    for file_name in file_names:
        A =  mmread(f"Matrici/{file_name}")
        A = A.tocsc()
        xe = np.ones(A.shape[0])
        b = A @ xe
        cholesky_mem, R = memory_usage((run_cholesky, (A,)), retval=True, max_usage=True, interval=0.01)
        start_time = time.perf_counter()
        R = cholesky(A)
        elapsed_time_cholesky = (time.perf_counter() - start_time) * 1000
        solution_mem, x = memory_usage((run_solver, (R, b)), retval=True, max_usage=True, interval=0.01)
        start_time = time.perf_counter()
        x = R(b)
        elapsed_time_solution= (time.perf_counter() - start_time) * 1000
        error = lg.norm(x - xe) / lg.norm(xe)
        data_to_save.write(f"{file_name},{A.nnz},{elapsed_time_cholesky:3e},{cholesky_mem:3e},{elapsed_time_solution:3e},{solution_mem:3e},{error}\n")
        print(f"{file_name} processed...")

if __name__ == '__main__':
    main()
