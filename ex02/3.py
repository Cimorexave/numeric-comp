import numpy as np
import matplotlib.pyplot as plt
import math

def f(x):
    return 1.0 / (4.0 - x**2)

def neville_interpolation(x_points, y_points, x_eval):
    """
    Performs polynomial interpolation at a specific point x_eval 
    using Neville's scheme, returning the interpolated value and the full table.
    """
    n = len(x_points)
    N = np.zeros((n, n))
    N[:, 0] = y_points  

    for j in range(1, n):
        for i in range(n - j):
            xi = x_points[i]        # x-coordinate of the first point used
            xj = x_points[i + j]    # x-coordinate of the last point used
            
            N[i, j] = ((x_eval - xj) * N[i, j - 1] - (x_eval - xi) * N[i + 1, j - 1]) / (xi - xj)

    # N[0, n-1] is the best accurate interpolated value
    fx = N[0, n - 1]
    return fx, N

def taylor_2n_f(x, n):
    term = 1.0 
    poly_sum = term
    
    for k in range(1, n + 1):
        term = term * (x**2 / 4.0)
        poly_sum = poly_sum + term
        
    return poly_sum / 4.0


N_MAX = 20
X_EVAL_COUNT = 100
TRUE_RANGE = np.linspace(-1.0, 1.0, X_EVAL_COUNT)
TRUE_F_VALUES = f(TRUE_RANGE)
N_VALUES = np.arange(1, N_MAX + 1)

max_error_cheb = np.zeros(N_MAX)
max_error_taylor = np.zeros(N_MAX)

for n in N_VALUES:
    # chebyshev nodes
    k_indices = np.arange(n + 1)
    cheb_nodes = np.cos((2 * k_indices + 1) * np.pi / (2 * n + 2))
    
    cheb_f_values = f(cheb_nodes)
    
    # Calculate Chebyshev Interpolant at 100 evaluation points (I_n^Cheb f(z_j))
    cheb_interpolated_values = np.zeros(X_EVAL_COUNT)
    for j in range(X_EVAL_COUNT):
        x_eval = TRUE_RANGE[j]
        cheb_interpolated_values[j], _ = neville_interpolation(cheb_nodes, cheb_f_values, x_eval)
        
    cheb_error = np.abs(TRUE_F_VALUES - cheb_interpolated_values)
    max_error_cheb[n-1] = np.max(cheb_error)
    
    taylor_values = taylor_2n_f(TRUE_RANGE, n)
    
    taylor_error = np.abs(TRUE_F_VALUES - taylor_values)
    max_error_taylor[n-1] = np.max(taylor_error)


    print(f"taylor errors at n={n}: {taylor_error}")
    print(f"chebyshev errors at n={n}: {cheb_error}")
    print(f"Completed degree n={n}. Max Cheb Error: {max_error_cheb[n-1]:.2e}, Max Taylor Error: {max_error_taylor[n-1]:.2e}")


plt.figure(figsize=(10, 6))

plt.semilogy(N_VALUES, max_error_cheb, 'ro-', label='Chebyshev Interpolant')
plt.semilogy(N_VALUES, max_error_taylor, 'bs--', label='Taylor Polynomial')

plt.xlabel('n', fontsize=14)
plt.ylabel('max abs err', fontsize=14)
plt.title('taylor vs chebyshev', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.show()

