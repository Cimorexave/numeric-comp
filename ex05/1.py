import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss

def composite_gauss(f, n, L, q):
    """
    Composite Gauss rule for integration over (0, 1).
    n: number of Gauss points per subinterval.
    L: number of subintervals.
    q: parameter for subinterval boundaries.
    """
    # standard Gaussian points and weights on [-1, 1]
    # x_G : standard points,
    #  w_G : standard weights
    x_G, w_G = leggauss(n)
    
    # subinterval boundaries (xj)
    powers = np.arange(L - 1, -1, -1)
    interior_points = q ** powers
    x_j = np.concatenate(([0.0], interior_points))
    
    total_integral = 0.0
    
    # L subintervals
    for j in range(L):
        a_j = x_j[j]
        b_j = x_j[j+1]
        
        # transform standard points
        x_i_j = (a_j + b_j) / 2.0 + ((b_j - a_j) / 2.0) * x_G
        
        integral_j = np.sum(((b_j - a_j) / 2.0) * w_G * f(x_i_j))
        
        total_integral += integral_j
        
    return total_integral

def f_c(x):
    # x_safe = np.where(x == 0, 1e-20, x) -> unnecessary
    return x**0.1 * np.log(x)


I_exact = -1.0 / 1.21

L_values = n_values = np.arange(1, 21) # n = L = 1...20
q_values = [0.5, 0.15, 0.05]

# store errors for plotting
error_data = {q: [] for q in q_values}

for q in q_values:
    for n in n_values:
        L = n 
        
        # approximation
        I_approx = composite_gauss(f_c, n, L, q)
        
        # absolute error
        error = np.abs(I_approx - I_exact)
        error_data[q].append(error)


# plotting 
plt.figure(figsize=(10, 6))

for q, errors in error_data.items():
    plt.semilogy(n_values, errors, label=f'q = {q}', marker='o', linestyle='-')

plt.title('gauss quad err vs. $n$ (composite gauss)')
plt.xlabel('$n = L$ (points count/subintervals)')
plt.ylabel('abs error $|I_{approx} - I_{exact}|$ (semilogy)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# q = 0.15 has the best convergence rate:
# making the q too big like q=0.5 or q=1 will make the points equally distributed
# and not counting for the discontinuity point near 0
# making the q too small makes the concentration near 0 too much which is good for
# approximation near the discontinuity but since the points are limited (L=n)
# makes the approximation points near the other areas like 1 for the function too dispersed
# and a worse approximation. q = 0.15 has the ideal balance between the two

# finding error formula
fit_results = {}

for q, errors in error_data.items():
    # make error linear: Y = log(Error), X = n
    Y = np.log(errors)
    X = n_values
    
    # fit Y = m*X + k, where m = -b and k = log(C)
    # degree of the polynomial is 1 (linear fit)
    m, k = np.polyfit(X, Y, 1)
    
    b_rate = -m
    C_factor = np.exp(k)
    
    fit_results[q] = {'b': b_rate, 'C': C_factor}
    
    print(f"For q = {q} b: {b_rate:.4f} C: {C_factor:.4e}")

