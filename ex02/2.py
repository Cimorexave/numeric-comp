import numpy as np
import matplotlib.pyplot as plt 

def neville_interpolation(x_points, y_points, x):

    n = len(x_points)
    N = np.zeros((n, n))
    N[:, 0] = y_points  

    for j in range(1, n):
        for i in range(n - j):
            xi = x_points[i]
            xj = x_points[i + j]
            N[i, j] = ((x - xj) * N[i, j - 1] - (x - xi) * N[i + 1, j - 1]) / (xi - xj)

    return N[0, n - 1], N

TRUE_DERIVATIVE = 2.0
X0 = np.pi / 4

def f(x):
    return np.tan(x)

def D_sym(h, x0):
    return (f(x0 + h) - f(x0 - h)) / (2 * h)

h_values = np.array([2.0**(-i) for i in range(11)]) # h_0=1 to h_10=1/1024
D0_values = D_sym(h_values, X0)

interpolated_value, Neville_table = neville_interpolation(h_values, D0_values, 0.0)

# errors for the first 4 columns m=0, 1, 2, 3
# Error for m=0 (using all 11 points)
error_m0 = np.abs(Neville_table[:, 0] - TRUE_DERIVATIVE)
# Error for m=1 (using 10 points: h_0 to h_9)
error_m1 = np.abs(Neville_table[:-1, 1] - TRUE_DERIVATIVE)
# Error for m=2 (using 9 points: h_0 to h_8)
error_m2 = np.abs(Neville_table[:-2, 2] - TRUE_DERIVATIVE)
# Error for m=3 (using 8 points: h_0 to h_7)
error_m3 = np.abs(Neville_table[:-3, 3] - TRUE_DERIVATIVE)


plt.figure(figsize=(10, 6))
plt.loglog(h_values, error_m0, 'o-', label=r'$m=0$')
plt.loglog(h_values[:-1], error_m1, 's-', label=r'$m=1$')
plt.loglog(h_values[:-2], error_m2, '^-', label=r'$m=2$')
plt.loglog(h_values[:-3], error_m3, 'd-', label=r'$m=3$')

# reference lines 
h_ref = h_values 
plt.loglog(h_ref, h_ref**2, 'k--', alpha=0.5, label=r'ref $h^2$')
plt.loglog(h_ref, h_ref**4, 'g--', alpha=0.5, label=r'ref $h^4$')
plt.loglog(h_ref, h_ref**6, 'b--', alpha=0.5, label=r'ref $h^6$')
plt.loglog(h_ref, h_ref**8, 'r--', alpha=0.5, label=r'ref $h^8$')

plt.xlabel(r'step $h = 2^{-i}$', fontsize=14)
plt.ylabel('absolute error', fontsize=14)
plt.title(r'extrapolated derivatives using Nevilles Scheme', fontsize=16)

plt.legend(loc='lower right', fontsize=10)
plt.grid(True, which="both", ls="--", alpha=0.7)

plt.show()