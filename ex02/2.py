import math
import numpy as np
import matplotlib.pyplot as plt 

def neville_interpolation(x_points, y_points, x):
    """
    Perform polynomial interpolation at a specific x using Neville's scheme.

    Parameters
    ----------
    x_points : list or array of float
        Known x-values (must all be distinct)
    y_points : list or array of float
        Known f(x)-values
    x : float
        The point where we want to interpolate f(x)

    Returns
    -------
    fx : float
        Interpolated value f(x)
    N : 2D numpy array
        The Neville interpolation table (each column represents a step)
    """

    n = len(x_points)
    N = np.zeros((n, n))
    N[:, 0] = y_points  

    # Fill the Neville table
    for j in range(1, n):
        for i in range(n - j):
            xi = x_points[i]
            xj = x_points[i + j]
            N[i, j] = ((x - xj) * N[i, j - 1] - (x - xi) * N[i + 1, j - 1]) / (xi - xj)

    return N[0, n - 1], N

TRUE_DERIVATIVE = 2.0
X0 = math.pi / 4

def f(x):
    return np.tan(x)

def D_sym(h, x0):
    return (f(x0 + h) - f(x0 - h)) / (2 * h)

h_values = np.array([2.0**(-i) for i in range(11)]) # h_0=1 to h_10=1/1024
D0_values = D_sym(h_values, X0)

# 4. Perform Extrapolation
# Note: For Romberg, the independent variable is usually h^2, not h. 
# We'll use h as requested, but a standard Romberg would use h_values**2
# The 'x' value in Neville's scheme is 0, since we are extrapolating to h=0.
interpolated_value, Neville_table = neville_interpolation(h_values, D0_values, 0.0)

# 5. Calculate Errors for first 4 columns (m=0, 1, 2, 3)
# Error for m=0 (using all 11 points)
Error_m0 = np.abs(Neville_table[:, 0] - TRUE_DERIVATIVE)
# Error for m=1 (using 10 points: h_0 to h_9)
Error_m1 = np.abs(Neville_table[:-1, 1] - TRUE_DERIVATIVE)
# Error for m=2 (using 9 points: h_0 to h_8)
Error_m2 = np.abs(Neville_table[:-2, 2] - TRUE_DERIVATIVE)
# Error for m=3 (using 8 points: h_0 to h_7)
Error_m3 = np.abs(Neville_table[:-3, 3] - TRUE_DERIVATIVE)


# --- 2. Create the Log-Log Plot with Labels ---

plt.figure(figsize=(10, 6))

# Plot the Data Lines
# Use markers ('o', 's', '^', 'd') to distinguish the lines
plt.loglog(h_values, Error_m0, 'o-', label=r'$m=0$ (Rate $\approx O(h^2)$)')
plt.loglog(h_values[:-1], Error_m1, 's-', label=r'$m=1$ (Rate $\approx O(h^4)$)')
plt.loglog(h_values[:-2], Error_m2, '^-', label=r'$m=2$ (Rate $\approx O(h^6)$)')
plt.loglog(h_values[:-3], Error_m3, 'd-', label=r'$m=3$ (Rate $\approx O(h^8)$)')

# Plot the Auxiliary Reference Lines (Slopes)
# Use 'k--' (black dashed line) for reference slopes
h_ref = h_values 
# Scale factor 'C' applied to the reference lines to make them visible and parallel
C = 1e-1 
plt.loglog(h_ref, C * h_ref**2, 'k--', alpha=0.5, label=r'Ref. Slope $h^2$')
plt.loglog(h_ref, C * h_ref**4, 'g--', alpha=0.5, label=r'Ref. Slope $h^4$')
plt.loglog(h_ref, C * h_ref**6, 'b--', alpha=0.5, label=r'Ref. Slope $h^6$')
plt.loglog(h_ref, C * h_ref**8, 'r--', alpha=0.5, label=r'Ref. Slope $h^8$')

# --- 3. Add Labels, Title, and Grid ---

plt.xlabel(r'Step Size $h = 2^{-i}$', fontsize=14)
plt.ylabel(r'Absolute Error $|\text{Approximation} - 2|$', fontsize=14)
plt.title(r'Convergence of Extrapolated Derivatives using Neville''s Scheme', fontsize=16)

plt.legend(loc='lower right', fontsize=10)
plt.grid(True, which="both", ls="--", alpha=0.7)

# The 'What convergence rates do you observe?' part is answered by looking at the plot:
# The slope of each data line should match the slope of the corresponding reference line.

plt.show()