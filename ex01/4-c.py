import numpy as np
import matplotlib.pyplot as plt

def neville_matrix(f, m, n):
    num_nodes = m + n + 1
    i_values = np.arange(num_nodes)
    h_values = 2.0**(-i_values)
    
    N = np.zeros((m + 1, n + 1))
    
    # k=0 column (Initial values: N_{j, 0} = f(h_j))
    N[:, 0] = f(h_values[:m + 1])
    
    # Neville Recursion for k = 1 to n
    for k in range(1, n + 1):
        # We only need to compute the first m+1-k rows in column k
        for j in range(m + 1 - k): 
            h_j = h_values[j]
            h_j_plus_k = h_values[j + k]
            
            # N_{j,k} = (-h_{j+k} * N_{j, k-1} + h_{j} * N_{j+1, k-1}) / (h_{j} - h_{j+k})
            N[j, k] = (-h_j_plus_k * N[j, k - 1] + h_j * N[j + 1, k - 1]) / (h_j - h_j_plus_k)

    return N, h_values[:m + n + 1]

def f_c(h):
    return (np.exp(h) - 1.0) / h

m = 10
n = 2

# Calculate the matrix N
N, h_grid = neville_matrix(f_c, m, n)

# The h-values relevant for the plot are h_j for j = 0 to m (the first m+1 nodes)
h = h_grid[:m + 1]


# Calculate the absolute errors for the first three columns (k=0, 1, 2)
# The matrix N has (m+1) rows, but column k only has entries up to row m-k.

# Error for k=0 (Column 1)
# N[:, 0] has m+1 entries (j=0 to m)
error_k0 = np.abs(N[:, 0] - 1) 

# Error for k=1 (Column 2)
# N[:, 1] has m entries (j=0 to m-1)
error_k1 = np.abs(N[:m, 1] - 1) 
h_k1 = h[:m] # Corresponding h-values for k=1

# Error for k=2 (Column 3)
# N[:, 2] has m-1 entries (j=0 to m-2)
error_k2 = np.abs(N[:m-1, 2] - 1)
h_k2 = h[:m-1] # Corresponding h-values for k=2

# Create the log-log plot
plt.figure(figsize=(10, 6))
plt.loglog(h, error_k0, 'o-', label=r'k=0: $N_{j,0}$ (Error $\propto h^1$)')
plt.loglog(h_k1, error_k1, 's-', label=r'k=1: $N_{j,1}$ (Error $\propto h^2$)')
plt.loglog(h_k2, error_k2, '^-', label=r'k=2: $N_{j,2}$ (Error $\propto h^3$)')

plt.xlabel(r'$h_j$ (Log Scale)')
plt.ylabel(r'Absolute Error $|N_{j, k} - 1|$ (Log Scale)')
plt.title("Convergence of Neville's Scheme Approximations to $\lim_{h \to 0} f(h) = 1$")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.gca().invert_xaxis() # h decreases as j increases
plt.show()