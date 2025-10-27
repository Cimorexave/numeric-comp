import numpy as np
import matplotlib.pyplot as plt

a, b = -5.0, 5.0
n_values = [10, 20, 40]

def f(x):
    return 1.0 / (1.0 + x**2)

def generate_nodes(n, point_type):
    i = np.arange(n + 1)
    if point_type == 'unif':
        return a + (b - a) * (i / n)
    elif point_type == 'cheb':
        return b * np.cos((i + 0.5) * np.pi / (n + 1))

def lagrange_basis(x_eval, nodes, i):
    x_i = nodes[i]
    
    mask = np.arange(len(nodes)) != i
    nodes_j = nodes[mask]  
    
    x_eval_reshaped = x_eval[:, np.newaxis] 
    
    numerator = np.prod(x_eval_reshaped - nodes_j, axis=1)
    denominator = np.prod(x_i - nodes_j)
    return numerator / denominator

def lagrange_interpolant(x_eval, nodes, y_nodes):
    n = len(nodes) - 1
    P_n_x = 0.0
    for i in range(n + 1):
        L_i_x = lagrange_basis(x_eval, nodes, i)
        P_n_x += y_nodes[i] * L_i_x
    return P_n_x

def lebesgue_constant(nodes):
    n = len(nodes) - 1
    
    x_test = np.linspace(a, b, 1000)
    
    L_x = 0.0
    for i in range(n + 1):
        L_i_x_test = lagrange_basis(x_test, nodes, i)
        L_x += np.abs(L_i_x_test)
        
    Lambda_n = np.max(L_x)
    return Lambda_n



print(f"--- a) ---")
for n_a in [10,20,40]:

    x_unif_a = generate_nodes(n_a, 'unif')
    y_unif_a = f(x_unif_a)

    x_cheb_a = generate_nodes(n_a, 'cheb')
    y_cheb_a = f(x_cheb_a)

    x_plot = np.linspace(a, b, 500)

    y_true = f(x_plot)

    y_unif_plot = lagrange_interpolant(x_plot, x_unif_a, y_unif_a)
    y_cheb_plot = lagrange_interpolant(x_plot, x_cheb_a, y_cheb_a)

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_true, 'k', linewidth=3, label='$f(x) = 1/(1+x^2')
    plt.plot(x_plot, y_unif_plot, 'r--', linewidth=1.5, label='Uniform Interpolant $P_{10}^{\\text{unif}}(x)$')
    plt.plot(x_plot, y_cheb_plot, 'b:', linewidth=2, label='Chebyshev Interpolant $P_{10}^{\\text{Cheb}}(x)$')
    plt.scatter(x_unif_a, y_unif_a, marker='o', s=15, color='r')
    plt.scatter(x_cheb_a, y_cheb_a, marker='x', s=20, color='b')

    plt.title(f'Lagrange Interpolation for $n={n_a}$')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.grid(True)
    plt.legend()
    plt.ylim(-1, 2)
    plt.show()

print("\n" + "="*80 + "\n")



print("--- b)  ---")
lambda_n_unif = {}
lambda_n_cheb = {}

for n in n_values:
    x_unif = generate_nodes(n, 'unif')
    lambda_unif = lebesgue_constant(x_unif)
    lambda_n_unif[n] = lambda_unif
    print(f"n={n}: Lambda_n^unif = {lambda_unif:.2f}")

    x_cheb = generate_nodes(n, 'cheb')
    lambda_cheb = lebesgue_constant(x_cheb)
    lambda_n_cheb[n] = lambda_cheb
    print(f"n={n}: Lambda_n^Cheb = {lambda_cheb:.2f}")

n_list = np.array(n_values)
lambda_unif_list = np.array([lambda_n_unif[n] for n in n_list])
lambda_cheb_list = np.array([lambda_n_cheb[n] for n in n_list])

plt.figure(figsize=(10, 6))
plt.semilogy(n_list, lambda_unif_list, 'ro-', linewidth=2, label='Uniform Points $\\Lambda_n^{\\text{unif}}$')
plt.semilogy(n_list, lambda_cheb_list, 'bs--', linewidth=2, label='Chebyshev Points $\\Lambda_n^{\\text{Cheb}}$')

plt.xlabel('$n$')
plt.ylabel('$log(\\Lambda_n)$')
plt.xticks(n_list)
plt.grid(True, which="both", ls="-", alpha=0.6)
plt.legend()
plt.show()

print("\n" + "="*80 + "\n")


print(" --- c) ---")
n_fit = np.array([20, 40])
Lambda_fit = np.array([lambda_n_unif[n] for n in n_fit])

coeffs = np.polyfit(n_fit, np.log(Lambda_fit), 1)
b = coeffs[0]      
ln_C = coeffs[1]   
C = np.exp(ln_C)

print(f"b= {b:.4f}")
print(f"ln_C= {ln_C:.4f}")
print(f"C= {C:.4f}")