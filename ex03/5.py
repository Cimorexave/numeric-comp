import numpy as np
import sympy as sp

a = -2.0
b = 4.0

def f(x):
    return 1 - x - 4 * x**3 + 2 * x**5

def calculate_exact_integral_f():
    x = sp.symbols('x')
    f_x = 1 - x - 4 * x**3 + 2 * x**5

    F_x = sp.integrate(f_x, x)
    result = F_x.subs(x, b) - F_x.subs(x, a)
    exact_value = result.evalf() 
    
    expression_result = F_x
    
    return exact_value


def relative_error(I, Q):
    return np.abs(I - Q) / np.abs(I)


I_exact = calculate_exact_integral_f()
# I_exact = 1104.0


# Trapezoidal Rule N=1
def trapezoidal_rule(f, a, b):
    h = b - a
    Q = h / 2 * (f(a) + f(b))
    return Q

# Midpoint Rule N=1
def midpoint_rule(f, a, b):
    h = b - a
    x_mid = (a + b) / 2
    Q = h * f(x_mid)
    return Q

# Simpson's Rule N=2
def simpsons_rule(f, a, b):
    h = (b - a) / 2
    x_mid = (a + b) / 2
    Q = h / 3 * (f(a) + 4 * f(x_mid) + f(b))
    return Q

# Milne Rule N=4 
def milne_rule(f, a, b):
    N = 4
    h = (b - a) / N
    x = a + np.arange(N + 1) * h
    
    # Coeffs: 7, 32, 12, 32, 7
    Q = (2 * h) / 45 * (7 * f(x[0]) + 32 * f(x[1]) + 12 * f(x[2]) + 32 * f(x[3]) + 7 * f(x[4]))
    
    return Q

# Composite Trapezoidal N=4 
def composite_trapezoidal_rule(f, a, b, N):
    h = (b - a) / N
    x = a + np.arange(N + 1) * h
    f_x = f(x)
    
    internal_sum = np.sum(f_x[1:-1])
    
    Q = h * (0.5 * f_x[0] + internal_sum + 0.5 * f_x[-1])
    return Q

# Composite Simpson's Rule N=3 
def simpsons_3_8_rule(f, a, b):
    N = 3
    h = (b - a) / N
    x = a + np.arange(N + 1) * h
    
    # Coeffs: 1, 3, 3, 1
    Q = (3 * h) / 8 * (f(x[0]) + 3 * f(x[1]) + 3 * f(x[2]) + f(x[3]))
    return Q



results = {}

results['Trapezoidal Rule (N=1)'] = trapezoidal_rule(f, a, b)
results['Midpoint Rule (N=1)'] = midpoint_rule(f, a, b)
results["Simpson's Rule (N=2)"] = simpsons_rule(f, a, b)
results['Milne Rule (N=4 points)'] = milne_rule(f, a, b)
results["Simpson's 3/8 Rule (N=3)"] = simpsons_3_8_rule(f, a, b)
results['Composite Trapezoidal (N=4)'] = composite_trapezoidal_rule(f, a, b, N=4)


print(f"Exact Value I={I_exact:.4f}")
print("\n" + "="*70)
print(f"{'Method'} | {'Approximation (Q)'} | {'Relative Error |I-Q|/|I|'}")
print("="*70)

for method, Q in results.items():
    error = relative_error(I_exact, Q)
    print(f"{method} | {Q} | {error}")

print("="*70)
print("Observation: Simpson's Rule (up to degree 3) and Milne's Rule (up to degree 5)" \
" are expected to be exact for this polynomial of degree 5.")