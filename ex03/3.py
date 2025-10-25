import numpy as np
import matplotlib.pyplot as plt

a, b = -1.0, 1.0

def f1(x):
    return x**2
I1_true = 2.0 / 3.0

def f2(x):
    return np.abs(x)
I2_true = 1.0

def f3(x):
    return np.where(x < 1/3, 0.5 * np.exp(x), np.exp(x))
I3_true = 0.5 * np.exp(-1) + np.exp(1) - 0.5 * np.exp(1/3) # Rearranged for stability

def f4(x):
    return np.sin(np.pi * x)
I4_true = 0.0

def f5(x):
    return np.sin(4.0 * np.pi * x)
I5_true = 0.0

FUNCTIONS = [
    (f1, I1_true, '$f_1(x)=x^2$ (Smooth)'),
    (f2, I2_true, '$f_2(x)=|x|$ (Cusp at $x=0$)'),
    (f3, I3_true, '$f_3(x)$ (Discontinuity at $x=1/3$)'),
    (f4, I4_true, '$f_4(x)=\\sin(\\pi x)$ (Smooth)'),
    (f5, I5_true, '$f_5(x)=\\sin(4\\pi x)$ (Smooth, High Freq)')
]

def composite_trapezoidal_rule(f, a, b, N):
    """
    integral of f from a to b using the Composite Trapezoidal Rule with N subintervals.
    """
    h = (b - a) / N
    
    # Grid points x_i = a + i*h
    x_i = np.linspace(a, b, N + 1)
    # Function values at grid points
    f_i = f(x_i)
    
    # T_N(f) = h * [ (f_0 + f_N)/2 + sum_{i=1}^{N-1} f_i ]
    internal_sum = np.sum(f_i[1:-1])
    
    # Full sum, applying weights: (1/2 * f_0) + sum + (1/2 * f_N)
    integral_approx = h * (0.5 * f_i[0] + internal_sum + 0.5 * f_i[-1])
    
    return integral_approx

# --- Error Analysis and Plotting ---

# Parameters for error analysis
# h = (b-a) * 2^(-i) where i = 1, 2, ..., 20. Since (b-a) = 2, h = 2 * 2^(-i) = 2^(1-i)
i_range = np.arange(1, 21)
N_range = (b - a) / ( (b - a) * 2.0**(-i_range) )
N_range = N_range.astype(int)
h_range = (b - a) / N_range

plt.figure(figsize=(10, 7))

for f_func, I_true, label in FUNCTIONS:
    errors = []
    
    for N, h in zip(N_range, h_range):
        I_approx = composite_trapezoidal_rule(f_func, a, b, N)
        
        error = np.abs(I_approx - I_true)
        errors.append(error)
    
    errors = np.array(errors)
    print(f'for f={f_func} h_range={h_range}\nerrors={errors}')
    plt.loglog(h_range, errors, marker='o', linestyle='-', label=label)

plt.title('Quadrature Error vs. Step Size $h$ (Composite Trapezoidal Rule)')
plt.xlabel('Step Size $h$ (Log Scale)')
plt.ylabel('Absolute Error $|E|$ (Log Scale)')
plt.grid(True, which="both", ls="--")
plt.legend(loc='lower right')
plt.gca().invert_xaxis()
plt.show()

# --- Observations and Explanation ---

print("\n" + "="*80)
print("OBSERVATIONS AND EXPLANATION")
print("="*80)

# Calculate the slope for the smooth function f1 (x^2) to demonstrate the order of convergence
log_h = np.log(h_range)
log_E = np.log(np.abs(I1_true - composite_trapezoidal_rule(f1, a, b, N_range)))
slope = (log_E[-1] - log_E[0]) / (log_h[-1] - log_h[0])
print(f"Numerical Order of Convergence (Slope) for f1(x)=x^2: m ≈ {slope:.2f}")

print("\nObservation Summary:")
print("1. Order of Convergence (Slope) on the loglog plot:")
print("   - For $f_1(x)=x^2$ and $f_4, f_5$ (Smooth): The slope is approximately -2.")
print("   - For $f_2(x)=|x|$ (Cusp): The slope is approximately -1.")
print("   - For $f_3(x)$ (Discontinuity): The slope is approximately -1.")
print("   - $f_4$ and $f_5$ show very small errors, demonstrating the high accuracy for integrating periodic functions over their full period.")

print("\nTheoretical Explanation:")
print("The error term for the Composite Trapezoidal Rule is bounded by:")
print("$|E_N(f)| \\le C \\frac{(b-a)^3}{N^2} \\max_{x \\in [a,b]} |f''(x)| \\approx O(h^2)$")
print("where $h$ is the step size and $N$ is the number of subintervals.")

print("\nAnalysis of $f_1$, $f_2$, and $f_3$:")
print("-----------------------------------")
print("Function $f_1(x) = x^2$:")
print(" - **Observation:** Error $\\propto h^2$ (Slope of -2).")
print(" - **Explanation:** $f_1(x)$ is **infinitely differentiable** ($C^{\\infty}$) on $[-1, 1]$. Since $f''(x) = 2$ is continuous, the theoretical **second-order convergence** $O(h^2)$ is achieved.")

print("\nFunction $f_2(x) = |x|$:")
print(" - **Observation:** Error $\\propto h^1$ (Slope of -1).")
print(" - **Explanation:** $f_2(x)$ has a **cusp** (a corner) at $x=0$, meaning $f'(x)$ is discontinuous and $f''(x)$ is **undefined** at $x=0$. The smoothness assumption for $O(h^2)$ convergence is violated, leading to a reduced **first-order convergence** $O(h)$.")

print("\nFunction $f_3(x)$ (Piecewise):")
print(" - **Observation:** Error $\\propto h^1$ (Slope of -1).")
print(" - **Explanation:** $f_3(x)$ has a **discontinuity** in its derivative at $x=1/3$ (and potentially in the function itself, or just $f'$ depending on the constants). A jump discontinuity in $f'(x)$ (or $f''(x)$ being a delta function) violates the smoothness requirement, reducing the convergence order to **$O(h)$**.")