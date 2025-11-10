import numpy as np
import matplotlib.pyplot as plt

def f(x):
    if x < 1/3:
        return 0.5 * np.exp(x)
    else:
        return np.exp(x)


def simpson_rule(a, b, fa, fm, fb):
    h = b - a
    return (h / 6.0) * (fa + 4.0 * fm + fb)

def adapt(f, a, b, tau, h_min):
    m = (a + b) / 2.0
    
    if (b - a) < h_min:
        print(f"reached h_min={h_min} on [{a:.4f}, {b:.4f}], returning simpson's value.")
        fa = f(a)
        fm = f(m)
        fb = f(b)
        return simpson_rule(a, b, fa, fm, fb)

    m1 = (a + m) / 2.0
    m2 = (m + b) / 2.0
    
    fa = f(a)
    fm = f(m)
    fb = f(b)
    fm1 = f(m1)
    fm2 = f(m2)

    S_original = simpson_rule(a, b, fa, fm, fb)
    
    S_left = simpson_rule(a, m, fa, fm1, fm)
    S_right = simpson_rule(m, b, fm, fm2, fb)
    S_refined = S_left + S_right
    
    # local error, E approx = (1/15) * |S_refined - S_original|
    error_estimate = (1.0 / 15.0) * np.abs(S_refined - S_original)
    
    if error_estimate < tau:
        return S_refined
    else:
        I_left = adapt(f, a, m, tau / 2.0, h_min)
        I_right = adapt(f, m, b, tau / 2.0, h_min)
        return I_left + I_right

def exact_integral():
    I1 = 0.5 * (np.exp(1/3) - np.exp(0))
    I2 = np.exp(1) - np.exp(1/3)
    return I1 + I2

# execution:

A, B = 0.0, 1.0
exact = exact_integral()
    
j_values = np.arange(0, 11)
tau_values = 2.00**(-j_values)
h_min_values = tau_values
errors = []
    
print("j | tau=h_min | Approx Integral | Absolute Error")
print("-" * 55)
    
for j, tau in enumerate(tau_values):
    h_min = h_min_values[j]
        
    I_approx = adapt(f, A, B, tau, h_min)
        
    error = np.abs(I_approx - exact)
    errors.append(error)
        
    print(f"{j} | {tau} | {I_approx} | {error}")

try:
    plt.figure(figsize=(8, 6))
    plt.loglog(tau_values, errors, 'o-', label='absolute error')
        
    # line with scope 1 for comparison
    plt.loglog(tau_values, tau_values, '--', color='gray', label=r'reference slope 1: error $\propto \tau$')
        
    plt.xlabel(r'tolerance ($\tau$)')
    plt.ylabel('absolute error')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()
except ImportError:
    print("\nerror plotting.")