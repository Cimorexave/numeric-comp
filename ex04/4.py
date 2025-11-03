import numpy as np
import matplotlib.pyplot as plt

def f(x):
    if x < 1/3:
        return 0.5 * np.exp(x)
    else:
        return np.exp(x)


def simpson_rule(a, b, fa, fm, fb):
    """
    Simpson's Rule in [a, b] using pre-calculated function values.
    a: left endpoint, b: right endpoint
    fa: f(a), fm: f((a+b)/2), fb: f(b)
    """
    h = b - a
    return (h / 6.0) * (fa + 4.0 * fm + fb)

# ---  Adaptive Quadrature Algorithm ---
def adapt(f, a, b, tau, h_min):
    """
    Adaptive Quadrature using Simpson's Rule.
    f: function, [a, b]: interval, tau: absolute accuracy, h_min: minimum interval length
    Returns the approximate integral value.
    """
    # 1. Base function evaluations
    m = (a + b) / 2.0
    
    # Check for minimal interval length
    # h = b - a
    if (b - a) < h_min:
        print(f"Warning: Reached h_min={h_min} on [{a:.4f}, {b:.4f}], returning Simpson's value.")
        fa = f(a)
        fm = f(m)
        fb = f(b)
        return simpson_rule(a, b, fa, fm, fb)

    # 2. Original and Refined Approximations
    # Function evaluations at 5 points (a, m1, m, m2, b)
    m1 = (a + m) / 2.0
    m2 = (m + b) / 2.0
    
    # Reusing f(a), f(m), f(b) if available, but need to be safe
    fa = f(a)
    fm = f(m)
    fb = f(b)
    fm1 = f(m1)
    fm2 = f(m2)

    S_original = simpson_rule(a, b, fa, fm, fb)
    
    # S_refined = S_[a, m](f) + S_[m, b](f)
    S_left = simpson_rule(a, m, fa, fm1, fm)
    S_right = simpson_rule(m, b, fm, fm2, fb)
    S_refined = S_left + S_right
    
    # 3. Error Estimation
    # Local error E approx (1/15) * |S_refined - S_original|
    error_estimate = (1.0 / 15.0) * np.abs(S_refined - S_original)
    
    # 4. Acceptance Check
    if error_estimate < tau:
        # Result accepted, return the more accurate value
        return S_refined
    else:
        # Result rejected, recurse on subintervals
        I_left = adapt(f, a, m, tau / 2.0, h_min) # Note: Split tolerance
        I_right = adapt(f, m, b, tau / 2.0, h_min)
        return I_left + I_right

# --- Exact Integral (for error plotting) ---
def exact_integral():
    I1 = 0.5 * (np.exp(1/3) - np.exp(0))
    I2 = np.exp(1) - np.exp(1/3)
    return I1 + I2

# --- Execution and Plotting ---
def run_and_plot():

    A, B = 0.0, 1.0
    exact = exact_integral()
    
    # Define tau and h_min sequence
    j_values = np.arange(0, 11)
    tau_values = 2.00**(-j_values)
    h_min_values = tau_values # Given constraint: h_min = tau

    errors = []
    
    print("j | tau=h_min | Approx Integral | Absolute Error")
    print("-" * 55)
    
    for j, tau in enumerate(tau_values):
        h_min = h_min_values[j]
        
        # Integrate
        I_approx = adapt(f, A, B, tau, h_min)
        
        # Calculate error
        error = np.abs(I_approx - exact)
        errors.append(error)
        
        print(f"{j} | {tau} | {I_approx} | {error}")

    try:
        plt.figure(figsize=(8, 6))
        plt.loglog(tau_values, errors, 'o-', label='absolute error')
        
        # Plot a line with slope 1 for comparison (Error ~ tau)
        plt.loglog(tau_values, tau_values, '--', color='gray', label=r'Reference line: Error $\propto \tau$')
        
        plt.xlabel(r'tolerance ($\tau$)')
        plt.ylabel('absolute error')
        plt.title('Error Convergence of Adaptive Simpson Quadrature')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.show()
    except ImportError:
        print("\nerror plotting")

run_and_plot()