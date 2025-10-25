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
I3_true = 0.5 * np.exp(-1) + np.exp(1) - 0.5 * np.exp(1/3)

def f4(x):
    return np.sin(np.pi * x)
I4_true = 0.0

def f5(x):
    return np.sin(4.0 * np.pi * x)
I5_true = 0.0

FUNCTIONS = [
    (f1, I1_true, '$f_1(x)=x^2$'),
    (f2, I2_true, '$f_2(x)=|x|$'),
    (f3, I3_true, '$f_3(x)$'),
    (f4, I4_true, '$f_4(x)=\\sin(\\pi x)$'),
    (f5, I5_true, '$f_5(x)=\\sin(4\\pi x)$')
]

def composite_trapezoidal_rule(f, a, b, N):
    """
    integral of f from a to b using the Composite Trapezoidal Rule with N subintervals.
    """
    h = (b - a) / N
    
    x_i = np.linspace(a, b, N + 1)
    f_i = f(x_i)
    
    # T_N(f) = h * [ (f_0 + f_N)/2 + sum_{i=1}^{N-1} f_i ]
    internal_sum = np.sum(f_i[1:-1])
    integral_approx = h * (0.5 * f_i[0] + internal_sum + 0.5 * f_i[-1])
    
    return integral_approx


# h = (b-a) * 2^(-i): i = 1, 2, ..., 20.
# (b-a) = 2, h = 2 * 2^(-i) = 2^(1-i)
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

plt.title('error vs. $h$ (Composite Trapezoidal Rule)')
plt.xlabel('Step Size $h$ (Log Scale)')
plt.ylabel('error $|e|$ (Log Scale)')
plt.grid(True, which="both", ls="--")
plt.legend(loc='lower right')
plt.gca().invert_xaxis()
plt.show()