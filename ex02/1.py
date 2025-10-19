import math
import numpy as np

def f(x):
    return math.exp(x)

x_vals = np.array([0, 1, 2])
y_vals = np.array([1, f(1), f(2)])

n = len(x_vals) - 1
coeffs = np.polyfit(x_vals, y_vals, n)
p_val = np.polyval(coeffs, 1.2)

print(f"polyval at 1.2: {p_val}")

real_err = abs(f(1.2) - p_val)
print(f"real error at 1.2: {real_err}")
