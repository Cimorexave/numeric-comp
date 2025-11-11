import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x = sp.Symbol('x')
P_x = (x**2 - 1)**4

# l=1
L_4_1_sym = sp.diff(P_x, x, 1)
# l=2
L_4_2_sym = sp.diff(P_x, x, 2)
# l=3
L_4_3_sym = sp.diff(P_x, x, 3)

L_4_1_func = sp.lambdify(x, L_4_1_sym, 'numpy')
L_4_2_func = sp.lambdify(x, L_4_2_sym, 'numpy')
L_4_3_func = sp.lambdify(x, L_4_3_sym, 'numpy')

x_vals = np.linspace(-1, 1, 500)
y_1 = L_4_1_func(x_vals)
y_2 = L_4_2_func(x_vals)
y_3 = L_4_3_func(x_vals)

plt.figure(figsize=(10, 6))

plt.plot(x_vals, y_1, label='$L_4^1(x) = P^{(1)}(x), \\ell=1$', color='blue')
plt.plot(x_vals, y_2, label='$L_4^2(x) = P^{(2)}(x), \\ell=2$', color='orange')
plt.plot(x_vals, y_3, label='$L_4^3(x) = P^{(3)}(x), \\ell=3$', color='green')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.grid(True)
plt.xlim([-1.05, 1.05])

plt.show()