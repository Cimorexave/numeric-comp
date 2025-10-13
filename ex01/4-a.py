import numpy as np 

x_values = input("Enter x values separated by spaces: ").split()
x_values = [float(x) for x in x_values]

y_values = input("Enter f values separated by spaces: ").split()
y_values = [float(y) for y in y_values]

if len(x_values) != len(y_values) or len(x_values) == 0:
    raise ValueError("Input vectors must have the same length.")

N = int(input(f"Input length (Should be less than {len(x_values)}): "))
# polynomial_degree_n = len(x_values) - 1
polynomial_degree_n = N - 1

coeffs = np.polyfit(x_values, y_values, polynomial_degree_n)
print(f"Coefficients (degree {polynomial_degree_n}): {coeffs}")

p_at_0 = np.polyval(coeffs, 0)
print(f"P(0): {p_at_0}")

