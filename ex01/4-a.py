import numpy as np 

x_values = input("Enter x values separated by spaces: ").split()
x_values = [float(x) for x in x_values]

y_values = input("Enter f values separated by spaces: ").split()
y_values = [float(y) for y in y_values]

n = int(input("N: "))

coeffs =np.polyfit(x_values, y_values, n)
print(coeffs)

y_fit = np.polyval(coeffs, 0)
print(f"P(0): {y_fit}")