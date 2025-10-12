import math
import matplotlib.pyplot as plt
import numpy as np 

def fa(x):
    return ((x ** 2) +(2 * x)) / 3

def fb(x):
    return 2 * (x ** 2)

def p2(x, coeffs):
    return coeffs[0] * (x ** 2) + coeffs[1] * (x) + coeffs[2]

x = np.array([0, 1, 4])
y = np.array([0, 2, 8])

coeffs = np.polyfit(x, y, 2)
print(coeffs)

y_fit = np.polyval(coeffs, x)
print(y_fit)