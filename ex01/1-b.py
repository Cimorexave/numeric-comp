import math
import matplotlib.pyplot as plt
import numpy as np 

def T(N, f):
    """
    Calculates the integral of function f from 0 to 1
    using the Trapezoidal Rule with N intervals.
    """
    h = 1 / N
    result = 0
    for i in range(N):
        x_i = i * h
        x_i_plus_1 = (i + 1) * h
        result += (h / 2) * (f(x_i) + f(x_i_plus_1))
    return result

def R(N, f):
    """
    Calculates the integral of function f from 0 to 1
    using the Box Rule with N intervals.
    """
    h = 1 / N
    result = 0
    for i in range(N):
        x_i = i * h
        result += h * f(x_i)
    return result

def f(x):   
    return math.exp(x)

def err(i, est):
    return f(1) - 1 - est


h_values = []
error_values_t = []
error_values_b = []

for i in range(10):
    N = 2 ** i  
    h = 1 / N
    t_res = T(N, f)
    b_res = R(N, f)
    err_res_i_t = err(i, t_res)
    err_res_i_b = err(i, b_res)
    print(f"i = {i}, T({N}, f) = {t_res}, Error = {err_res_i_t}\n")
    h_values.append(h)
    error_values_t.append(abs(err_res_i_t))
    error_values_b.append(abs(err_res_i_b))

plt.loglog(h_values, error_values_t, 
           color='red', 
           label='Trapezoidal Error vs. h')

plt.loglog(h_values, error_values_b, 
           color='blue', 
           label='Box Rule Error vs. h')

plt.xlabel('log(h)')
plt.ylabel('log(abs(err))')
# plt.gca().invert_xaxis() # Since error decreases as h decreases, invert the x-axis for a positive slope
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.legend()
plt.tight_layout()

plt.show() 

# does it behave like Error = C(h^2)? Yes
# log(E) = Log(C) + 2*log(h)
# and you can see the plot has a slope of 2

# does it work the same for the box rule?
# kind of. the errors are higher in the same step and also the slope is lower than 2. but still
# bigger than 1 seems like it's in the middle. 1.5