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
    """
    Calculate the Real Error
    """
    return f(1) - 1 - est

def E(N, f):
    """
    Calculates the Estimated Error
    """
    h = 1 / N
    return (T(2 * N , f) - T(N , f)) / 3

def err_ratio(est, real):
    return real / est


h_values = []
error_values_t = []
error_values_b = []
err_ratios = []

for i in range(10):
    N = 2 ** i  
    h = 1 / N

    t_res = T(N, f)
    b_res = R(N, f)

    err_res_i_t = err(i, t_res)
    err_res_i_b = err(i, b_res)
    err_est_i_t = E(N, f)
    err_ratio_i = err_ratio(err_est_i_t, err_res_i_t)
    
    print(f"i = {i}, T({N}, f) = {t_res},\n Real Error = {err_res_i_t}, Est Error: {err_est_i_t}\n")

    h_values.append(h)
    error_values_t.append(abs(err_res_i_t))
    error_values_b.append(abs(err_res_i_b))
    err_ratios.append(abs(err_ratio_i))

plt.loglog(h_values, error_values_t, 
           color='red', 
           label='Trapezoidal Error vs. h')

plt.loglog(h_values, error_values_b, 
           color='blue', 
           label='Box Rule Error vs. h')

plt.semilogx(h_values, err_ratios, 
           color='black', 
           label='err ratio real / est vs. h')

plt.xlabel('log(h)')
plt.ylabel('log(abs(err))')
# plt.gca().invert_xaxis() # Since error decreases as h decreases, invert the x-axis for a positive slope
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.legend()
plt.tight_layout()

plt.show() 

# the result of the ration behavior was very much a consisten line 
# with a slope of 1 it only deviated slightly towards the highest h
# the slope was 4.
