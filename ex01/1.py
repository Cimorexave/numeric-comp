import math

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

def get_n():
    """
    Get N (integer) from user
    """
    while True:
        try:
            N_input = int(input("N: "))
            if N_input > 0:
                return N_input
            print("N must be a positive integer.")
        except ValueError:
            print("Invalid input")

def get_f():
    """
    Get f (function expression) from user
    """
    while True:
        expr_str = input("Enter the function epxression f"
        "(x) in terms of 'x' (e.g., math.sin(x), x**2): ")
        try:
            safe_globals = {'__builtins__': None, 'math': math}
            f = eval(f'lambda x: {expr_str.strip()}', safe_globals)
            
            test_value = f(1)
            return f
            
        except Exception as e:
            print(f"Invalid function definition or syntax: {e}")


N = get_n()
f = get_f()
t_result = T(N, f)
b_result = R(N, f)

# Example: If user inputs x**2 and N=1000, the result should be close to 1/3 (0.333...)

print(f"\nT(h) for f(x) with N={N} intervals:")
print(f"T({N}, f) = {t_result}\n")
print("------")
print(f"\nR(h) for f(x) with N={N} intervals:")
print(f"R({N}, f) = {b_result}\n")