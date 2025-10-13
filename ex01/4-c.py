import sympy as sp

def F(x):
    return (x ** 2) / ((x ** 2) + 1)

def PA(x):
    return F(0) + (((F(1) - F(0))) * x) + ((F(2) - 2*F(1) +F(0)) / 2) * (x) * (x - 1)

def PB(x):
    return PA(x) + ((F(3) - 3*F(2) +3*F(1) - F(0)) / 6) * (x) * (x- 1) * (x - 2)

x = sp.Symbol('x')
f = (x ** 2) / ((x ** 2) + 1)
Pa = F(0) + (((F(1) - F(0))) * x) + ((F(2) - 2*F(1) +F(0)) / 2) * (x) * (x - 1)
Pb = PA(x) + ((F(3) - 3*F(2) +3*F(1) - F(0)) / 6) * (x) * (x- 1) * (x - 2)

f_I = sp.integrate(f, (x, 0, 3))
Pa_I = sp.integrate(Pa, (x, 0, 3))
Pb_I = sp.integrate(Pb, (x, 0, 3))
print("Real I from 0 to 3:", f_I.evalf())
print(f"Pa I from 0 to 3: {Pa_I}, Err: {abs(f_I.evalf() - Pa_I)}")
print(f"Pb I from 0 to 3:{Pb_I}, Err: {abs(f_I.evalf() - Pb_I)}")