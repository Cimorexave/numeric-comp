import time
import numpy as np

def S(N):
    out = 0
    for i in range(1, N+1):
        out += (1 / N)
    return out

# Ax = s
A = np.array([
    np.array([1, (1/10), (1/100)]),
    np.array([1, (1/100), (1/10000)]),
    np.array([1, (1/1000), (1/1000000)]),
]
)
s = np.array([
    S(10), S(100), S(1000)
])

a = np.linalg.solve(A, s)
print("a0, a1, a2")
print(a)

def P(N):
    return np.log(N) + a[0] + a[1]*(N ** (-1)) + a[2]*(N ** (-2))

def err(N):
    return abs(S(N) - P(N))


n = 10**1
print(f'error at N=10**1 is: {err(n)}')
n = 10**3
print(f'error at N=10**3 is: {err(n)}')
n = 10**6
print(f'error at N=10**6 is: {err(n)}')
n = 10**8
print(f'error at N=10**8 is: {err(n)}')
tp8_start = time.time()
p_8 = P(n)
tp8_end = time.time()
print(f'processing time approximation P(10^8)= {tp8_end - tp8_start}')
tr8_start = time.time()
r_8 = S(n)
tr8_end = time.time()
print(f'processing time real S(10^8)= {tr8_end - tr8_start}')

n = 10**9
tp9_start = time.time()
p_9 = P(n)
tp9_end = time.time()
print(f'processing time approximation P(10^9)= {tp9_end - tp9_start}')
tr9_start = time.time()
r_9 = S(n)
tr9_end = time.time()
print(f'processing time real S(10^9)= {tr9_end - tr9_start}')


# a0, a1, a2 = [ 1.00000000e+00  9.59451997e-16 -8.72229088e-14]
# error at N=1000000 is: 13.815510557956355
# error at N=100000000 is: 18.4206807416625
# processing time approximation P(10^8)= 2.9087066650390625e-05
# processing time real S(10^8)= 5.296080589294434
# processing time approximation P(10^9)= 5.1021575927734375e-05
# processing time real S(10^9)= 51.8939847946167

