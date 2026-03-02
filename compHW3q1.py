# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 15:33:04 2026

@author: micha
"""

import numpy as np
import matplotlib.pyplot as plt

# dy/dx = y^2 + 1, with y(0)=0 so exact solution is y = tan(x)
def f(x, y):
    return y*y + 1

def euler(x0, y0, xf, N):
    h = (xf - x0) / N
    x = np.linspace(x0, xf, N+1)
    y = np.zeros(N+1)
    y[0] = y0
    for i in range(N):
        y[i+1] = y[i] + h * f(x[i], y[i])
    return x, y

def rk4(x0, y0, xf, N):
    h = (xf - x0) / N
    x = np.linspace(x0, xf, N+1)
    y = np.zeros(N+1)
    y[0] = y0
    for i in range(N):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + 0.5*h, y[i] + 0.5*h*k1)
        k3 = f(x[i] + 0.5*h, y[i] + 0.5*h*k2)
        k4 = f(x[i] + h,     y[i] + h*k3)
        y[i+1] = y[i] + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return x, y


# part a -  compare Euler vs RK4 vs exact using same N

x0, y0 = 0.0, 0.0
xf_a = 1.40           # below pi/2
N_a  = 50             # same steps for both

xE, yE = euler(x0, y0, xf_a, N_a)
xR, yR = rk4(x0, y0, xf_a, N_a)
y_exact = np.tan(xE)

plt.figure()
plt.plot(xE, y_exact, label="Exact tan(x)")
plt.plot(xE, yE, "o-", markersize=3, label="Euler")
plt.plot(xR, yR, "s-", markersize=3, label="RK4")
plt.xlabel("x")
plt.ylabel("y")
plt.title("(a) Euler vs RK4 vs exact")
plt.legend()
plt.tight_layout()
plt.show()

print("\n(a) Sample table:")
print("   x      Euler        RK4        Exact")
for i in range(0, N_a+1, 10):
    print(f"{xE[i]:6.3f}  {yE[i]:10.6f}  {yR[i]:10.6f}  {y_exact[i]:10.6f}")




# part b - Where does it break down?

xf_b = 1.56     # close to pi/2 ~ 1.5708
N_b  = 400

xEb, yEb = euler(x0, y0, xf_b, N_b)
xRb, yRb = rk4(x0, y0, xf_b, N_b)
y_ex_b = np.tan(xEb)

relE = np.abs(yEb - y_ex_b) / (np.abs(y_ex_b) + 1.0)
relR = np.abs(yRb - y_ex_b) / (np.abs(y_ex_b) + 1.0)

thresh = 0.05

idxE = np.where(relE > thresh)[0]
idxR = np.where(relR > thresh)[0]

print("\n(b) Breakdown (relative error > 5%):")
if len(idxE) > 0:
    j = idxE[0]
    print(f"Euler breaks near x = {xEb[j]:.6f}, y ~ {yEb[j]:.3e}, dy/dx ~ {yEb[j]**2 + 1:.3e}")
else:
    print("Euler did not exceed 5% error in this range.")

if len(idxR) > 0:
    j = idxR[0]
    print(f"RK4 breaks near x = {xRb[j]:.6f}, y ~ {yRb[j]:.3e}, dy/dx ~ {yRb[j]**2 + 1:.3e}")
else:
    print("RK4 did not exceed 5% error in this range.")

plt.figure()
plt.semilogy(xEb, relE, label="Euler rel error")
plt.semilogy(xRb, relR, label="RK4 rel error")
plt.xlabel("x")
plt.ylabel("relative error")
plt.title("(b) Error grows near pi/2 because y and dy/dx blow up")
plt.legend()
plt.tight_layout()
plt.show()



# part c-  checking if smaller step size increases accuracy (use exact at x=1.0)

x_check = 1.0
hs = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
errE_list = []
errR_list = []

for h in hs:
    N = int((x_check - x0) / h)
    x_end = x0 + N*h  # should be 1.0 for these h values
    _, y1 = euler(x0, y0, x_end, N)
    _, y2 = rk4(x0, y0, x_end, N)
    ytrue = np.tan(x_end)
    errE_list.append(abs(y1[-1] - ytrue))
    errR_list.append(abs(y2[-1] - ytrue))

print("\n(c) Error at x=1.0 for different step sizes:")
print("    h       Euler_err      RK4_err")
for i in range(len(hs)):
    print(f"{hs[i]:7.4f}   {errE_list[i]:11.3e}  {errR_list[i]:11.3e}")



# part d - convergence study pretending exact unknown

h_ref = 0.0005
N_ref = int((x_check - x0) / h_ref)

_, yE_ref = euler(x0, y0, x_check, N_ref)
_, yR_ref = rk4(x0, y0, x_check, N_ref)

yE_ref_end = yE_ref[-1]
yR_ref_end = yR_ref[-1]

fracE = []
fracR = []
for h in hs:
    N = int((x_check - x0) / h)
    _, yE_tmp = euler(x0, y0, x_check, N)
    _, yR_tmp = rk4(x0, y0, x_check, N)
    fracE.append(abs(yE_tmp[-1] - yE_ref_end) / abs(yE_ref_end))
    fracR.append(abs(yR_tmp[-1] - yR_ref_end) / abs(yR_ref_end))

plt.figure()
plt.loglog(hs, fracE, "o-", label="Euler")
plt.loglog(hs, fracR, "s-", label="RK4")
plt.xlabel("step size h")
plt.ylabel("fractional difference vs best run")
plt.title("(d) Convergence study (pretend exact unknown)")
plt.legend()
plt.tight_layout()
plt.show()

print("\n(d) On the log-log plot, Euler should look ~1st order, RK4 ~4th order.")

# part e - comparing convergence uncertainty estimate to true error

h1 = 0.0005
h2 = 0.00025
N1 = int((x_check - x0) / h1)
N2 = int((x_check - x0) / h2)

_, yE1 = euler(x0, y0, x_check, N1)
_, yE2 = euler(x0, y0, x_check, N2)
_, yR1 = rk4(x0, y0, x_check, N1)
_, yR2 = rk4(x0, y0, x_check, N2)

ytrue = np.tan(x_check)

proxyE = abs(yE2[-1] - yE1[-1])
proxyR = abs(yR2[-1] - yR1[-1])

trueE  = abs(yE2[-1] - ytrue)
trueR  = abs(yR2[-1] - ytrue)

print("\n(e) Uncertainty proxy (difference between two finest runs) vs true error:")
print(f"Euler: proxy = {proxyE:.3e}, true error = {trueE:.3e}")
print(f"RK4:   proxy = {proxyR:.3e}, true error = {trueR:.3e}")




















