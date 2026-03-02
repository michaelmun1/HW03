import numpy as np
import matplotlib.pyplot as plt

# constants in SI units
kB = 1.380649e-23
mH = 1.6735575e-27
eV = 1.602176634e-19
T  = 10000.0

# n=1 -> n=2 energy for H
dE = 10.2 * eV

# Maxwell-Boltzmann speed PDF f(v)
def f(v):
    A = 4*np.pi * (mH/(2*np.pi*kB*T))**1.5
    return A * v*v * np.exp(-mH*v*v/(2*kB*T))

# simple trapezoid integrator (your own)
def trapz(x, y):
    s = 0.0
    for i in range(len(x)-1):
        s += 0.5*(y[i] + y[i+1])*(x[i+1] - x[i])
    return s

# typical speeds (just for sanity check)
v_mp  = np.sqrt(2*kB*T/mH)   # most probable
v_rms = np.sqrt(3*kB*T/mH)

# part a
v = np.linspace(0, 5*v_rms, 2000)
plt.plot(v, f(v))
plt.axvline(v_mp,  linestyle="--", label="v_mp")
plt.axvline(v_rms, linestyle="--", label="v_rms")
plt.xlabel("v (m/s)")
plt.ylabel("f(v)")
plt.title("(a) Maxwell-Boltzmann speeds, H at 10000 K")
plt.legend()
plt.tight_layout()
plt.show()




# part b 
vmin = np.sqrt(2*dE/mH)
print("vmin for 10.2 eV =", vmin, "m/s")

vmax = 20*v_rms      # "infinity"
N = 200000
v2 = np.linspace(vmin, vmax, N+1)
frac = trapz(v2, f(v2))
print("(b) fraction ~", frac)




# part c
# 1) step-size convergence (same vmax, change N)
N1, N2 = 50000, 100000
vA = np.linspace(vmin, vmax, N1+1)
vB = np.linspace(vmin, vmax, N2+1)
frac1 = trapz(vA, f(vA))
frac2 = trapz(vB, f(vB))
err_step = abs(frac2 - frac1)


# 2) cutoff convergence (same N, change vmax)
vmax1, vmax2 = 16*v_rms, 20*v_rms
vC = np.linspace(vmin, vmax1, N1+1)
vD = np.linspace(vmin, vmax2, N1+1)
frac3 = trapz(vC, f(vC))
frac4 = trapz(vD, f(vD))
err_vmax = abs(frac4 - frac3)


err = err_step + err_vmax   # simple conservative combo
print("(c) rough error bar ~", err)
print("report: frac =", frac, "+/-", err)