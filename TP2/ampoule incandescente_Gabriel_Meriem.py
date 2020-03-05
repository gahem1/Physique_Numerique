from gaussxw_Gabriel_Meriem import gaussxw, gaussxwab
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as scte


def f(x):
    return x ** 3 / (np.exp(x) - 1)


def efficacite(T):
    a = scte.h * scte.c / (750 * 10 ** -9 * scte.k * T)
    b = scte.h * scte.c / (390 * 10 ** -9 * scte.k * T)
    N = 100
    # Calculate the sample points and weights, then map them
    # to the required integration domain
    x, w = gaussxw(N)
    xp = 0.5 * (b - a) * x + 0.5 * (b + a)
    wp = 0.5 * (b - a) * w

    # Perform the integration
    s = 0.0
    for k in range(N):
        s += wp[k] * f(xp[k])
    return 15 * s / np.pi ** 4


def efficaciter(T):
    a = scte.h * scte.c / (750 * 10 ** -9 * scte.k * T)
    b = scte.h * scte.c / (390 * 10 ** -9 * scte.k * T)
    N = 100
    # Calculate the sample points and weights, then map them
    # to the required integration domain
    x, w = gaussxw(N)
    xp = 0.5 * (b - a) * x + 0.5 * (b + a)
    wp = 0.5 * (b - a) * w

    # Perform the integration
    s = 0.0
    for k in range(N):
        s += wp[k] * f(xp[k])
    return np.pi ** 4 / (15 * s)


T = np.linspace(300, 10000, num=100)
efficacite = np.vectorize(efficacite)
plt.plot(T, efficacite(T))
plt.xlabel('Température [K]')
plt.ylabel('Efficacité [-]')
plt.show()

# golden ratio
gold = 1 + np.sqrt(5) / 2


# initial values
t1 = 5000
t4 = 8000
t2 = t4 - (t4 - t1) / gold
t3 = t1 + (t4 - t1) / gold

while (t4 - t1) > 1:
    if efficacite(t2) < efficacite(t3):
        t4, t3 = t3, t2
        t2 = t4 - (t4 - t1) / gold
    else:
        t2, t1 = t3, t2
        t3 = t1 + (t4 - t1) / gold

print('Le maximum se trouve à {} K'.format((t4 + t1) / 2))

import scipy.optimize as so

print(so.golden(efficaciter, brack=(6000, 7000), tol=1))

