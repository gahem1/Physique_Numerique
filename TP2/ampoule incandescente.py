from gaussxw import gaussxw, gaussxwab
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


T = np.linspace(300, 10000, num=100)
efficacite = np.vectorize(efficacite)
plt.plot(T, efficacite(T))
plt.xlabel('Température (K)')
plt.ylabel('Efficacité')
plt.show()
