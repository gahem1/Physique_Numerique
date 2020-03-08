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
    # Calculate the sample points and weights
    x, w = gaussxw(N)
    xp = (b - a) / 2 * x + (b + a) / 2
    wp = (b - a) * w / 2
    # Perform the integration
    s = 0.0
    for i in range(N):
        s += wp[i] * f(xp[i])
    return 15 * s / np.pi ** 4


T = np.linspace(300, 10000, num=100)
efficacite = np.vectorize(efficacite)
plt.plot(T, efficacite(T))
plt.xlabel('Température [K]')
plt.ylabel('Efficacité [-]')
plt.show()

# golden ratio
gold = 1 + np.sqrt(5) / 2


# initial values
t1 = 6850
t4 = 7150
t2 = t4 - (t4 - t1) / gold
t3 = t1 + (t4 - t1) / gold

while (t4 - t1) > 1:
    if efficacite(t2) > efficacite(t3):
        t1 = t2
        t2 = t4 - (t4 - t1) / gold
    else:
        t4 = t3
        t3 = t1 + (t4 - t1) / gold

print('Le maximum se trouve à {} K'.format((t2 + t3) / 2))
