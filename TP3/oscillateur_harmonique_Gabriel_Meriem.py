from matplotlib import pyplot as plt
import numpy as np


def rk4(n, x1, x2, omega, func):
    xlist = np.empty(n)
    xplist = np.empty(n)
    t = 0
    for i in range(n):
        xlist[i] = x1
        xplist[i] = x2
        t += step

        r1 = step * x2
        k1 = step * func(x1, omega)
        r2 = step * (x2 + 0.5 * k1)
        k2 = step * func(x1 + 0.5 * r1, omega)
        r3 = step * (x2 + 0.5 * k2)
        k3 = step * func(x1 + 0.5 * r2, omega)
        r4 = step * (x2 + k3)
        k4 = step * func(x1 + r3, omega)

        x1 += (r1 + 2 * r2 + 2 * r3 + r4) / 6
        x2 += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return xlist, xplist


def xpp(x1, omega):
    return -x1 * (omega ** 2)


N = 100000
tf = 50
tvals = np.linspace(0, tf, num=N)
step = tf / N

if __name__ == "__main__":
    xi = 1
    xpi = 0
    angular_frequency = 1

    plt.figure()
    plt.plot(tvals, rk4(N, xi, xpi, angular_frequency, xpp)[0], linewidth=2)
    plt.ylabel("x(t)", fontsize=18)
    plt.xlabel("t", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()
