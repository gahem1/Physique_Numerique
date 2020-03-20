from matplotlib import pyplot as plt
import numpy as np


def rk4(n, x1, x2, t):
    xlist = np.empty(n)
    xplist = np.empty(n)
    for i in range(n):
        xlist[i] = x1
        xplist[i] = x2
        t += step

        r1 = step * x2
        r2 = step * (x2 + 0.5 * r1)
        r3 = step * (x2 + 0.5 * r2)
        r4 = step * (x2 + r3)

        k1 = step * xpp(x1)
        k2 = step * xpp(x1 + 0.5 * k1)
        k3 = step * xpp(x1 + 0.5 * k2)
        k4 = step * xpp(x1 + k3)

        x1 += (r1 + 2 * r2 + 2 * r3 + r4) / 6
        x2 += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return xlist, xplist


def xpp(x1):
    return -x1 * (omega ** 2)


N = 100000
ti = 0
tf = 50
tvals = np.linspace(ti, tf, num=N)
step = (tf - ti) / N

omega = 1
xi = 1
xpi = 0

if __name__ == "__main__":
    xvals, xpvals = rk4(N, xi, xpi, ti)

    plt.figure()
    plt.plot(tvals, xvals, linewidth=2)
    plt.ylabel("x(t)", fontsize=18)
    plt.xlabel("t", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()
