from matplotlib import pyplot as plt
import numpy as np


def xpp(x1):
    return -x1 * (omega ** 2)


N = 100000
t = 0
tf = 50
tvals = np.linspace(t, tf, num=N)
xvals = np.empty(N)
step = (tf - t) / N

omega = 1
x = 1
xp = 0

for i in range(N):
    xvals[i] = x
    t += step

    r1 = step * xp
    r2 = step * (xp + 0.5 * r1)
    r3 = step * (xp + 0.5 * r2)
    r4 = step * (xp + r3)
    x += (r1 + 2 * r2 + 2 * r3 + r4) / 6

    k1 = step * xpp(x)
    k2 = step * xpp(x + 0.5 * k1)
    k3 = step * xpp(x + 0.5 * k2)
    k4 = step * xpp(x + k3)
    xp += (k1 + 2 * k2 + 2 * k3 + k4) / 6

plt.figure()
plt.plot(tvals, xvals, linewidth=2)
plt.ylabel("x(t)", fontsize=18)
plt.xlabel("t", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()
