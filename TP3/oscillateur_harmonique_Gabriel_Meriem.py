from matplotlib import pyplot as plt
import numpy as np


def xpp(x1):
    """"""
    return -omega ** 2 * x1


N = 1000
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

    k1 = step * xpp(x)
    r1 = step * xp
    k2 = step * xpp(x + 0.5 * r1)
    r2 = step * (xp + 0.5 * k1)
    k3 = step * xpp(x + 0.5 * r2)
    r3 = step * (xp + 0.5 * k2)
    k4 = step * xpp(x + k3)
    r4 = step * (xp + r3)

    x += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    xp += (r1 + 2 * r2 + 2 * r3 + r4) / 6

