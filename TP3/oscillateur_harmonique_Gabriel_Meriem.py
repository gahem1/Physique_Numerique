from matplotlib import pyplot as plt
import numpy as np


def xpp(x):
    return -omega ** 2 * x


N = 1000
ti = 0
tf = 50
tvals = np.linspace(ti, tf, num=N)
xvals, xpvals = np.empty(N), np.empty(N)
step = (tf - ti) / N

omega = 1
x = 1
xp = 0

for i in range(N):
    xvals[i], xpvals[i] = x, xp

