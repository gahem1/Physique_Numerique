import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fixed_point
import scipy.constants as scte


def f(x):
    return 5 - np.exp(-x)

x = 0
for i in range(20):
    x = f(x)
print('La solution avec 20 itérations est de {}'.format(x))


x = 3
x2 = x+1
erreur = 10 ** -6
while abs(x2 - x) > erreur:
    x2 = x
    x = f(x)
print("La solution itérative est de {}".format(x))

x = fixed_point(f, 0, xtol=erreur)
print("La solution à l'aide de scipy.optimize.fixed.point est de {}".format(x))

b = scte.h * scte.c / (x * scte.k)
print("La constante de déplacement de Wien calculé est {}".format(b))

t = np.arange(-3, 8, 0.2)
plt.plot(t, t, 'r', label = 'y = x')
plt.plot(t, f(t), 'b', label = "y = 5 - exp(-x)")
plt.legend()
plt.show()
