from hamiltonian import Operator
from random import random, randint
import numpy as np

print("Gram-Schmidt")
a = 6
test = Operator(a, random(), random())
va, ve, di, gg, te = test.eigenalgo(10 ** -14, 10000, "Gram-Schmidt")
for i in range(a):
    print(np.sum(np.tile(ve[i, :], (a, 1)).T * ve, axis=0))
    print(np.abs(ve - test.matrix @ ve))

print("Givens")
a = 6
test = Operator(a, random(), random())
va, ve, di, gg, te = test.eigenalgo(10 ** -14, 10000, "Givens")
for i in range(a):
    print(np.sum(np.tile(ve[i, :], (a, 1)).T * ve, axis=0))
    print(np.abs(ve - test.matrix @ ve))

print("Rayleigh")
a = 6
test = Operator(a, random(), random())
va, ve, di, gg, te = test.eigenalgo(10 ** -14, 10000, "Rayleigh")
for i in range(a):
    print(np.sum(np.tile(ve[i, :], (a, 1)).T * ve, axis=0))
    print(np.abs(ve - test.matrix @ ve))
