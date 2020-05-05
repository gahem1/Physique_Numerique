from hamiltonian import Operator
import numpy as np
from random import random, randint

print("Gram-Schmidt")
a = randint(5, 10)
test = Operator(a, random(), random())
va, ve, di, gg, te = test.eigenalgo(10 ** -14, 20000, "Gram-Schmidt")
for i in range(a):
    print(np.sum(np.tile(ve[:, i], (a, 1)).T * ve, axis=0))
print(np.abs(ve @ np.diag(va) - test.matrix @ ve))

print("Givens")
a = randint(5, 10)
test = Operator(a, random(), random())
va, ve, di, gg, te = test.eigenalgo(10 ** -14, 20000, "Givens")
for i in range(a):
    print(np.sum(np.tile(ve[:, i], (a, 1)).T * ve, axis=0))
print(np.abs(ve @ np.diag(va) - test.matrix @ ve))

print("Rayleigh")
a = randint(5, 10)
test = Operator(a, random(), random())
va, ve, di, gg, te = test.eigenalgo(10 ** -14, 20000, "Rayleigh")
for i in range(a):
    print(np.sum(np.tile(ve[:, i], (a, 1)).T * ve, axis=0))
print(np.abs(ve @ np.diag(va) - test.matrix @ ve))

