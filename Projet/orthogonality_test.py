from hamiltonian import Operator
import numpy as np
from random import random, randrange, randint

print("Gram-Schmidt")
a = randint(5, 15)
test = Operator(a, random(), random())
va, ve, di, gg, te = test.eigenalgo(5 * 10 ** -12, 10000, "Gram-Schmidt")
for i in range(a):
    print(np.sum(np.tile(ve[:, i], (a, 1)).T * ve, axis=0))
print(np.abs(ve @ np.diag(va) - test.matrix @ ve))

print("Givens")
a = randint(5, 15)
test = Operator(a, random(), random())
va, ve, di, gg, te = test.eigenalgo(5 * 10 ** -12, 10000, "Givens")
for i in range(a):
    print(np.sum(np.tile(ve[:, i], (a, 1)).T * ve, axis=0))
print(np.abs(ve @ np.diag(va) - test.matrix @ ve))

print("Rayleigh")
a = randint(5, 15)
test = Operator(a, randrange(1, 50000, 1) / 1000000, randrange(1, 50000, 1) / 1000000)
va, ve, di, gg, te = test.eigenalgo(5 * 10 ** -12, 10000, "Rayleigh")
print(gg)
for i in range(a):
    print(np.sum(np.tile(ve[:, i], (a, 1)).T * ve, axis=0))
print(np.abs(ve @ np.diag(va) - test.matrix @ ve))
