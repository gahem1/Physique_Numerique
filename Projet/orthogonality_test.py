from operator import Operator
from random import random, randint
import numpy as np

print("Gram-Schmidt")
a = randint(5, 20)
test = Operator(a, random(), random())
va, ve, di, gg, te = test.eigenalgo(10 ** -12, 10000, "Gram-Schmidt")
for i in range(a):
    print(np.sum(np.tile(ve[i, :]).T * ve, axis=0))
    print(np.abs(ve - test.matrix @ ve))

print("Givens")
a = randint(5, 20)
test = Operator(a, random(), random())
va, ve, di, gg, te = test.eigenalgo(10 ** -12, 10000, "Givens")
for i in range(a):
    print(np.sum(np.tile(ve[i, :]).T * ve, axis=0))
    print(np.abs(ve - test.matrix @ ve))

print("Rayleigh")
a = randint(5, 20)
test = Operator(a, random(), random())
va, ve, di, gg, te = test.eigenalgo(10 ** -12, 10000, "Rayleigh")
for i in range(a):
    print(np.sum(np.tile(ve[i, :]).T * ve, axis=0))
    print(np.abs(ve - test.matrix @ ve))
