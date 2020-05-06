from hamiltonian import *
from perturbation import Perturbation
import matplotlib.pyplot as plt
import seaborn as

dim = 10
num = 200
va, val = np.empty((num, num, dim)), np.empty((num, num, dim))

for i in range(num):
    for j in range(num):
        test1 = Operator(dim, i / 2000, j / 2000)
        test2 = Perturbation(dim, i / 2000, j / 2000)
        va[i, j, :], ve, di, gg, te = test1.eigenalgo(10 ** -12, 99999, "Givens")
        val[i, j, :] = test2.second_order_energy() + test2.first_order_energy() + np.arange(0.5, dim + 0.5)



print(val)
print(va)
