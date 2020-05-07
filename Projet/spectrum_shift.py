from hamiltonian import *
import matplotlib.pyplot as plt

num = 3
dim = 18
axis = np.arange(dim)
en = np.empty((num, num, dim))
for i in range(num):
    for j in range(num):
        test = Operator(dim, (i + 1) * 0.005, (j + 1) * 0.005)
        en[i, j, ::-1] = (test.eigenalgo(10 ** -12, 99999999, "Givens"))[0]

list_of_markers = ['.', '*', '1']

plt.figure()
for i in range(num):
    k = 0
    for j in range(num):
        plt.plot(axis, en[i, j, :], list_of_markers[k], markersize=8)
        k += 1

plt.xlabel("Niveau d'énergie", fontsize=18)
plt.ylabel("Énergie", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(("alpha=0.005 beta=0.005", "alpha=0.005 beta=0.01", "alpha=0.005 beta=0.015", "alpha=0.01 beta=0.005", "alpha=0.01 beta=0.01", "alpha=0.01 beta=0.015", "alpha=0.015 beta=0.005", "alpha=0.015 beta=0.01", "alpha=0.015 beta=0.015"), fontsize=14)
plt.show()
