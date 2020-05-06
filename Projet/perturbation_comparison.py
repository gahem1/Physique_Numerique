from hamiltonian import *
from perturbation import Perturbation
import matplotlib.pyplot as plt
import seaborn as sns

dim = 10
num = 50
denom = 10000
va, val = np.empty((num, num, dim)), np.empty((num, num, dim))
ve, vec = np.empty((num, num, dim, dim)), np.empty((num, num, dim, dim))
for i in range(num):
    for j in range(num):
        test1 = Operator(dim, i / denom, j / denom)
        test2 = Perturbation(dim, i / denom, j / denom)
        va[i, j, ::-1], ve[i, j, ::-1, ::-1] = (test1.eigenalgo(10 ** -10, 99999, "Givens"))[:2]
        val[i, j, :] = test2.second_order_energy() + test2.first_order_energy() + np.arange(0.5, dim + 0.5)
        fos = test2.first_order_state()
        vectors = np.diag(np.ones(dim)) + fos + test2.second_order_state(fos)
        norms = np.sqrt(np.sum(vectors * vectors, axis=0))
        for k in range(dim):
            vectors[:, k] = vectors[:, k] / norms[k]

        vec[i, j, :, :] = vectors

data1 = np.sum(np.abs(va - val), axis=2) / dim
data2 = np.sum(np.abs(ve - vec)) / dim ** 2

ax = sns.heatmap(data1, cbar_kws={'label': 'Average energy difference with 2nd order perturbation theory'})
ax.set_xlabel("beta", fontsize=18)
ax.set_ylabel("alpha", fontsize=18)
ax.set_xticks(np.arange(0, num, 10) + 0.5)
ax.set_xticklabels(np.arange(0, num, 10) / denom, fontsize=18)
ax.set_yticks(np.arange(0, num, 10) + 0.5)
ax.set_yticklabels(np.arange(0, num, 10) / denom, fontsize=18)
ax.invert_yaxis()
ax.figure.axes[-1].yaxis.label.set_size(18)
plt.show()

ax = sns.heatmap(data2, cbar_kws={'label': 'Average component difference with 2nd order perturbation theory'})
ax.set_xlabel("beta", fontsize=18)
ax.set_ylabel("alpha", fontsize=18)
ax.set_xticks(np.arange(0, num, 10) + 0.5)
ax.set_xticklabels(np.arange(0, num, 10) / denom, fontsize=18)
ax.set_yticks(np.arange(0, num, 10) + 0.5)
ax.set_yticklabels(np.arange(0, num, 10) / denom, fontsize=18)
ax.invert_yaxis()
ax.figure.axes[-1].yaxis.label.set_size(18)
plt.show()
