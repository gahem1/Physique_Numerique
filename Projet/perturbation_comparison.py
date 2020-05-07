from hamiltonian import *
from perturbation import Perturbation
import matplotlib.pyplot as plt
import seaborn as sns

dim = 8
num = 50
denom = 10000
va, val = np.empty((num, num, dim)), np.empty((num, num, dim))
ve, vec = np.empty((num, dim, dim)), np.empty((num, dim, dim))
for i in range(num):
    for j in range(num):
        test1 = Operator(dim, i / denom, j / denom)
        test2 = Perturbation(dim, i / denom, j / denom)
        if i == 10:
            va[i, j, :], ve[j, :, :] = (test1.eigenalgo(10 ** -12, 99999, "Rayleigh"))[:2]
            fos = test2.first_order_state()
            vectors = np.diag(np.ones(dim)) + fos + test2.second_order_state(fos)
            norms = np.sqrt(np.sum(vectors * vectors, axis=0))
            for k in range(dim):
                vectors[:, k] = vectors[:, k] / norms[k]
            vec[j, :, :] = vectors
        else:
            va[i, j, :] = (test1.eigenalgo(10 ** -10, 99999, "Rayleigh"))[0]

        val[i, j, :] = test2.second_order_energy() + test2.first_order_energy() + np.arange(0.5, dim + 0.5)

data1 = np.sum(np.abs(va - val), axis=2) / dim
data2 = np.abs(va[20, :, :] - val[20, :, :])
data3 = np.sum(np.abs(ve - vec), axis=2) / dim

plt.figure()
ax = sns.heatmap(data1, cbar_kws={'label': 'Average energy difference with 2nd order perturbation theory'})
ax.set_xlabel("alpha", fontsize=18)
ax.set_ylabel("beta", fontsize=18)
ax.set_xticks(np.arange(0, num, 10) + 0.5)
ax.set_xticklabels(np.arange(0, num, 10) / denom, fontsize=18)
ax.set_yticks(np.arange(0, num, 10) + 0.5)
ax.set_yticklabels(np.arange(0, num, 10) / denom, fontsize=18)
ax.invert_yaxis()
ax.figure.axes[-1].yaxis.label.set_size(18)
plt.show()

plt.figure()
ax = sns.heatmap(data2.T, cbar_kws={'label': 'Energy difference with 2nd order perturbation theory'})
ax.set_xlabel("beta", fontsize=18)
ax.set_ylabel("Energy level", fontsize=18)
ax.set_xticks(np.arange(0, num, 10) + 0.5)
ax.set_xticklabels(np.arange(0, num, 10) / denom, fontsize=18)
ax.set_yticks(np.arange(dim) + 0.5)
ax.set_yticklabels(np.arange(dim), fontsize=18)
ax.invert_yaxis()
ax.figure.axes[-1].yaxis.label.set_size(18)
plt.show()

plt.figure()
ax = sns.heatmap(data3.T, cbar_kws={'label': 'Component difference with 2nd order perturbation theory'})
ax.set_xlabel("beta", fontsize=18)
ax.set_ylabel("Energy level", fontsize=18)
ax.set_xticks(np.arange(0, num, 10) + 0.5)
ax.set_xticklabels(np.arange(0, num, 10) / denom, fontsize=18)
ax.set_yticks(np.arange(dim) + 0.5)
ax.set_yticklabels(np.arange(dim), fontsize=18)
ax.invert_yaxis()
ax.figure.axes[-1].yaxis.label.set_size(18)
plt.show()
