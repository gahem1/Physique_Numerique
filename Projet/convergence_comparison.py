from hamiltonian import *
import matplotlib.pyplot as plt


gramj, givensj, rayleighj = np.empty(15), np.empty(15), np.empty(15)
gramt, givenst, rayleight = np.empty(15), np.empty(15), np.empty(15)
acc, alpha, beta, naxis = 10 ** -10, 0.0075, 0.007, np.arange(20, 5, -1)

test = Operator(20, alpha, beta)
for i in range(15):
    va, ve, di, gramj[i], gramt[i] = test.eigenalgo(acc, 99999999999, "Gram-Schmidt")
    test.reset_vap_vep()
    va, ve, di, givensj[i], givenst[i] = test.eigenalgo(acc, 99999999999, "Givens")
    test.reset_vap_vep()
    va, ve, di, rayleighj[i], rayleight[i] = test.eigenalgo(acc, 99999999999, "Rayleigh")
    test = Operator(test.N - 1, alpha, beta, test.matrix[:-1, :-1])

ran = 1000
jaxis, gramd1, givensd1, rayleighd1 = np.arange(ran), np.empty(ran), np.empty(ran), np.empty(ran)
gramd2, givensd2, rayleighd2 = np.empty(ran), np.empty(ran), np.empty(ran)
t1, t2, t3 = Operator(8, alpha, beta), Operator(8, alpha, beta), Operator(8, alpha, beta)
t4, t5, t6 = Operator(19, alpha, beta), Operator(19, alpha, beta), Operator(19, alpha, beta)
for i in range(ran):
    va, ve, gramd1[i], gg, te = t1.eigenalgo(0, 1, "Gram-Schmidt")
    va, ve, givensd1[i], gg, te = t2.eigenalgo(0, 1, "Givens")
    va, ve, rayleighd1[i], gg, te = t3.eigenalgo(0, 1, "Rayleigh")
    va, ve, gramd2[i], gg, te = t4.eigenalgo(0, 1, "Gram-Schmidt")
    va, ve, givensd2[i], gg, te = t5.eigenalgo(0, 1, "Givens")
    va, ve, rayleighd2[i], gg, te = t6.eigenalgo(0, 1, "Rayleigh")


plt.figure()
plt.plot(naxis, gramj, '*', naxis, givensj, '.', naxis, rayleighj, 'o', markersize=7)
plt.xlabel("Dimension", fontsize=18)
plt.ylabel("Nombre d'itérations", fontsize=18)
plt.yscale("log")
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(("Gram-Schmidt", "Givens", "Rayleigh"), fontsize=18)
plt.show()

plt.figure()
plt.plot(naxis, gramt, '*', naxis, givenst, '.', naxis, rayleight, 'o', markersize=7)
plt.xlabel("Dimension", fontsize=18)
plt.ylabel("Durée [s]", fontsize=18)
plt.yscale("log")
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(("Gram-Schmidt", "Givens", "Rayleigh"), fontsize=18)
plt.show()

plt.figure()
plt.plot(jaxis, gramd1, jaxis, givensd1, jaxis, rayleighd1, jaxis, gramd2, ':', jaxis, givensd2, ':', jaxis, rayleighd2, ':', linewidth=0.5)
plt.ylabel("Erreur", fontsize=18)
plt.xlabel("Nombre d'itérations", fontsize=18)
plt.yscale("log")
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(("Gram-Schmidt N=8", "Givens N=8", "Rayleigh N=8", "Gram-Schmidt N=18", "Givens N=18", "Rayleigh N=18"), fontsize=18, loc='upper right')
plt.show()
