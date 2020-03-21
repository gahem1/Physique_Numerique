from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


def pol(t, phasei):
    """Le premier élément de phasei/f est x et le deuxième est xpoint"""
    phasef = np.empty(2)
    phasef[0] = phasei[1]
    phasef[1] = -(phasei[0] + epsilon * (phasei[0] ** 2 - 1) * phasei[1])

    return phasef


epsilon = 1

if __name__ == "__main__":
    tlim = (0, 8 * np.pi)
    cond_initiales = np.array([0.5, 0])
    erreur_a = 10 ** -10
    erreur_r = 10 ** -10

    sol = solve_ivp(pol, tlim, cond_initiales, method='RK45', atol=erreur_a, rtol=erreur_r)

    plt.figure()
    plt.plot(sol.t, sol.y[0, :], sol.t, sol.y[1, :], linewidth=2)
    plt.ylabel("f(t)", fontsize=18)
    plt.xlabel("t", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(["x(t)", "xp(t)"], loc="upper left", fontsize=16)
    plt.show()
