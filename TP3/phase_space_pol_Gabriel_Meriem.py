from van_der_pol_Gabriel_Meriem import *


if __name__ == "__main__":
    tlim = (0, 8 * np.pi)
    ci1 = np.array([1, 0])
    ci2 = np.array([2, 0])
    ci3 = np.array([3, 0])
    erreur_a = 10 ** -10
    erreur_r = 10 ** -10

    sol1 = solve_ivp(pol, tlim, ci1, method='RK45', atol=erreur_a, rtol=erreur_r)
    sol2 = solve_ivp(pol, tlim, ci2, method='RK45', atol=erreur_a, rtol=erreur_r)
    sol3 = solve_ivp(pol, tlim, ci3, method='RK45', atol=erreur_a, rtol=erreur_r)

    plt.figure()
    plt.plot(sol1.y[0, :], sol1.y[1, :], sol2.y[0, :], sol2.y[1, :], sol3.y[0, :], sol3.y[1, :])
    plt.ylabel("xp", fontsize=18)
    plt.xlabel("x", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(["x(0)=1", "x(0)=2", "x(0)=3"], loc="upper right", fontsize=16)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
