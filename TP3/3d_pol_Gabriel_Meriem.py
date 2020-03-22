from phase_space_pol_Gabriel_Meriem import *
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    tlim = (0, 8 * np.pi)
    erreur_a = 10 ** -10
    erreur_r = 10 ** -10
    sol = solve_ivp(pol, tlim, np.array([1, 3]), method='RK45', atol=erreur_a, rtol=erreur_r)

    plt.figure()
    ax = plt.subplot(projection='3d')
    plt.plot(sol.y[0, :], sol.y[1, :], sol.t)
    plt.ylabel("xp", fontsize=14)
    plt.xlabel("x", fontsize=14)
    ax.set_zlabel("t", fontsize=14)
    plt.show()
