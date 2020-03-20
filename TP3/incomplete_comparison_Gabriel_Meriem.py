from oscillateur_anharmonique_Gabriel_Meriem import *


if __name__ == "__main__":
    omega = 1
    xi = 1
    xpi = 0
    
    xharm1, xpharm1 = rk4(N, xi, xpi, omega, xpp)
    xharm2, xpharm2 = rk4(N, xi * 3, xpi, omega, xpp)
    xanharm1, xpanharm1 = rk4(N, xi, xpi, omega, newxpp)
    xanharm2, xpanharm2 = rk4(N, xi * 3, xpi, omega, newxpp)

    xlin = np.array([0, xharm1[-1], 3 * xharm1[-1] / abs(xharm1[-1])])
    xplin = [0, xpharm1[-1], 3 * xpharm1[-1] / abs(xharm1[-1])]

    plt.figure()
    plt.plot(xharm1[::N - 1], xpharm1[::N - 1], '.', xharm2[::N - 1], xpharm2[::N - 1], '.', xanharm1[::N - 1],
             xpanharm1[::N - 1], '.', xanharm2[::N - 1], xpanharm2[::N - 1], '.', xlin, xplin, ':', markersize=10)
    plt.ylabel("xp", fontsize=18)
    plt.xlabel("x", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(["Harmonique avec x(0)={}".format(xi), "Harmonique avec x(0)={}".format(2 * xi),
                "Anharmonique avec x(0)={}".format(0.5 * xi), "Anharmonique avec x(0)={}".format(3 * xi)],
               loc="upper right", fontsize=14)
    plt.show()
