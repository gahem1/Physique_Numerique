from oscillateur_harmonique_Gabriel_Meriem import *

if __name__ == "__main__":
    om = 1

    plt.figure()
    plt.plot(tvals, rk4(N, xi, xpi, om)[0], tvals, rk4(N, 2 * xi, xpi, om)[0], tvals, rk4(N, 3 * xi, xpi, om)[0],
             linewidth=2)
    plt.ylabel("x(t)", fontsize=18)
    plt.xlabel("t", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(["x(0)={}".format(xi), "x(0)={}".format(2 * xi), "x(0)={}".format(3 * xi)], fontsize=12)
    plt.show()
