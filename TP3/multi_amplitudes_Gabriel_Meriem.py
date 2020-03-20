from oscillateur_harmonique_Gabriel_Meriem import *

if __name__ == "__main__":
    plt.figure()
    plt.plot(tvals, rk4(N, xi, xpi, ti)[0], tvals, rk4(N, 2 * xi, xpi, ti)[0], tvals, rk4(N, 3 * xi, xpi, ti)[0],
             linewidth=2)
    plt.ylabel("x(t)", fontsize=18)
    plt.xlabel("t", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(["x(0)={}".format(xi), "x(0)={}".format(2 * xi), "x(0)={}".format(3 * xi)], fontsize=12)
    plt.show()
