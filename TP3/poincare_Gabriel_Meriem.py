from van_der_pol_Gabriel_Meriem import *


if __name__ == "__main__":
    tlim = (0, 16 * np.pi)
    erreur_a = 10 ** -10
    erreur_r = 10 ** -10

    cycle_lim = solve_ivp(pol, tlim, np.array([2, 0]), method='RK45', atol=erreur_a, rtol=erreur_r)
    tlim2, condz = (0, 10), cycle_lim.y[:, -1]
    cycle_lim = solve_ivp(pol, tlim2, condz, method='RK45', atol=erreur_a, rtol=erreur_r)
    ma = int(len(cycle_lim.t) / 2)
    potential_times = cycle_lim.t[:ma]
    dist = ((cycle_lim.y[0, -1] - cycle_lim.y[0, :ma]) ** 2 + (cycle_lim.y[1, -1] - cycle_lim.y[1, :ma]) ** 2)
    period = (cycle_lim.t[-1] - potential_times[dist == np.amin(dist)])
    print(period)

    tlim = (0, 6 * period + 0.5)
    sol = solve_ivp(pol, tlim, np.array([1, 3]), method='RK45', atol=erreur_a, rtol=erreur_r)
    test_values = [0]
    for i in range(1, 6):
        T = i * period
        proximity = abs(sol.t - T)
        test_values.append(np.argwhere(proximity == np.amin(proximity))[0][0])

    plt.figure()
    plt.plot(sol.y[0, :][test_values][0], sol.y[1, :][test_values][0], '.')
    plt.plot(sol.y[0, :][test_values][1], sol.y[1, :][test_values][1], '.')
    plt.plot(sol.y[0, :][test_values][2], sol.y[1, :][test_values][2], '.')
    plt.plot(sol.y[0, :][test_values][3], sol.y[1, :][test_values][3], '.')
    plt.plot(sol.y[0, :][test_values][4], sol.y[1, :][test_values][4], '.')
    plt.plot(sol.y[0, :][test_values][5], sol.y[1, :][test_values][5], '.')
    plt.ylabel("xp", fontsize=12)
    plt.xlabel("x", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(["t=0", "t=ti", "t=2ti", "t=3ti", "t=4ti", "t=5ti", "t=6ti", "t=7ti"], fontsize=12, loc='upper left')
    plt.show()
