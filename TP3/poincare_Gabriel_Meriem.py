from van_der_pol_Gabriel_Meriem import *


if __name__ == "__main__":
    tlim = (0, 10)
    erreur_a = 10 ** -10
    erreur_r = 10 ** -10

    cycle_lim = solve_ivp(pol, tlim, np.array([2, 0]), method='RK45', atol=erreur_a, rtol=erreur_r)
    ma = int(len(cycle_lim.t) / 2)
    potential_times = cycle_lim.t[:ma]
    dist = ((cycle_lim.y[0, -1] - cycle_lim.y[0, :ma]) ** 2 + (cycle_lim.y[1, -1] - cycle_lim.y[1, :ma]) ** 2)
    period = (cycle_lim.t[-1] - potential_times[dist == np.amin(dist)])
    print(period)

    tlim = (0, 6 * period + 0.5)
    sol = solve_ivp(pol, tlim, np.array([1, 0]), method='RK45', atol=erreur_a, rtol=erreur_r)
    test_values = [0]
    for i in range(1, 7):
        T = i * period
        proximity = abs(sol.t - T)
        test_values.append(np.argwhere(proximity == np.amin(proximity))[0][0])

    plt.figure()
    plt.plot(sol.y[0, :][test_values], sol.y[1, :][test_values], '.')
    plt.ylabel("xp", fontsize=18)
    plt.xlabel("x", fontsize=18)
    plt.show()
