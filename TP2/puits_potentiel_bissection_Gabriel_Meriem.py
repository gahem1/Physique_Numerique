import puits_potentiel_intro_Gabriel_Meriem as puits


def transcendental_equation(energy, evenness: str):
    return puits.LHS(energy) - puits.RHS(puits.roots(energy), evenness)


error = 0.001
start_points = [0.29, 1.15, 2.75, 5.0, 7.75, 11.1]
end_points = [0.35, 1.35, 2.95, 5.1, 7.95, 11.3]

for state in range(6):
    start, end = start_points[state], end_points[state]
    N = puits.np.ceil(puits.np.log2((end - start) / error)).astype(int)

    if state % 2 == 0:
        parity = "even"
    else:
        parity = "odd"

    for i in range(N):
        midpoint = (start + end) / 2
        if transcendental_equation(start, parity) * transcendental_equation(midpoint, parity) > 0:
            start = midpoint
        else:
            end = midpoint

    print("The energy of the {0} state is {1:06.3f}eV".format(state, (start + end) / 2))
