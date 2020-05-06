from hamiltonian import *
from perturbation import Perturbation

test1 = Operator(10, 0, 0.0001)
test2 = Perturbation(10, 0, 0.0001)

va, ve, di, gg, te = test1.eigenalgo(10 ** -14, 50000, "Givens")
val = test2.first_order_energy() + np.arange(0.5, 10.5)
vec = test2.first_order_state()

print(val)
print(va)
