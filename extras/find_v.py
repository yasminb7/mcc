import numpy as np
import matplotlib.pyplot as plt
import math
plt.ion()

import plotting

q = 0.5
delta = 0.3
gamma = 0.2
d = 1.0
zeta = 2.0 
#m = 1.0

def v(dissipation, q, plus=True):
    discr = dissipation**2 / (4*d*d) + q/gamma
    if plus:
        return - dissipation/(2*d) + math.sqrt( discr )
    else:
        return - dissipation/(2*d) - math.sqrt( discr )

v1 = v(delta, q, plus=True)
v2 = v(delta, q, plus=False)

v3 = v(delta+zeta, q, plus=True)
v4 = v(delta+zeta, q, plus=False) 

print v1
print v2
print v3
print v4

q  = np.linspace(0, 10., 1000)
v_q = [[v(delta+zeta, _) for _ in q] for zeta in [0, 5, 10]]
plotting.plotlines(len(v_q)*[q], v_q, folder="results/numerical_integration", savefile="v_q.eps")

d_values = [0.01, 0.1, 1, 10, 100]
v_d = [[v(delta+zeta, _, d) for _ in q] for d in d_values]
mylegend_d = ["d = %g" % _ for _ in d_values ]
plotting.plotlines(len(v_d)*[q], v_d, legend=mylegend_d, xlabel="q", ylabel="v", folder="results/numerical_integration", savefile="v_d.pdf")