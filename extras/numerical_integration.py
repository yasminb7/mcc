import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import plotting
plt.ioff()

q = 0.5
delta = 0.3
gamma = 0.2
d = 1.0
zeta = 2.0
#m = 1.0

def kappa(t):
    return 1.0 if t>200 else 0.0

# solve the system dy/dt = f(y, t)
def f(y, t):
        x = y[0]
        v = y[1]
        E = y[2]
        # the model equations
        f0 = v
        f1 = d*E - gamma*v
        f2 = q - delta * E - d * v * E - kappa(t) * zeta * E
        return [f0, f1, f2]

# initial conditions
x0 = 0.0
v0 = 0.0
E0 = 0.0
y0 = [x0, v0, E0]
t  = np.linspace(0, 500., 100000)

# solve the DEs
soln = odeint(f, y0, t)
x = soln[:, 0]
v = soln[:, 1]
e = soln[:, 2]

y_axes = [e]
y_axes2 = [v]
mylegend = None
mylegend2 = mylegend
resultsfolder = "results/numerical_integration"
mysavefile = "numerical.eps"
plotting.plotlines_vert_subplot([t], [e], [t], [v], xlabel="Time", ylabel="Energy", xlabel2="Time", ylabel2="Velocity", legend=mylegend,legend2=mylegend2, folder=resultsfolder, savefile=mysavefile)

## plot results
#plt.figure()
#plt.plot(t, v, label='v')
#plt.plot(t, e, label='e')
##plt.x_lim((0, t_end))
#plt.xlabel('t')
##plt.ylabel('')
##plt.title('')
#plt.legend(loc=0)
#plt.savefig("integrated_plot.png")
