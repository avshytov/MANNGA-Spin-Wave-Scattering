import numpy as np
import pylab as pl
import constants
from slab import Slab
from dispersion import Dispersion

d_slab = 20 * constants.nm
Bext = 5 * constants.mT
Ms = 140 * constants.kA_m
Aex = 2 * 3.5 * constants.pJ_m
Jex = Aex / Ms**2
alpha = 1e-3


slab = Slab( - d_slab, 0, Bext, Ms, Jex, alpha, 10)
def omega_k(kx, ky):
    mode_plus_E, mode_minus_E  = slab.make_modes(kx, ky)
    mode_plus, E = mode_plus_E
    return mode_plus.omega

disp = Dispersion(omega_k)

s_vals = np.array([5, 10, 15, 20, 25, 30]) * constants.nm

o_vals = np.linspace(1.0, 10.0, 901) * constants.GHz_2pi
theta = 0.0

pl.figure()
for s in s_vals:
    F_vals = []
    for o in o_vals:
        print ("s = ", s, "o = ", o / constants.GHz_2pi)
        k_o = disp.k_omega(o, theta)
        u_o = disp.u_omega(o, theta)
        F_o = k_o**3 / u_o  * np.exp(-2.0 * k_o * s)
        F_vals.append(np.abs(F_o))
    F_vals = np.array(F_vals)
    pl.plot(o_vals / constants.GHz_2pi, F_vals,
            label=r'$s = %g$nm' % (s / constants.nm))

pl.legend()
pl.xlabel("Frequency $\omega / 2\pi$, GHz")
pl.ylabel("Radiative width $\Gamma_\mathrm{rad}$, a.u.")
pl.show()
