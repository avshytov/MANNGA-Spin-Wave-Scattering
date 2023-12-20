import numpy as np
import pylab as pl
import sys
import constants
from scipy import interpolate

def Sigma(o_vals, o_tab, Gamma_tab):
    Sigma_vals = np.zeros((len(o_vals)))
    for i_o, o in enumerate(o_vals):
        S_o = 0.0 # + 0.0j
        for j_o, o_1 in enumerate(o_tab[:-1]):
            o_2 = o_tab[j_o + 1]
            F_1 = Gamma_tab[j_o    ] / o_1**2
            F_2 = Gamma_tab[j_o + 1] / o_2**2
            I_log   = np.log(np.abs((o - o_1)/(o - o_2)))
            I_const = - (o_2 - o_1)
            F_prime = (F_2 - F_1) / (o_2 - o_1)
            F_o = F_1 + (o - o_1) * F_prime
            S_o += F_o * I_log + F_prime * I_const
            #print ("o, o1, S", o / GHz_2pi, o_1 / GHz_2pi, S_o)
        S_o *= o**2 / np.pi
        Sigma_vals[i_o] += S_o
    return Sigma_vals
    
def show_gamma(fname):
    d = np.load(fname)
    pl.figure()
    omega = d['omega']
    Gamma = d['Gamma']
    pl.plot(d['omega'] / constants.GHz_2pi, d['Gamma'], label='Gamma')
    Gamma_spl = interpolate.splrep(omega, Gamma)
    omega_fine = np.linspace(omega[0], omega[-1], 10 * (len(omega) - 1) + 1)
    Gamma_fine = interpolate.splev(omega_fine, Gamma_spl)
    pl.plot(omega_fine / constants.GHz_2pi, Gamma_fine, '--',
            label='Gamma interolated')
    omega_m = 0.5 * (omega_fine[1:] + omega_fine[:-1])
    Sigma_tab  = Sigma(omega_m, omega_fine, Gamma_fine)
    #print ("Gamma_fine = ", Gamma_fine)
    pl.plot(omega_m / constants.GHz_2pi, Sigma_tab, label='Re Sigma')
    pl.legend()
    pl.title(fname)
    pl.show()

for fname in sys.argv[1:]:
    show_gamma(fname)
