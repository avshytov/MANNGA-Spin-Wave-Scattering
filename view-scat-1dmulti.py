import numpy as np
import pylab as pl
import sys

from printconfig import print_config

from constants import ns, GHz_2pi

def readData(fname):
    d = np.load(fname)
    for k in d.keys():
        print(k)

    omega = d['omega']
    T1 = d['T1']
    R1 = d['R1']
    A1 = d['A1']
    T2 = d['T2']
    R2 = d['R2']
    A2 = d['A2']
    if 'Gamma_rad' in d.keys():
        Gamma_rad = d['Gamma_rad']
    else:
        Gamma_rad = 0.0 * omega

    pl.figure()
    pl.plot(omega / GHz_2pi, np.abs(T1), label=r'$|T_R(\omega)|$')
    pl.plot(omega / GHz_2pi, np.abs(T2), '--', label=r'$|T_L(\omega)|$')
    pl.plot(omega / GHz_2pi, np.abs(R1), label=r'$|R_R(\omega)|$')
    pl.plot(omega / GHz_2pi, np.abs(R2), '--', label=r'$|R_L(\omega)|$')
    pl.plot(omega / GHz_2pi, A1, label=r'$|A_R(\omega)|$')
    pl.plot(omega / GHz_2pi, A2, '--', label=r'$|A_L(\omega)|$')
    pl.xlabel(r"Frequency $\omega/2\pi$, GHz")
    pl.legend()

    pl.figure()
    Gamma_shape = np.shape(Gamma_rad)
    if len(Gamma_shape) < 2:
       pl.plot(omega / GHz_2pi, Gamma_rad, label=r'$\Gamma_{rad}(\omega)$')
    else:
       for i in range(Gamma_shape[1]):
           pl.plot(omega / GHz_2pi, Gamma_rad[:, i, i],
                   label=r'$\Gamma_{rad}(\omega)$ mode %d' % i)
    pl.xlabel(r"Frequency $\omega/2\pi$, GHz")
    pl.ylabel(r'Radiative linewidth $\Gamma_{rad}(\omega)$, ${ns}^{-1}$')
    
    
    print_config(d)
    pl.legend()
    pl.show()


for fname in sys.argv[1:]:
    readData(fname)
