import numpy as np
import pylab as pl
from constants import GHz_2pi

d_sigma = np.load('sigma-nrf.npz')
d_Gamma = np.load('Gamma_for_sigma-nrf.npz')
for k in d_sigma.keys(): print(k)

N, M = np.shape(d_sigma['Sigma_11'][:, :, 0])

for i in range(0, min(5, N)):
    pl.figure()
    pl.title("Sigma for mode %d" % i)
    print (np.shape(d_sigma['Sigma_11']))
    pl.plot(d_sigma['omegas'] / GHz_2pi,
            d_sigma['Sigma_11'][i, i, :], label=r'Re $\Sigma_{++}$')
    pl.plot(d_sigma['omegas'] / GHz_2pi,
            d_sigma['Sigma_12'][i, i, :].real, label=r'Re $\Sigma_{+-}$')
    pl.plot(d_sigma['omegas'] / GHz_2pi,
            d_sigma['Sigma_12'][i, i, :].imag, label=r'Im $\Sigma_{+-}$')
    pl.plot(d_sigma['omegas'] / GHz_2pi,
            d_sigma['Sigma_21'][i, i, :].real, label=r'Re $\Sigma_{-+}$')
    pl.plot(d_sigma['omegas'] / GHz_2pi,
            d_sigma['Sigma_21'][i, i, :].imag, label=r'Im $\Sigma_{-+}$')
    pl.plot(d_sigma['omegas'] / GHz_2pi,
            d_sigma['Sigma_22'][i, i, :], label=r'Re $\Sigma_{--}$')
    pl.legend()

    pl.figure()
    pl.title("Gamma for mode %d" % i)
    for k in d_Gamma.keys(): print(k)
    print (np.shape(d_Gamma['Gamma_11_ab']))
    pl.plot(d_Gamma['o_tab'] / GHz_2pi,
            d_Gamma['Gamma_11_ab'][i, i, :], label=r'$\Gamma_{++}$')
    pl.plot(d_Gamma['o_tab'] / GHz_2pi,
            d_Gamma['Gamma_22_ab'][i, i, :], label=r'$\Gamma_{--}$')
    pl.plot(d_Gamma['o_tab'] / GHz_2pi,
            d_Gamma['Gamma_12_ab'][i, i, :].real, label=r'Re $\Gamma_{+-}$')
    pl.plot(d_Gamma['o_tab'] / GHz_2pi,
            d_Gamma['Gamma_12_ab'][i, i, :].imag, label=r'Im $\Gamma_{+-}$')
    pl.plot(d_Gamma['o_tab'] / GHz_2pi,
            d_Gamma['Gamma_21_ab'][i, i, :].real, label=r'Re $\Gamma_{-+}$')
    pl.plot(d_Gamma['o_tab'] / GHz_2pi,
            d_Gamma['Gamma_21_ab'][i, i, :].imag, label=r'Im $\Gamma_{-+}$')
    pl.legend()

pl.show()
