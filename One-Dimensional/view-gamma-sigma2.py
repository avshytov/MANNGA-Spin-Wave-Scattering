import numpy as np
import pylab as pl
from constants import GHz_2pi

d_sigma = np.load('sigma-nrf.npz')
d_Gamma = np.load('Gamma_for_sigma-nrf.npz')
for k in d_sigma.keys(): print(k)
pl.figure()
print (np.shape(d_sigma['Sigma_11']))
pl.plot(d_sigma['omegas'] / GHz_2pi,
        d_sigma['Sigma_11'][0, 0, :], label=r'Re $\Sigma_{11}$')
pl.plot(d_sigma['omegas'] / GHz_2pi,
        d_sigma['Sigma_12'][0, 0, :].real, label=r'Re $\Sigma_{12}$')
pl.plot(d_sigma['omegas'] / GHz_2pi,
        d_sigma['Sigma_12'][0, 0, :].imag, label=r'Im $\Sigma_{12}$')
pl.plot(d_sigma['omegas'] / GHz_2pi,
        d_sigma['Sigma_21'][0, 0, :].real, label=r'Re $\Sigma_{21}$')
pl.plot(d_sigma['omegas'] / GHz_2pi,
        d_sigma['Sigma_21'][0, 0, :].imag, label=r'Im $\Sigma_{21}$')
pl.plot(d_sigma['omegas'] / GHz_2pi,
        d_sigma['Sigma_22'][0, 0, :], label=r'Re $\Sigma_{22}$')
pl.legend()

pl.figure()
for k in d_Gamma.keys(): print(k)
print (np.shape(d_Gamma['Gamma_11_ab']))
pl.plot(d_Gamma['o_tab'] / GHz_2pi,
        d_Gamma['Gamma_11_ab'][0, 0, :], label=r'$\Gamma_{11}$')
pl.plot(d_Gamma['o_tab'] / GHz_2pi,
        d_Gamma['Gamma_22_ab'][0, 0, :], label=r'$\Gamma_{22}$')
pl.plot(d_Gamma['o_tab'] / GHz_2pi,
        d_Gamma['Gamma_12_ab'][0, 0, :].real, label=r'Re $\Gamma_{12}$')
pl.plot(d_Gamma['o_tab'] / GHz_2pi,
        d_Gamma['Gamma_12_ab'][0, 0, :].imag, label=r'Im $\Gamma_{12}$')
pl.plot(d_Gamma['o_tab'] / GHz_2pi,
        d_Gamma['Gamma_21_ab'][0, 0, :].real, label=r'Re $\Gamma_{21}$')
pl.plot(d_Gamma['o_tab'] / GHz_2pi,
        d_Gamma['Gamma_21_ab'][0, 0, :].imag, label=r'Im $\Gamma_{21}$')
pl.legend()

pl.show()
