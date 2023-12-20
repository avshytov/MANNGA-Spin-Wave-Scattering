import pylab as pl
import numpy as np
import sys

from constants import ns, um, GHz, GHz_2pi

rad_um = 1.0 / um
ns_inv = 1.0 / ns


def read_data(fname):
    d = np.load(fname)
    for k in d.keys():
        print (k)

    kx_vals = d['kx_vals']
    ky_vals = d['ky_vals']
    omega_x = d['omega_x']
    omega_y = d['omega_y']
    omega_x_n = d['omega_x_n']
    omega_y_n = d['omega_y_n']
    omega_x_naive = d['omega_x_naive']
    omega_y_naive = d['omega_y_naive']
    gamma_x_n = d['gamma_x_n']
    gamma_y_n = d['gamma_y_n']
    #gamma_x_naive = d['gamma_x_naive']
    #gamma_y_naive = d['gamma_y_naive']
    pl.figure()
    pl.plot(kx_vals / rad_um, omega_x.real / GHz_2pi,
            label=r'exact, ${\bf k} \parallel \hat{\bf x}$')
    pl.plot(ky_vals / rad_um, omega_y.real / GHz_2pi,
            label=r'exact, ${\bf k} \parallel \hat{\bf y}$')
    pl.plot(kx_vals / rad_um, omega_x_n / GHz_2pi, '--',
            label='Interpolation, DE')
    pl.plot(ky_vals / rad_um, omega_y_n / GHz_2pi, '--',
            label='Interpolation, BV')
    pl.plot(kx_vals / rad_um, omega_x_naive.real / GHz_2pi, '--', 
            label=r'thin film, ${\bf k} \parallel \hat{\bf x}$')
    pl.plot(ky_vals / rad_um, omega_y_naive.real / GHz_2pi, '--', 
            label=r'thin film, ${\bf k} \parallel \hat{\bf y}$')
    pl.xlabel(r"Wavenumber $k_{x, y}$, [rad/$\mu$m]")
    pl.ylabel(r"Frequency $\omega({\bf k})/2\pi$, GHz")
    pl.legend()

    pl.figure()
    pl.plot(kx_vals / rad_um, -omega_x.imag / ns_inv,
            label=r'exact, ${\bf k} \parallel \hat{\bf x}$')
    pl.plot(ky_vals / rad_um, -omega_y.imag / ns_inv,
            label=r'exact, ${\bf k} \parallel \hat{\bf y}$')
    pl.plot(kx_vals / rad_um, gamma_x_n / ns_inv,
            '--', label='Interpolation, DE')
    pl.plot(ky_vals / rad_um, gamma_y_n / ns_inv,
            '--', label='Interpolation, BV')
    #pl.plot(kx_vals, gamma_x_naive.real, '--', 
    #        label='thin film, ${\bm k} \parallel \hat{\bm x}$')
    #pl.plot(ky_vals, gamma_y_naive.real, '--', 
    #        label='thin film, ${\bm k} \parallel \hat{\bm y}$')
    pl.xlabel(r"Wavenumber $k_{x, y}$, [rad/$\mu$m]")
    pl.ylabel(r"Decay rate $\gamma({\bf k})$, $\rm{ns}^{-1}$")
    pl.legend()   
    pl.show()
    


for fname in sys.argv[1:]:
    read_data(fname)
