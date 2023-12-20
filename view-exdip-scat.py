import pylab as pl
import numpy as np
import sys

from constants import GHz, GHz_2pi

from printconfig import print_config 

def plot_sorted(x, y, *args, **kwargs):
    xy = zip(list(x), list(y))
    xy = list(xy)
    xy.sort(key = lambda t: t[0])
    x_new = np.array([t[0] for t in xy])
    y_new = np.array([t[1] for t in xy])
    pl.plot(x_new, y_new, *args, **kwargs)

def read_data(fname):
    d = np.load(fname)
    for k in d.keys():
        print (k)
    pl.figure()
    plot_sorted(d['omega_tab'] / GHz_2pi, d['gamma_rad_o'])
    pl.title(r"$\Gamma_\mathrm{rad}(\omega)$")
    pl.xlabel("Frequency $\omega/2\pi$, GHz")

    result = d['result']
    theta_tab = d['theta_tab']
    omega_tab = d['omega_tab']
    print ("result: ", np.shape(result))
    pl.figure()
    for i_theta, theta in enumerate(theta_tab):
        plot_sorted(omega_tab / GHz_2pi, np.abs(result[i_theta, :])**2,
                label=r'$\theta = %g$' % (theta * 180.0/np.pi))
    pl.xlabel(r"Frequency $\omega/2\pi$, GHz")
    pl.ylabel(r"Differential cross-section $|f(\theta)|^2$")
    pl.legend()
    pl.figure()
    i_zero = np.argmin(np.abs(theta_tab - 0))
    plot_sorted(omega_tab / GHz_2pi, np.abs(result[i_zero, :])**2,
                label=r'$\theta = %g$' % (theta_tab[i_zero] * 180.0/np.pi))
    pl.xlabel(r"Frequency $\omega/2\pi$, GHz")
    pl.ylabel(r"Differential cross-section $|f(0)|^2$")
    pl.legend()
    if True and 'scat_tot_o' in d.keys():
        pl.figure()
        print (np.shape(d['scat_tot_o']), np.shape(omega_tab))
        plot_sorted(omega_tab / GHz_2pi, d['scat_tot_o'],
                label='Scattering cross-secion')
        plot_sorted(omega_tab / GHz_2pi, d['scat_abs_o'],
                label='Absorption cross-secion')
        pl.legend()


    print_config(d)
    pl.show()


for fname in sys.argv[1:]:
    read_data(fname)
