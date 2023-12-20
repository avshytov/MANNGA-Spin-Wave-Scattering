import pylab as pl
import numpy as np
import sys

from printconfig import print_config

def read_data(fname):
    d = np.load(fname)
    for k in d.keys():
        print (k)
    theta = d['theta']
    alpha_theta = d['alpha_theta']
    alpha = alpha_theta
    pl.figure()
    pl.plot(theta / np.pi * 180.0, alpha_theta / np.pi * 180.0)
    pl.plot(theta / np.pi * 180.0, theta / np.pi * 180.0, '--')
    pl.xlabel(r"$\theta$")
    pl.ylabel(r"$\alpha(\theta)$")
    for i in range(len(theta)):
        col = 'ok'
        if np.cos(theta[i] - alpha[i]) > 0:
            col = 'or'
        pl.plot([theta[i]/np.pi * 180.0], [alpha[i] / np.pi * 180.0],
                col, ms=1.0)     
    pl.figure()
    alpha_m = 0.5 * ( alpha[1:] + alpha[:-1])
    theta_m = 0.5 * ( theta[1:] + theta[:-1])
    dalpha = alpha[1:] - alpha[:-1]
    dtheta = theta[1:] - theta[:-1]
    pl.plot(theta_m / np.pi * 180.0, dtheta / dalpha)
    pl.plot(theta_m / np.pi * 180.0, dalpha / dtheta)
    pl.xlabel(r"$\theta$")
    pl.ylabel(r"$\alpha(\theta)$")
    

    if 'ug_theta' in d.keys():
       u_theta = d['ug_theta']
       v_theta = d['vg_theta']
    else:
       u_theta = d['u_theta']
       v_theta = d['v_theta']
    pl.figure()
    pl.polar(theta, u_theta.real, label=r'$u_\omega(\theta)$')
    pl.polar(theta, v_theta.real, label=r'$v_\omega(\theta)$')
    pl.legend()
    pl.figure()
    pl.plot(theta, u_theta.real, label=r'$u_\omega(\theta)$')
    pl.plot(theta, v_theta.real, label=r'$v_\omega(\theta)$')
    pl.legend()
    pl.figure()
    pl.polar(alpha, u_theta.real, label=r'$u_\omega(\alpha)$')
    pl.polar(alpha, v_theta.real, label=r'$v_\omega(\alpha)$')
    pl.legend()
    k_theta = d['k_theta']
    pl.figure()
    pl.polar(theta, k_theta, label=r'$k_\omega(\theta)$')
    pl.legend()
    pl.figure()
    pl.plot(theta, k_theta, label=r'$k_\omega(\theta)$')
    pl.legend()
    pl.figure()
    pl.polar(alpha, k_theta, label=r'$k_\omega(\alpha)$')
    pl.legend()
    Delta_theta = d['Delta_theta']
    pl.figure()
    pl.polar(theta, np.abs(Delta_theta), label=r'$\Delta(\theta)$')
    pl.legend()
    pl.figure()
    pl.polar(alpha, np.abs(Delta_theta), label=r'$\Delta_\omega(\alpha)$')
    pl.legend()
    pl.figure()
    pl.plot(theta, np.abs(Delta_theta), label=r'$\Delta_\omega(\theta)$')
    pl.legend()
    print_config(d)
    pl.show()


for fname in sys.argv[1:]:
    read_data(fname)
