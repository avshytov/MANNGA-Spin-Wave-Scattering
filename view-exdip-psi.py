import pylab as pl
import numpy as np
import sys

from constants import GHz_2pi, rad_um

from printconfig import print_config

def read_data(fname):
    d = np.load(fname)
    for k in d.keys():
        print (k)
    kx = d['kx']
    ky = d['ky']
    KY, KX = np.meshgrid(ky, kx)
    G = d['G']
    psi_scat = d['psi_scat']
    from matplotlib.colors import LogNorm 
    pl.figure()
    pl.pcolormesh(KX / rad_um, KY / rad_um, np.abs(G), cmap='magma',
                  norm=LogNorm(vmin=1.0, vmax=3e+1))
    pl.gca().set_aspect('equal', 'box')
    cb = pl.colorbar()
    pl.xlabel(r"$k_x$, [rad/$\mu$m]")
    pl.ylabel(r"$k_y$  [rad/$\mu$m]")
    cb.set_label(r"$|G(\omega, {\bf k})|$")
    pl.title(r"$f = %g$GHz" % (d['omega'] / GHz_2pi))
    
    pl.figure()
    vmin = 0.001
    vmax = 0.1 * np.max(np.abs(psi_scat)**2)
    vmax = min(1e+6, vmax)
    pl.pcolormesh(KX / rad_um, KY / rad_um, np.abs(psi_scat)**2,
                  cmap='magma',                          
                  norm=LogNorm(vmin=vmin, vmax=vmax))
    pl.gca().set_aspect('equal', 'box')
    cb = pl.colorbar()
    print ("max: ", np.max(np.abs(psi_scat)))
    pl.xlabel(r"$k_x$, [rad/$\mu$m]")
    pl.ylabel(r"$k_y$  [rad/$\mu$m]")
    cb.set_label(r"$|\psi_{\bf k}(\omega)|^2$")
    pl.title(r"$f = %g$GHz" % (d['omega'] / GHz_2pi))
    
    pl.figure()
    vmin = 0.0
    vmax = 0.01 * np.max(np.abs(psi_scat)**2)
    vmax = min(1e+4, vmax)
    pl.pcolormesh(KX, KY, np.abs(psi_scat)**2,
                  cmap='magma',                          
                  vmin=vmin, vmax=vmax)
    pl.gca().set_aspect('equal', 'box')
    cb = pl.colorbar()
    print ("max: ", np.max(np.abs(psi_scat)))
    cb.set_label(r"$|\psi_{\bf k}(\omega)|^2$")
    pl.xlabel(r"$k_x$, [rad/$\mu$m]")
    pl.ylabel(r"$k_y$  [rad/$\mu$m]")
    pl.title(r"$f = %g$GHz" % (d['omega'] / GHz_2pi))


    if False:
        xvals = np.linspace(-10.0, 10.0, 501)
        yvals = np.linspace(-10.0, 10.0, 501)
        Y, X = np.meshgrid(yvals, xvals)
        PSI = 0.0 * X + 0.0j
        for i in range(len(kx)):
            exp_X = np.exp(1j * kx[i] * xvals)
            print ("i", i, len(kx))
            for j in range(len(ky)):
                exp_Y = np.exp(1j * ky[j] * yvals)
                PSI += np.outer(exp_X, exp_Y) * psi_scat[i, j]
        pl.figure()
        pl.pcolormesh(X, Y, np.abs(PSI), cmap='magma')
        pl.gca().set_aspect('equal', 'box')
        pl.colorbar()
        pl.figure()
        pl.pcolormesh(X, Y, np.angle(PSI), cmap='hsv')
        pl.gca().set_aspect('equal', 'box')
        pl.colorbar()

    print_config(d)
    pl.show()


for fname in sys.argv[1:]:
    read_data(fname)
