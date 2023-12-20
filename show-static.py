import numpy as np
import pylab as pl
import sys

def show_static(fname):
    d = np.load(fname)
    for k in d.keys():
        print (k)
    X, Y, Z = d['X'], d['Y'], d['Z']
    n_x, n_y, n_z = d['n_x'], d['n_y'], d['n_z']
    H_x, H_y, H_z = d['H_x'], d['H_y'], d['H_z']
    for z in [np.max(Z), np.min(Z)]:
        z_i = Z[np.argmin(np.abs(Z - z))]
        i_z = [t for t in range(len(Z)) if abs(z_i - Z[t]) < 1e-6]
        X_z = np.array([X[t] for t in i_z])
        Y_z = np.array([Y[t] for t in i_z])
        nx_z = np.array([n_x[t] for t in i_z])
        ny_z = np.array([n_y[t] for t in i_z])
        nz_z = np.array([n_z[t] for t in i_z])
        Hx_z = np.array([H_x[t] for t in i_z])
        Hy_z = np.array([H_y[t] for t in i_z])
        Hz_z = np.array([H_z[t] for t in i_z])
        Habs_z = np.sqrt(Hx_z**2 + Hy_z**2  + Hz_z**2)
        pl.figure()
        pl.title("%s @ z = %g" % (fname, z_i))
        pl.tripcolor(X_z, Y_z, Habs_z)
        pl.gca().set_aspect('equal', 'box')
        pl.colorbar()

        pl.figure()
        pl.gca().set_aspect('equal', 'box')
        
        pl.quiver(X_z[::4], Y_z[::4], nx_z[::4], ny_z[::4], pivot='mid',
                  scale=150, scale_units='xy', headwidth=5, headlength=10)
        pl.title("nx, ny")
        
        pl.figure()
        pl.gca().set_aspect('equal', 'box')
        pl.quiver(X_z, Y_z, nx_z, ny_z - 1.0)
        pl.title("dnx, dny")

        pl.figure()
        pl.gca().set_aspect('equal', 'box')
        pl.tripcolor(X_z, Y_z, np.sqrt(nx_z**2 + nz_z**2 + (ny_z - 1.0)**2))
        pl.title("dn")
        pl.colorbar()
        
        pl.figure()
        pl.gca().set_aspect('equal', 'box')
        #pl.tripcolor(X_z, Y_z, Habs_z)
        if z_i > 0:
           pl.quiver(X_z[::4], Y_z[::4], Hx_z[::4], Hy_z[::4], Habs_z[::4],
                  pivot='mid', scale_units='xy',
                  scale=50, headwidth=5, headlength=10, cmap='plasma')
        else:
           pl.quiver(X_z[::3], Y_z[::3], Hx_z[::3], Hy_z[::3], Habs_z[::3],
                  pivot='mid', scale_units='xy',
                  scale=1, headwidth=5, headlength=10, cmap='plasma')
        pl.title("Hx, Hy")
    
    
    

for fname in sys.argv[1:]:
    show_static(fname)
    pl.show()
