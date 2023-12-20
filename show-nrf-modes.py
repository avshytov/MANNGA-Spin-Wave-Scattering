import numpy as np
import pylab as pl
import sys
import constants

from matplotlib.colorbar import Colorbar

def show_mode(fname):
    d = np.load(fname)
    #for k in d.keys():
    #    print (k)
    f = d['omega'] / constants.GHz_2pi
    print ("Frequency", f)
    X = d['X']
    Y = d['Y']
    Z = d['Z']
    if  False:
        grid_f = open("grid-data-fname.dat", "w")
        for x, y, z in zip(X, Y, Z):
            grid_f.write("%g\t%g\t%g\n" % (x, y, z))
        grid_f.close()
    ma = d['ma']
    mb = d['mb']
    z_list = [np.min(Z)]
    for z in z_list:
        z_i = Z[np.argmin(np.abs(Z - z))]
        i_z = [t for t in range(len(Z))  if abs(Z[t] - z_i) < 1e-6]
        X_z = np.array([X[t] for t in i_z])
        Y_z = np.array([Y[t] for t in i_z])
        ma_z = np.array([ma[t] for t in i_z])
        mb_z = np.array([mb[t] for t in i_z])
        pl.figure()
        ax_global = pl.gca()
        ax_global.clear()
        ax_global.set_frame_on(False)
        ax_global.set_xticks([])
        ax_global.set_yticks([])
        pl.suptitle("Near-field mode")
        pl.title(" @ %g + i %g GHz" % (f.real, f.imag))
        #ax_global.set_xlabel([])
        #ax_global.set_ylabel([])
        ax_ma_abs = pl.axes([0.25, 0.15, 0.23, 0.3])
        ax_mb_abs = pl.axes([0.25, 0.55, 0.23, 0.3])
        ax_ma_angle = pl.axes([0.58, 0.15, 0.23, 0.3])
        ax_mb_angle = pl.axes([0.58, 0.55, 0.23, 0.3])
        ax_cb_left = pl.axes([0.10,  0.2, 0.025, 0.6])
        ax_cb_right = pl.axes([0.86, 0.2, 0.025, 0.6])
        #ma_res[mask > 0] = np.inf
        #mb_res[mask > 0] = np.inf
        for ax in [ax_ma_abs, ax_mb_abs, ax_ma_angle, ax_mb_angle]:
           ax.set_aspect('equal', 'box')

        dx = 1.2 * (np.max(X_z) - np.min(X_z))
        dy = 1.2 * (np.max(Y_z) - np.min(Y_z))
        m_scale = max(np.max(np.abs(ma_z)), np.max(np.abs(mb_z)))
        ma_abs_pc = ax_ma_abs.tripcolor(X_z / constants.nm, Y_z / constants.nm,
                                        np.abs(ma_z) / m_scale,
                                        vmin=0.0, vmax=1.0,
                                        cmap='magma')
        ax_ma_abs.set_xlabel("$x$, nm")
        ax_ma_abs.set_ylabel("$y$, nm")
        mb_abs_pc = ax_mb_abs.tripcolor(X_z / constants.nm,
                                        Y_z / constants.nm + 0 * dy,
                                        np.abs(mb_z) / m_scale,
                                        vmin=0.0, vmax=1.0,
                                        cmap='magma')
        #pl.colorbar(location='left')
        ax_ma_abs.annotate(r"$m_a$", (0.07, 0.93), (0.07, 0.93),
                           xycoords='axes fraction',
                           textcoords='axes fraction',
                           backgroundcolor='white',
                           bbox=dict(facecolor='white', alpha=0.5,
                                     edgecolor='white'),
                           horizontalalignment='left',
                           verticalalignment='top')
        ax_mb_abs.annotate(r"$m_b$", (0.07, 0.93), (0.07, 0.93),
                           xycoords='axes fraction',
                           textcoords='axes fraction',
                           backgroundcolor='white',
                           bbox=dict(facecolor='white', alpha=0.5,
                                     edgecolor='white'),
                           horizontalalignment='left',
                           verticalalignment='top')
        cb = Colorbar(ax_cb_left, ma_abs_pc, location='left')
        cb.set_label(r"$|m_{a, b}(x, y)|$")
        ma_ang_pc = ax_ma_angle.tripcolor(X_z / constants.nm + 0 * dx,
                                 Y_z / constants.nm,
                                 np.angle(ma_z) * 180.0 / np.pi,
                     vmin = -180, vmax=180,
                     cmap='hsv')
        mb_ang_pc = ax_mb_angle.tripcolor(X_z / constants.nm + 0 *dx,
                                          Y_z / constants.nm + 0 * dy,
                     np.angle(mb_z) * 180.0/np.pi,
                     vmin = -180.0, vmax=180.0, cmap='hsv')
        ax_ma_angle.annotate(r"$m_a$", (0.07, 0.93), (0.07, 0.93),
                           xycoords='axes fraction',
                           textcoords='axes fraction',
                           backgroundcolor='white',
                           bbox=dict(facecolor='white', alpha=0.5,
                                     edgecolor='white'),
                           horizontalalignment='left',
                           verticalalignment='top')
        ax_mb_angle.annotate(r"$m_b$", (0.07, 0.93), (0.07, 0.93),
                           xycoords='axes fraction',
                           textcoords='axes fraction',
                           backgroundcolor='white',
                           bbox=dict(facecolor='white', alpha=0.5,
                                     edgecolor='white'),
                           horizontalalignment='left',
                           verticalalignment='top')
        #cb = pl.colorbar(location = 'right')
        cb = Colorbar(ax_cb_right, mb_ang_pc, location='right')
        cb.set_ticks([-180.0, -90.0, 0, 90.0, 180.0])
        #pl.gca().set_aspect('equal', 'box')
        cb.set_label(r"${\rm arg} m_{a, b}(x, y)$") 
        #pl.title("z = %g f = %g + i %g" % (z, f.real, f.imag))

    

for fname in sys.argv[1:]:
    show_mode(fname)

pl.show()
