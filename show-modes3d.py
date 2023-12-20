import numpy as np
import pylab as pl
import sys
import numpy.ma as npma
import constants

def show_res_mode(XR, YR, ZR, mask, ma_res, mb_res, title, subtitle):   
    pl.figure()
    from matplotlib.colorbar import Colorbar
    #ma_res, mb_res = mode.m_ab('resonator')
    #ma_res = npma.array(ma_res)
    #mb_res = npma.array(mb_res)
    #N, M, K = np.shape(ma_res)
    #mask = np.zeros(np.shape(ma_res))
    #for i in range(N):
    #    for j in range(M):
    #        for k in range(K):
    #            if np.abs(ma_res[i, j, k]) + np.abs(mb_res[i, j, k]) < 1e-10:
    #                mask[i, j, k] = 1
    #                #ma_res[i, j, k] = np.inf
    #                #mb_res[i, j, k] = np.inf
    ma_res = npma.masked_array(ma_res, mask=mask)
    mb_res = npma.masked_array(mb_res, mask=mask)
    ax_global = pl.gca()
    ax_global.clear()
    pl.suptitle(title)
    pl.title(subtitle)
    ax_global.set_frame_on(False)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    #ax_global.set_xlabel([])
    #ax_global.set_ylabel([])
    ax_ma_abs = pl.axes([0.25, 0.15, 0.23, 0.3])
    ax_mb_abs = pl.axes([0.25, 0.55, 0.23, 0.3])
    ax_ma_angle = pl.axes([0.58, 0.15, 0.23, 0.3])
    ax_mb_angle = pl.axes([0.58, 0.55, 0.23, 0.3])
    ax_cb_left = pl.axes([0.10,  0.2, 0.03, 0.6])
    ax_cb_right = pl.axes([0.86, 0.2, 0.03, 0.6])
    #ma_res[mask > 0] = np.inf
    #mb_res[mask > 0] = np.inf
    for ax in [ax_ma_abs, ax_mb_abs, ax_ma_angle, ax_mb_angle]:
        ax.set_aspect('equal', 'box')

    m_scale = max(np.max(np.abs(ma_res)), np.max(np.abs(mb_res)))
    ma_pcolor = ax_ma_abs.pcolor(XR[:, :, 0] / constants.nm,
                                 YR[:, :, 0] / constants.nm,
                                 np.abs(ma_res[:, :, 0]) / m_scale,
                                 vmin = 0.0, vmax = 1.0)
    ax_ma_abs.set_xlabel("$x$, nm")
    ax_ma_abs.set_ylabel("$y$, nm")
    #pl.colorbar(location='left')
    dx = 1.2 * (XR[-1, 0, 0] - XR[0, 0, 0])
    dy = 1.2 * (YR[0, -1, 0] - YR[0, 0, 0])
    mb_pcolor = ax_mb_abs.pcolor(XR[:, :, 0] / constants.nm,
                                 YR[:, :, 0] / constants.nm + 0*dy,
                                 np.abs(mb_res[:, :, 0]) / m_scale,
                                 vmin = 0.0, vmax = 1.0)
    #pl.gca().set_aspect('equal', 'box')
    #pl.colorbar(location='left')
    cb = Colorbar(ax_cb_left, ma_pcolor, location='left')
    cb.set_label(r"$|m_{a, b}(x, y)|$")
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
    
    ma_angle_pc = ax_ma_angle.pcolor(XR[:, :, 0] / constants.nm + 0*dx,
                                     YR[:, :, 0] / constants.nm,
              np.angle(ma_res[:, :, 0]) * 180.0/np.pi,
              cmap='hsv', vmin=-180.0, vmax=180.0)
    mb_angle_pc = ax_mb_angle.pcolor(XR[:, :, 0] / constants.nm + 0 * dx,
                                     YR[:, :, 0] / constants.nm + 0 * dy,
              np.angle(mb_res[:, :, 0]) * 180.0/np.pi,
              cmap='hsv', vmin = -180.0, vmax=180.0)
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
    #cb = ax_global.colorbar(location='right')
    cb = Colorbar(ax_cb_right, ma_angle_pc, location='right')
    cb.set_ticks([-180.0, -90.0, 0, 90.0, 180.0])
    cb.set_label(r"${\rm arg} m_{a, b}(x, y)$")



def show_mode(X, Y, Z, ma_all, mb_all, z_list, title, subtitle):

    Z_min = np.min(Z)
    Z_max = np.max(Z)
    X_min = np.min(X)
    X_max = np.max(X)
    Y_min = np.min(Y)
    Y_max = np.max(Y)
    #print ("ma_film.shape: ", np.shape(ma_film))
    #print ("ma_res.shape: ", np.shape(ma_res))
    #print ("XF.shape = ", np.shape(XF))
    #print ("XR.shape = ", np.shape(XR))
    for z in z_list: 
        z_i  = Z[np.argmin(np.abs(Z - z))]
        i_z  = [t for t in range(len(Z)) if abs(Z[t] - z_i) < 1e-6]
        X_z  = np.array([X[t] for t in i_z])
        Y_z  = np.array([Y[t] for t in i_z])
        Z_z  = np.array([Z[t] for t in i_z])
        ma_z = np.array([ma_all[t] for t in i_z]) 
        mb_z = np.array([mb_all[t] for t in i_z])
        pl.figure()
        pl.suptitle(title)
        pl.title(subtitle + (" @ z= %g" % z_i))
        m_scale = max(np.max(np.abs(ma_z)), np.max(np.abs(mb_z)))
        pl.tripcolor(X_z, Y_z, np.abs(ma_z), vmin=0.0, vmax=m_scale)
        if z_i > 0:
            dx = 1.1 * (np.max(X_z) - np.min(X_z))
            dy = 0.0
        else:
            dy = 1.1 * (np.max(Y_z) - np.min(Y_z))
            dx = 0.0
        pl.tripcolor(X_z + dx, Y_z + dy,
                     np.abs(mb_z), vmin=0.0, vmax=m_scale)
        pl.gca().set_aspect('equal', 'box')
        pl.colorbar()
        #pl.title("z = %g f = %g + i %g" % (z_i, mode.f.real, mode.f.imag))
        pl.xlabel("x")
        pl.ylabel("y")



def show_data(fname):
    d = np.load(fname)
    for k in d.keys():
        print (k)

    mode_max = dict()

    for k in d.keys():
        key_pattern = 'mode_max_'
        if k[0:len(key_pattern)] == key_pattern:
            fields = k.split('_')
            print (k, fields)
            mode_no = int(fields[2])
            if mode_no not in mode_max.keys():
                mode_max[mode_no] = dict()
            field_name = '_'.join(fields[3:])
            mode_max[mode_no][field_name] = d[k]

    mode_max_copy = mode_max.copy()
    for mode_no, mode in mode_max_copy.items():
        mode_copy = mode.copy()
        for mode_key, mode_val in mode_copy.items():
            overlap_pattern = "overlap_"
            if mode_key[:len(overlap_pattern)] == overlap_pattern:
                overlap_fields = mode_key.split('_')
                if 'overlaps' not in mode.keys():
                    mode['overlaps'] = dict()
                mode['overlaps'][int(overlap_fields[1])] = mode_val

    for mode_no, mode in mode_max.items():
        print ("*** mode ", mode_no)
        print (mode)


    for mode_no, mode in mode_max.items():
        print ("*** mode ", mode_no, "overlaps")
        keys = list(mode['overlaps'].keys())
        keys.sort()
        for k in keys:
            print ("   with nrf mode", k, " = ", mode['overlaps'][k])

    mode_no_list = list(mode_max.keys())
    mode_no_list.sort()
    #for mode_no, mode in mode_max.items():
    for mode_no in mode_no_list:
        mode = mode_max[mode_no]
        print (list(mode.keys()))
        if 'ma_res' not in mode.keys():
            print ("mode ", mode_no, "has no ma keys")
            continue
        ma_res = mode['ma_res']
        mb_res = mode['mb_res']
        show_res_mode(d['XR'], d['YR'], d['ZR'], d['mask'], ma_res, mb_res,
                      "mode %d" % mode_no, "@ %g" % mode['f'].real)

    #for mode_no, mode in mode_max.items():
    for mode_no in mode_no_list:
        mode = mode_max[mode_no]
        print (list(mode.keys()))
        if 'ma_res' not in mode.keys():
            print ("mode ", mode_no, "has no ma keys")
            continue
        ma_all = mode['ma_all']
        mb_all= mode['mb_all']
        z_list = [np.min(d['Z']), np.max(d['Z'])]
        show_mode(d['X'], d['Y'], d['Z'], ma_all, mb_all, z_list,
                      "mode %d" % mode_no, "@ %g" % mode['f'].real)


for fname in sys.argv[1:]:
    show_data(fname)

pl.show()
