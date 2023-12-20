import numpy as np
import pylab as pl
import numpy.ma as npma

from modes3 import CellArrayModel, Material, Area, Grid
from modes3 import H_patch, VectorField, Mode
import constants
from nearfield import do_solveNearField

class ResonantMode3D:
    def __init__ (self, L, W, H, X, Y, Z, dV, omega,
                  ma, mb, mx, my, mz, theta_or = 0.0):
        self.omega = omega
        self.ma = ma
        self.mb = mb
        self.mx = mx
        self.my = my
        self.mz = mz
        self.X = X
        self.Y = Y
        self.Z = Z
        self.L = L
        self.W = W
        self.V = L * W * H
        self.H = H
        self.dV = dV
        self.theta_or = theta_or
        self.theta_cs = np.cos(self.theta_or)
        self.theta_sn = np.sin(self.theta_or) 

    def describe(self):
        return dict(
            res_name  = "res_mode @ " + str(self.omega.real / constants.GHz_2pi) + "GHz",
            res_V     = self.V,
            res_L     = self.L,
            res_H     = self.H,
            res_W     = self.W, 
            res_omega_0 = self.omega_0(),
            res_gamma_0 = self.gamma_0(),
            res_orientation = self.theta_or)
        
    def mk_mode(self, kx, ky, q, m_vals):
        exp_kr = np.exp( - q * self.Z - 1j * kx * self.X  - 1j * ky * self.Y)
        return np.sum(self.dV * m_vals * exp_kr) #* self.W

    def Mx(self, kx, ky, q):
        return self.mk_mode(kx, ky, q, self.mx)
    
    def My(self, kx, ky, q):
        return self.mk_mode(kx, ky, q, self.my)
    
    def Mz(self, kx, ky, q):
        return self.mk_mode(kx, ky, q, self.mz)
    
    def form_factors(self, kx, ky, q):
        #return 1.0, 1.0    
        #f_x = sinc(kx * self.L / 2.0)
        #f_y = 1.0
        #f_y = sinc(ky * self.W / 2.0)
        #f_z = sinch(q * self.H / 2.0)
        #f_xyz = f_x * f_y * f_z
        return 1.0, 1.0, 1.0 #f_y, f_y, f_y
        
    def K(self, kx, ky):
        #omega_x, omega_y = self.omega_xz()
        K_vals = []
        k = np.sqrt(kx**2 + ky**2)
        #k_hor = kx * self.theta_cs + ky * self.theta_sn
        kx_1 =   kx * self.theta_cs + ky * self.theta_sn
        ky_1 =  -kx * self.theta_sn + ky * self.theta_cs
        #F_x, F_y, F_z = self.form_factors(kx_1, ky_1, k)
        #Fk_hor = F_x * kx_1 + F_y * ky_1
        Mk_hor = self.Mx(kx_1, ky_1, k) * kx_1 + self.My(kx_1, ky_1, k) * ky_1
        return (1j * Mk_hor + k * self.Mz(kx_1, ky_1, k))

    def K_1D(self, k):
        kx_1 =   k
        ky_1 =   0.0
        #F_x, F_y, F_z = self.form_factors(kx_1, ky_1, np.abs(k))
        #print ("k = ", kx_1, ky_1, "F = ", F_x, F_y, F_z)
        #Fk_hor = F_x * k
        Mk_hor  = self.Mx(kx_1, ky_1, abs(k)) * kx_1
        Mk_hor += self.My(kx_1, ky_1, abs(k)) * ky_1
        wt = 1.0 / np.sqrt(self.W)
        #print ("mx: ", self.mx)
        return (1j * Mk_hor  + np.abs(k) * self.Mz(kx_1, np.abs(k))) * wt

    def freq_and_damping(self):
        return self.omega.real, -self.omega.imag

    def omega_0(self):
        omega_, gamma_ = self.freq_and_damping()
        return omega_

    def gamma_0(self):
        omega_, gamma_ = self.freq_and_damping()
        return gamma_    
        

def make_mode_collection_3d(model, res_mask, iomega, mab, I, f_min, f_max,
                            Xr, Yr, Zr, dV_r, Lr, Wr, Hr, theta_or):

    omega = 1j * iomega

    i_pos = [t for t in range(len(omega)) if omega[t].real > 0]
    i_pos.sort(key = lambda t: omega[t].real)

    modes = []

    res_to_global = dict()
    n_res = 0
    for i in range(len(res_mask)):
        if res_mask[i] > 0:
            res_to_global[n_res] = i
            n_res += 1

    def extend_to_global(res_field):
        out = np.zeros((len(res_mask)), dtype=complex)
        for i in range(len(res_field)):
            out[res_to_global[i]] = res_field[i]
        return out
    
    for i in i_pos:
        print ("mode freq: ", omega[i] / constants.GHz_2pi)
        if omega[i].real < f_min * constants.GHz_2pi: continue
        if omega[i].real > f_max * constants.GHz_2pi: continue
        mab_i = np.array(mab[:, i])
        C_norm = np.dot(mab_i.conj(), np.dot(I, mab_i))
        mab_i *= np.sqrt(4.0 / np.abs(C_norm))
        #mab_i *= np.sqrt(4.0 / np.sqrt(Wr))
        i_max = np.argmax(np.abs(mab_i))
        mab_i *= np.abs(mab_i[i_max]) / mab_i[i_max]
        ma_i = mab_i[0::2]
        mb_i = mab_i[1::2]
        ma_global_i = extend_to_global(ma_i)
        mb_global_i = extend_to_global(mb_i)
        #mx_i, my_i, mz_i = ma_i, 0.0 * ma_i, mb_i
        
        mxyz_global_i = model.ab_to_xyz(ma_global_i, mb_global_i)
        mx_i = mxyz_global_i.x[res_mask > 0]
        my_i = mxyz_global_i.y[res_mask > 0]
        mz_i = mxyz_global_i.z[res_mask > 0]
        theta_or = 0.0
        mode = ResonantMode3D(Lr, Wr, Hr, Xr, Yr, Zr, dV_r,
                                omega[i],              
                                ma_i, mb_i, mx_i, my_i, mz_i,
                                theta_or)
        print ("append: ", mode)
        modes.append(mode)

    print ("modes:", modes)
    return modes


def solveNearField_3d(model, f_min, f_max, Lr, Wr, Hr, theta_or, *areas):
    iomega, mxz, I_nrf, res_mask = do_solveNearField(model, *areas)
    Nr = len([t for t in range(len(res_mask)) if res_mask[t] > 0])
    Xr = np.zeros((Nr))
    Yr = np.zeros((Nr))
    Zr = np.zeros((Nr))
    Xr = model.box_array.X[res_mask > 0]
    Yr = model.box_array.Y[res_mask > 0]
    Zr = model.box_array.Z[res_mask > 0]
    dV_r = model.box_array.dV()[res_mask > 0]
    
    modes = make_mode_collection_3d(model, res_mask,
                                    iomega, mxz, I_nrf,
                                    f_min, f_max, Xr, Yr, Zr, dV_r,
                                    Lr, Wr, Hr, theta_or)

    return modes

def show_mode(model, result, mode, z_list):

    box = model.box_array
    XR, YR, ZR = result['coords']['resonator']
    N, M, K = np.shape(XR)
    if K < 2:
            XR_new = np.zeros((N, M, 2))
            YR_new = np.zeros((N, M, 2))
            ZR_new = np.zeros((N, M, 2))
            XR_new[:, :, 0] = XR[:, :, 0]
            XR_new[:, :, 1] = XR[:, :, 0]
            YR_new[:, :, 0] = YR[:, :, 0]
            YR_new[:, :, 1] = YR[:, :, 0]
            ZR_new[:, :, 0] = ZR[:, :, 0] - 0.005
            ZR_new[:, :, 1] = ZR[:, :, 0] + 0.005
            XR, YR, ZR = XR_new, YR_new, ZR_new
    
    if 'film' in result['coords'].keys():
        XF, YF, ZF = result['coords']['film']
        N, M, K = np.shape(XF)
        if K < 2:
            XF_new = np.zeros((N, M, 2))
            YF_new = np.zeros((N, M, 2))
            ZF_new = np.zeros((N, M, 2))
            XF_new[:, :, 0] = XF[:, :, 0]
            XF_new[:, :, 1] = XF[:, :, 0]
            YF_new[:, :, 0] = YF[:, :, 0]
            YF_new[:, :, 1] = YF[:, :, 0]
            ZF_new[:, :, 0] = ZF[:, :, 0] - 0.005
            ZF_new[:, :, 1] = ZF[:, :, 0] + 0.005
            XF, YF, ZF = XF_new, YF_new, ZF_new
    X, Y, Z = result['coords_all']
    mx, my, mz = mode.m('resonator')
    ma_res, mb_res = mode.m_ab("resonator")
    N, M, K = np.shape(ma_res)
    if K < 2:
        ma_res_new = np.zeros((N, M, 2), dtype=ma_res.dtype)
        mb_res_new = np.zeros((N, M, 2), dtype=mb_res.dtype)
        ma_res_new[:, :, 0] = ma_res[:, :, 0]
        ma_res_new[:, :, 1] = ma_res[:, :, 0]
        mb_res_new[:, :, 0] = mb_res[:, :, 0]
        mb_res_new[:, :, 1] = mb_res[:, :, 0]
        ma_res, mb_res = ma_res_new, mb_res_new
    if 'film' in result['coords'].keys():
       ma_film, mb_film = mode.m_ab("film")
       N, M, K = np.shape(ma_film)
       if K < 2:
           ma_film_new = np.zeros((N, M, 2), dtype=ma_film.dtype)
           mb_film_new = np.zeros((N, M, 2), dtype=mb_film.dtype)
           ma_film_new[:, :, 0] = ma_film[:, :, 0]
           ma_film_new[:, :, 1] = ma_film[:, :, 0]
           mb_film_new[:, :, 0] = mb_film[:, :, 0]
           mb_film_new[:, :, 1] = mb_film[:, :, 0]
           ma_film, mb_film = ma_film_new, mb_film_new

    ma_all, mb_all = mode.m_ab_all()
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
        pl.title("z = %g f = %g + i %g" % (z_i, mode.f.real, mode.f.imag))
        pl.xlabel("x")
        pl.ylabel("y")

    for x in [0.0]:
        if 'film' not in result['coords'].keys(): break
        x_i  = X[np.argmin(np.abs(X - x))]
        i_x  = [t for t in range(len(X)) if abs(X[t] - x_i) < 1e-6]
        i_f = np.argmin(np.abs(XF[:, 0, 0] - x_i))
        i_r = np.argmin(np.abs(XR[:, 0, 0] - x_i))
        #print ("XF = ", XF)
        #i_xf = [t for t in range(len(XF)) if abs(XF[t] - x_i) < 1e-6]
        #i_xr = [t for t in range(len(XR)) if abs(XR[t] - x_i) < 1e-6]
        X_x  = np.array([X[t] for t in i_x])
        Y_x  = np.array([Y[t] for t in i_x])
        Z_x  = np.array([Z[t] for t in i_x])
        ma_x = np.array([ma_all[t] for t in i_x]) 
        mb_x = np.array([ma_all[t] for t in i_x])
        pl.figure()
        m_scale = max(np.max(np.abs(ma_x)), np.max(np.abs(mb_x)))
        pl.pcolormesh(YF[i_f, :, :], ZF[i_f, :, :],
                     np.abs(ma_film[i_f, :, :]), vmin=0.0, vmax=m_scale)
        pl.pcolor(YR[i_r, :, :], ZR[i_r, :, :],
                  np.abs(ma_res[i_r, :, :]), vmin=0.0, vmax=m_scale)
        #if z_i > 0:
        #    dx = 1.1 * (np.max(X) - np.min(X))
        #    dy = 0.0
        #else:
        #dz = 1.2 * (d_film + s + c)
        dz = 1.2 * (np.max(Z) - np.min(Z))
        dy = 0.0
        pl.pcolormesh(YF[i_f, :, :] + dy, ZF[i_f, :, :] + dz,
                     np.abs(mb_film[i_f, :, :]), vmin=0.0, vmax=m_scale)
        pl.pcolor(YR[i_r, :, :] + dy, ZR[i_r, :, :] + dz,
                  np.abs(mb_res[i_r, :, :]), vmin=0.0, vmax=m_scale)
        pl.gca().set_aspect('equal', 'box')
        pl.colorbar()
        pl.title("x = %g f = %g + i %g" % (x_i, mode.f.real, mode.f.imag))
        pl.xlabel("y")
        pl.ylabel("z")

    for y in [0.0]:
        if 'film' not in result['coords'].keys(): break
        y_i  = Y[np.argmin(np.abs(Y - y))]
        i_y  = [t for t in range(len(Y)) if abs(Y[t] - y_i) < 1e-6]
        j_f = np.argmin(np.abs(XF[0, :, 0] - y_i))
        j_r = np.argmin(np.abs(XR[0, :, 0] - y_i))
        X_y  = np.array([X[t] for t in i_y])
        Y_y  = np.array([Y[t] for t in i_y])
        Z_y  = np.array([Z[t] for t in i_y])
        ma_y = np.array([ma_all[t] for t in i_y]) 
        mb_y = np.array([ma_all[t] for t in i_y])
        pl.figure()
        m_scale = max(np.max(np.abs(ma_y)), np.max(np.abs(mb_y)))
        #pl.tripcolor(X_y, Z_y, np.abs(ma_y), vmin=0.0, vmax=m_scale)
        pl.pcolormesh(XF[:, j_f, :], ZF[:, j_f, :],
                     np.abs(ma_film[:, j_f, :]), vmin=0.0, vmax=m_scale)
        pl.pcolor(XR[:, j_r, :], ZR[:, j_r, :],
                  np.abs(ma_res[:, j_r, :]), vmin=0.0, vmax=m_scale)
        #if z_i > 0:
        #    dx = 1.1 * (np.max(X) - np.min(X))
        #    dy = 0.0
        #else:
        #dz = 1.1 * (np.max(Z) - np.min(Z))
        dx = 0.0
        #dz = 1.2 * (d_film + s + c)
        dz = 1.2 * (np.max(Z) - np.min(Z))
        pl.pcolormesh(XF[:, j_f, :] + dx, ZF[:, j_f, :] + dz,
                     np.abs(mb_film[:, j_f, :]), vmin=0.0, vmax=m_scale)
        pl.pcolor(XR[:, j_r, :] + dx, ZR[:, j_r, :] + dz,
                  np.abs(mb_res[:, j_r, :]), vmin=0.0, vmax=m_scale)
        #pl.tripcolor(X_y + dx, Z_y + dz,
        #             np.abs(mb_y), vmin=0.0, vmax=m_scale)
        pl.gca().set_aspect('equal', 'box')
        pl.colorbar()
        pl.title("y = %g f = %g + i %g" % (y_i, mode.f.real, mode.f.imag))
        pl.xlabel("x")
        pl.ylabel("z")

    pl.show()    


def show_res_mode(model, XR, YR, ZR, mode, mask, title, subtitle):   
    pl.figure()
    from matplotlib.colorbar import Colorbar
    ma_res, mb_res = mode.m_ab('resonator')
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


def show_static_state(model, X, Y, Z, n, H, z_list):
    for z in z_list: #[s + c/2, 0.0]:
        z_i = Z[np.argmin(np.abs(Z - z))]
        i_z = [t for t in range(len(Z)) if abs(Z[t] - z_i) < 1e-6]
        X_z = np.array([X[t] for t in i_z])
        Y_z = np.array([Y[t] for t in i_z])
        Z_z = np.array([Z[t] for t in i_z])
        n_x = np.array([n.x[t] for t in i_z])
        n_y = np.array([n.y[t] for t in i_z])
        n_z = np.array([n.z[t] for t in i_z])
        H_x = np.array([H.x[t] for t in i_z])
        H_y = np.array([H.y[t] for t in i_z])
        H_z = np.array([H.z[t] for t in i_z])
        if False:
           H_inf_arr = [H_inf(X[t], Y[t], Z[t]) for t in i_z]
           H_inf_x = np.array([t[0] for t in H_inf_arr])
           H_inf_y = np.array([t[1] for t in H_inf_arr])
           H_inf_z = np.array([t[2] for t in H_inf_arr])


        pl.figure()
        pl.quiver(X_z, Y_z, n_x, n_y, pivot='mid')
        pl.gca().set_aspect('equal', 'box')
        pl.xlabel(r"$x$")
        pl.ylabel(r"$y$")
        pl.title(r"${\bf n}(x, y, z = %g)$" % z_i)

        pl.figure()
        pl.quiver(X_z, Y_z, n_x, n_y - 1.0, pivot='mid')
        pl.gca().set_aspect('equal', 'box')
        pl.xlabel(r"$x$")
        pl.ylabel(r"$y$")
        pl.title(r"${\bf n}(x, y, z = %g) - {\bf n}_\infty$" % z_i)

        pl.figure()
        pl.tripcolor(X_z, Y_z, n_x, cmap='bwr', vmin=-0.2, vmax=0.2)
        pl.gca().set_aspect('equal', 'box')
        pl.colorbar()
        pl.xlabel(r"$x$")
        pl.ylabel(r"$y$")
        pl.title(r"$n_x$" % z_i)

        pl.figure()
        pl.tripcolor(X_z, Y_z, n_z, cmap='bwr', vmin=-0.2, vmax=0.2)
        pl.gca().set_aspect('equal', 'box')
        pl.colorbar()
        pl.xlabel(r"$x$")
        pl.ylabel(r"$y$")
        pl.colorbar()
        pl.title(r"$n_z$" % z_i)

        pl.figure()
        pl.tripcolor(X_z, Y_z, 1.0 - n_y, cmap='magma', vmin=0.0, vmax=0.03)
        pl.gca().set_aspect('equal', 'box')
        pl.colorbar()
        pl.xlabel(r"$x$")
        pl.ylabel(r"$y$")
        pl.title(r"$1.0 - n_y$" % z_i)

        pl.figure()
        pl.quiver(X_z, Y_z, H_x, H_y, pivot='mid')
        #pl.quiver(X_z, Y_z, H_inf_x, H_inf_y,
        #          color='blue', pivot='mid')
        #pl.quiver(X_z, Y_z, H_x + H_inf_x, H_y + H_inf_y,
        #          color='red', pivot='mid')
        pl.gca().set_aspect('equal', 'box')
        pl.xlabel(r"$x$")
        pl.ylabel(r"$y$")
        pl.title(r"${\bf H}(x, y, z = %g)$" % z_i)


        if False:
            pl.figure()
            pl.quiver(X_z, Y_z, H_inf_x, H_inf_y,
                      color='blue', pivot='mid')
            #pl.quiver(X_z, Y_z, H_x + H_inf_x, H_y + H_inf_y,
            #          color='red', pivot='mid')
            pl.gca().set_aspect('equal', 'box')
            pl.xlabel(r"$x$")
            pl.ylabel(r"$y$")
            pl.title(r"${\bf H}_\infty(x, y, z = %g)$" % z_i)

            pl.figure()
            #pl.quiver(X_z, Y_z, H_inf_x, H_inf_y,
            #          color='blue', pivot='mid')
            pl.quiver(X_z, Y_z, H_x + H_inf_x, H_y + H_inf_y,
                      color='red', pivot='mid')
            pl.gca().set_aspect('equal', 'box')
            pl.xlabel(r"$x$")
            pl.ylabel(r"$y$")
            pl.title(r"${\bf H} + {\bf H}_\inf(x, y, z = %g)$" % z_i)

        if False:
            pl.figure()
            pl.tripcolor(X_z, Y_z, np.abs(H_x)**2 + np.abs(H_y)**2
                         + np.abs(H_z)**2)
            pl.gca().set_aspect('equal', 'box')
            pl.colorbar()

            pl.figure()
            pl.tripcolor(X_z, Y_z, np.abs(H_inf_x)**2 + np.abs(H_inf_y)**2
                         + np.abs(H_inf_z)**2)
            pl.gca().set_aspect('equal', 'box')
            pl.colorbar()
            pl.title("")

            pl.figure()
            pl.tripcolor(X_z, Y_z, np.abs(H_inf_x + H_x)**2
                         + np.abs(H_inf_y + H_y)**2
                         + np.abs(H_inf_z + H_z)**2)
            pl.gca().set_aspect('equal', 'box')
            pl.colorbar()
            pl.title("sum")

    pl.show()


def show_nrf_mode(model, mode, z_list):
    from matplotlib.colorbar import Colorbar
    f = mode.omega / constants.GHz_2pi
    print ("mode: f = ", f)
    X = mode.X
    Y = mode.Y
    Z = mode.Z
    ma = mode.mx
    mb = mode.mz
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
                                    np.abs(ma_z) / m_scale, vmin=0.0, vmax=1.0)
        ax_ma_abs.set_xlabel("$x$, nm")
        ax_ma_abs.set_ylabel("$y$, nm")
        mb_abs_pc = ax_mb_abs.tripcolor(X_z / constants.nm,
                                        Y_z / constants.nm + 0 * dy,
                     np.abs(mb_z) / m_scale, vmin=0.0, vmax=1.0)
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
    pl.show()

def simulate_3d():
    Py_Ms = 800 * constants.kA_m
    Aex_Py = 13 * constants.pJ_m * 2
    Jex_Py = Aex_Py / Py_Ms**2
    alpha_Py = 5e-3
    Py_gamma_s = constants.gamma_s
    
    YIG_Ms = 140 * constants.kA_m
    YIG_alpha = 1 * 0.001
    YIG_gamma_s = constants.gamma_s
    Aex = 2 * 3.5 * constants.pJ_m
    YIG_Jex = Aex / YIG_Ms**2

    YIG = Material("YIG", YIG_Ms, YIG_Jex, YIG_alpha, YIG_gamma_s)
    Py  = Material("Py", Py_Ms, Jex_Py, alpha_Py, Py_gamma_s)

    Bext = 70 * constants.mT

    a = 70 * constants.nm
    b = 60 * constants.nm
    c = 10 * constants.nm

    #Na, Nb, Nc = 32, 32, 3
    Na, Nb, Nc = 36, 32, 2
    #Na, Nb, Nc = 12, 12, 2 #16, 16, 2
    #Na = 6
    #Nb = 6
    #Nc = 3

    W  = 300 * constants.nm
    Nw = 48 #Nw = 16
    #Nw = 16 #24
    d_film = 20 * constants.nm
    #Nd = 3
    Nd = 2
    s = 20 * constants.nm

    def mask_func_res(x, y, z):
        return ((2*x/a)**2 + (2*y/b)**2 < 1)

    resonator = Area("resonator", Grid(-a/2, a/2, -b/2, b/2, s, s + c,
                                       Na, Nb, Nc), Py, mask_func_res)

    film = Area("film", Grid(-W/2, W/2, -W/2, W/2, -d_film, 0,
                             Nw, Nw, Nd), YIG)
    model = CellArrayModel()

    def H_inf(x, y, z):
        R_n = np.array([0.0, W/2, -d_film/2])
        R_s = np.array([0.0, -W/2, -d_film/2])
        a_n = np.array([W, 0, 0])
        b_n = np.array([0, 0, d_film])
        a_s =  a_n
        b_s =  b_n

        R   = np.array([x, y, z])
        
        H_n = H_patch(a_n, b_n, R - R_n) 
        H_s = H_patch(a_s, b_s, R - R_s)
        return (H_n - H_s) * YIG_Ms * constants.mu_0 / 4.0 / np.pi
        
    def B_ext_x(x, y, z):
        return 0.0  + 1 * H_inf(x, y, z)[0]
    def B_ext_y(x, y, z):
        return Bext + 1 * H_inf(x, y, z)[1]
    def B_ext_z(x, y, z):
        return 0.0  + 1 * H_inf(x, y, z)[2]
    
    model.add_area(resonator, B_ext_x, B_ext_y, B_ext_z)
    model.add_area(film, B_ext_x, B_ext_y, B_ext_z)
    n = model.relax_magnetisation()
    X = model.box_array.X
    Y = model.box_array.Y
    Z = model.box_array.Z
    #n = VectorField(0.0 * X, 1.0 + 0.0 * Y, 0.0 * Z)
    H = model.compute_static_field(n)
    E = model.compute_static_energy(n)
    model.setup_dreibein(n)

    show_static_state(model, X, Y, Z, n, H, [s + c/2, -d_film/2])

    if  True:
        nrf_mode_collection = solveNearField_3d(model,
                            0.5 * constants.GHz, 15 * constants.GHz,
                            a, b, c, 0.0, 'resonator')
        for mode_num, mode in enumerate(nrf_mode_collection):
            #show_nrf_mode(model, mode,  [s + c/2, -d_film/2])
            show_nrf_mode(model, mode,  [s + c/2])
            fname_mode = "NRF3d-mode-%d" %  mode_num
            fname_sz="%gx%gx%gnm" % (a / constants.nm, b/constants.nm,
                                     c / constants.nm)
            fname_field="%gmT"    % (Bext / constants.mT)
            fname_film="s=%gnm-d=%gnm" % (s / constants.nm,
                                          d_film/constants.nm)
            fname_grid="%dx%dx%d" % (Na, Nb, Nc)
            fname = "%s-%s-%s-%s-%s" % (fname_mode, fname_sz, fname_field,
                                     fname_film, fname_grid)
            np.savez(fname,
                     a=a, b=b, c=c, Na=Na, Nb=Nb, Nc=Nc,
                     n_x = n.x, n_y = n.y, n_z = n.z,
                     s=s, d_film = d_film,
                     Bext=Bext,
                     omega = mode.omega, 
                     X=mode.X, Y=mode.Y, Z=mode.Z, dV = mode.dV,
                     ma=mode.ma, mb=mode.mb,
                     mx=mode.mx, my=mode.my, mz=mode.mz)
            
    if  True:
        result = model.solve()
        box = model.box_array
        XR, YR, ZR = result['coords']['resonator']
        if 'film' in result['coords'].keys():
           XF, YF, ZF = result['coords']['film']
        X, Y, Z = result['coords_all']

        mode_set = []

        ones = np.ones((len(X)))
        volume_tot = model.integrate(ones)
        volume_res = model.integrate(ones, "resonator")

        volume_ratio = volume_res / volume_tot
        
        
        for mode in result['modes']:
            print ("mode @ ", mode.f)
            #show_mode(model, result, mode, [s + c/2, -d_film/2])
            if mode.f.real > 15.0: break
            mode_set.append(mode)

        from clusterize import Clusterizer
        clusterizer = Clusterizer(model, ["resonator"], ["film"])
        clusterizer.compute_mode_overlap(mode_set)
        clusterizer.clusterize_modes()
        cluster_results = clusterizer.analyze_clusters()

        def get_p_max_no(cluster_data):
            return np.argmax(np.abs(cluster_data[1]))
        def get_f_max(cluster_data):
            return cluster_data[0][get_p_max_no(cluster_data)].real

        clusters_sorted = list(enumerate(cluster_results))
        clusters_sorted.sort(key = lambda x: get_f_max (x[1]))

        pl.figure()
        modes_to_show = []
        for cluster, c_result in clusters_sorted: #enumerate(cluster_results):
            f_vals, p_vals, C_cluster, S_cluster = c_result
            max_p_in_cluster = np.argmax(np.abs(p_vals))
            f_max = f_vals[max_p_in_cluster]
            mode_max_no = clusterizer.clusters[cluster][max_p_in_cluster]
            p_max = p_vals[max_p_in_cluster]
            modes_to_show.append((mode_max_no, f_max, p_max / volume_ratio))
            f_axis = np.linspace(np.min(f_vals.real), np.max(f_vals.real),
                                 2000)
            p_axis = 0.0 * f_axis
            gamma = 1e-3
            for f_val, p_val in zip(f_vals, p_vals):
                p_cur = p_val / volume_ratio
                f_cur = f_val.real
                column = (gamma**2 / (gamma**2 + (f_axis - f_cur)**2))**2
                p_axis += column * p_cur
            p = pl.plot(f_vals.real, p_vals  / volume_ratio, '-o', ms=5.0,
                        label='cluster @ %g' % f_max.real)
            #p = pl.plot(f_vals.real, p_vals  / volume_ratio, '-')
            #pl.plot(f_axis, p_axis, '-o', ms=1.0,
            #        label='cluster @ %g' % f_max.real, 
            #        color=p[0].get_color())
        for nrf_mode in nrf_mode_collection:
            nrf_f = nrf_mode.omega / constants.GHz_2pi
            pl.plot([nrf_f, nrf_f], [0.0, 10.0], 'k--')
        pl.legend()
        pl.xlabel(r"Frequency $f$, GHz")
        pl.ylabel(r"Mode participation")
        #pl.show()
        print ("modes_to_show:", modes_to_show)

        XR, YR, ZR = result['coords']['resonator']
        Nm, Mm, Km = np.shape(XR)
        mask = 0.0 + 0 * XR
        for i in range(Nm):
            for j in range(Mm):
                for k in range(Km):
                    if not resonator.belongs(i, j, k): mask[i, j, k] = 1
        for mode_no, f_max, p_max in modes_to_show:
            title = "mode #%d @ %f GHz" % (mode_no, f_max.real)
            subtitle = "participation = %g" % (p_max)
            if p_max <  0.1: continue
            show_res_mode(model, XR, YR, ZR,
                          clusterizer.mode_norm[mode_no],
                          mask, 
                          title, subtitle)
        for mode_no, f_max, p_max in modes_to_show:
            mode = clusterizer.mode_norm[mode_no].copy()
            mode_ma, mode_mb = mode.m_ab_all()
            for nrf_mode in nrf_mode_collection:
                mode_ma = mode_ma[:len(nrf_mode.ma)]
                mode_mb = mode_mb[:len(nrf_mode.mb)]
                #nrf_mab = np.zeros((2 * len(nrf_mode.ma)),
                #                   dtype=nrf_mode.ma.dtype)
                #nrf_mab[0::2] = nrf_mode.ma
                #nrf_mab[1::2] = nrf_mode.mb
                nrf_f = nrf_mode.omega / constants.GHz_2pi
                #nrf_mode_3d = Mode(model, nrf_f, nrf_mab)
                #nrf_mode_3d.normalize()
                #mode.normalize()
                #overlap = model.dot_product(mode, nrf_mode_3d)
                dV = nrf_mode.dV
                def dot_prod(mode1_a, mode1_b, mode2_a, mode2_b):
                    S  = np.sum(mode1_a.conj() * mode2_b * dV)
                    S -= np.sum(mode1_b.conj() * mode2_a * dV)
                    return S
                C_mode = dot_prod(mode_ma, mode_mb, mode_ma, mode_mb)
                C_nrf  = dot_prod(nrf_mode.ma, nrf_mode.mb,
                                  nrf_mode.ma, nrf_mode.mb)
                C_over = dot_prod(mode_ma, mode_mb, nrf_mode.ma, nrf_mode.mb)
                overlap = np.abs(C_over / np.sqrt(np.abs(C_mode * C_nrf)))
                print ("mode @ %g GHz: "
                       "overlap with nrf mode @ nrf_mode % g = %g"
                       % (mode.f, nrf_f, overlap))
                #nrf_mode.ma
                #mode_ma, mode_mb = mode.m_ab('resonator')

                
                
        pl.show()
        
simulate_3d()
    

    
