import numpy as np
import pylab as pl
from scipy import linalg

import constants

# Field induced by a vertical segment (x_a, z_a) -- (x_a, z_b)
# with unit charge density at the point (x_o, z_o)
def H_xx(x_o, z_o, x_a, z_a, z_b):
    #dx_oa = x_o - x_a
    #atan_d = np.arctan((z_o - z_b)/dx_oa) - np.arctan((z_o - z_a)/dx_oa)
    #atan_d =    np.angle( 1j * (x_o - x_a + 1j * (z_o - z_a))) - np.pi/2.0
    #atan_d +=   np.angle( 1j * (x_o - x_a + 1j * (z_o - z_b))) + np.pi/2.0
    atan_d = np.angle((x_o - x_a + 1j * (z_o - z_a))/(x_o - x_a + 1j * (z_o - z_b)))
    return atan_d

def H_zx(x_o, z_o, x_a, z_a, z_b):
    log_d  = np.log(np.abs((x_a - x_o) + 1j * (z_b - z_o)))
    log_d -= np.log(np.abs((x_a - x_o) + 1j * (z_a - z_o)))
    return -log_d

# Field induced by a horizontal segment (x_a, z_a) -- (x_b, z_a)
# with unit charge density at the point (x_o, z_o)
def H_xz(x_o, z_o, x_a, x_b, z_a):
    log_d  = np.log(np.abs((x_b - x_o) + 1j * (z_a - z_o)))
    log_d -= np.log(np.abs((x_a - x_o) + 1j * (z_a - z_o)))
    return -log_d

def H_zz(x_o, z_o, x_a, x_b, z_a):
    dx_oa = x_o - x_a
    dx_ob = x_o - x_b
    #atan_d = np.arctan(dx_ob/(z_o - z_a)) - np.arctan(dx_oa/(z_o - z_a))
    atan_d = np.angle((-1j * (x_o - x_b) + (z_o - z_a))/(-1j * (x_o - x_a) +  (z_o - z_a)))
    #atan_d  =   np.angle((x_o - x_a + 1j * (z_o - z_a))) 
    #atan_d += - np.angle(-(x_o - x_b + 1j * (z_o - z_a))) + np.pi 
    return  atan_d

def modes_big(Bext, Ms, Jex, a, b, Nrx, Nrz, s, W, d, Nsx, Nsz):
    Nr = (Nrx - 1) * (Nrz - 1)
    Ns = (Nsx - 1) * (Nsz - 1)
    N = Ns + Nr
    Hxx = np.zeros((N, N))
    Hxz = np.zeros((N, N))
    Hzx = np.zeros((N, N))
    Hzz = np.zeros((N, N))
    xc_idx = np.zeros((N))
    zc_idx = np.zeros((N))
    xl_idx = np.zeros((N))
    xr_idx = np.zeros((N))
    zt_idx = np.zeros((N))
    zb_idx = np.zeros((N))
    mask_r = np.zeros((N))
    mask_s = np.zeros((N))
    def ij_r_idx(i, j):
        return i * (Nrz - 1) + j
    def ij_s_idx(i, j):
        return Nr + i * (Nsz - 1) + j

    xr = np.linspace(-a/2, a/2, Nrx)
    zr = np.linspace(   0, b, Nrz)

    xs = np.linspace(-W/2, W/2, Nsx)
    zs = np.linspace(-d - s, -s, Nsz)
    xrm = 0.5 * (xr[1:] + xr[:-1])
    zrm = 0.5 * (zr[1:] + zr[:-1])
    xsm = 0.5 * (xs[1:] + xs[:-1])
    zsm = 0.5 * (zs[1:] + zs[:-1])

    i_r = np.zeros((N))
    j_r = np.zeros((N))
    i_s = np.zeros((N))
    j_s = np.zeros((N))
    i_r[:] = -1
    j_r[:] = -1
    i_s[:] = -1
    j_s[:] = -1
     
    for i in range(0, Nrx - 1):
        for j in range(0, Nrz - 1):
            idx_ij = ij_r_idx(i, j)
            i_r[idx_ij] = i
            j_r[idx_ij] = j
            xc_idx[idx_ij] = xrm[i]
            zc_idx[idx_ij] = zrm[j]
            xl_idx[idx_ij] = xr[i]
            xr_idx[idx_ij] = xr[i + 1]
            zb_idx[idx_ij] = zr[j]
            zt_idx[idx_ij] = zr[j + 1]
            mask_r[idx_ij] = 1.0
            
    for i in range(0, Nsx - 1):
        for j in range(0, Nsz - 1):
            idx_ij = ij_s_idx(i, j)
            i_s[idx_ij] = i
            j_s[idx_ij] = j
            xc_idx[idx_ij] = xsm[i]
            zc_idx[idx_ij] = zsm[j]
            xl_idx[idx_ij] = xs[i]
            xr_idx[idx_ij] = xs[i + 1]
            zb_idx[idx_ij] = zs[j]
            zt_idx[idx_ij] = zs[j + 1]
            mask_s[idx_ij] = 1.0

    dx_idx = xr_idx - xl_idx
    dz_idx = zt_idx - zb_idx
            
    for idx1 in range(N):
            xc = xc_idx[idx1]
            zc = zc_idx[idx1]
            xl = xl_idx[idx1]
            xr = xr_idx[idx1]
            zb = zb_idx[idx1]
            zt = zt_idx[idx1]
            
            Hxx[:, idx1] += H_xx(xc_idx, zc_idx, xr, zb, zt)
            Hxx[:, idx1] -= H_xx(xc_idx, zc_idx, xl, zb, zt)
            Hzx[:, idx1] += H_zx(xc_idx, zc_idx, xr, zb, zt)
            Hzx[:, idx1] -= H_zx(xc_idx, zc_idx, xl, zb, zt)
            Hxz[:, idx1] += H_xz(xc_idx, zc_idx, xl, xr, zt)
            Hxz[:, idx1] -= H_xz(xc_idx, zc_idx, xl, xr, zb)
            Hzz[:, idx1] += H_zz(xc_idx, zc_idx, xl, xr, zt)
            Hzz[:, idx1] -= H_zz(xc_idx, zc_idx, xl, xr, zb)

    Hxx /= 2.0 * np.pi 
    Hxz /= 2.0 * np.pi 
    Hzx /= 2.0 * np.pi 
    Hzz /= 2.0 * np.pi 

    J = np.zeros((N, N))
    
    for i in range(Nrx - 2):
        for j in range(Nrz - 1):
            idx_w = ij_r_idx(i, j)
            idx_e = ij_r_idx(i + 1, j    )
            dx_inv2 = 1.0 / (xrm[i + 1] - xrm[i])**2 * Jex
            J[idx_e, idx_e] +=  dx_inv2
            J[idx_w, idx_e] += -dx_inv2
            J[idx_e, idx_w] += -dx_inv2
            J[idx_w, idx_w] +=  dx_inv2
            
    for i in range(Nrx - 1):
        for j in range(Nrz - 2):
            idx_s = ij_r_idx(i, j)
            idx_n = ij_r_idx(i, j + 1) 
            dz_inv2 = 1.0 / (zrm[j + 1] - zrm[j])**2 * Jex
            J[idx_s, idx_s] +=  dz_inv2
            J[idx_n, idx_s] += -dz_inv2
            J[idx_s, idx_n] += -dz_inv2
            J[idx_n, idx_n] +=  dz_inv2
    
    for i in range(Nsx - 2):
        for j in range(Nsz - 1):
            idx_w = ij_s_idx(i, j)
            idx_e = ij_s_idx(i + 1, j    )
            dx_inv2 = 1.0 / (xsm[i + 1] - xsm[i])**2 * Jex
            J[idx_e, idx_e] +=  dx_inv2
            J[idx_w, idx_e] += -dx_inv2
            J[idx_e, idx_w] += -dx_inv2
            J[idx_w, idx_w] +=  dx_inv2
            
    for i in range(Nsx - 1):
        for j in range(Nsz - 2):
            idx_s = ij_s_idx(i, j)
            idx_n = ij_s_idx(i, j + 1)
            dz_inv2 = 1.0 / (zsm[j + 1] - zsm[j])**2 * Jex
            J[idx_s, idx_s] +=  dz_inv2 
            J[idx_n, idx_s] += -dz_inv2 
            J[idx_s, idx_n] += -dz_inv2 
            J[idx_n, idx_n] +=  dz_inv2
    
    L = np.zeros((2 * N, 2 * N))
    I = np.eye(N)
    L[0::2, 1::2] += Bext * I + Ms * J 
    L[1::2, 0::2] += - Bext * I - Ms * J
    L[0::2, 0::2] += -  Ms * constants.mu_0 * Hzx 
    L[0::2, 1::2] += -  Ms * constants.mu_0 * Hzz 
    L[1::2, 0::2] += +  Ms * constants.mu_0 * Hxx 
    L[1::2, 1::2] += +  Ms * constants.mu_0 * Hxz
    L *= constants.gamma_s / constants.GHz_2pi

    iomega, mxz = linalg.eig(L)
    
    print ("eigenvals: ", iomega)
    print ("dangerous: ", [t for t in iomega if abs(t.real) > 1e-10])

    om = iomega.imag
    #pl.figure()
    #pl.hist(om, bins=np.linspace(-10.0, 10.0, 201)) 

    def show_mode(i_mode):
        mxz_i = mxz[:, i_mode]
        print ("mode ", i_mode, "o = ", iomega[i_mode], "check: ",
               linalg.norm(np.dot(L, mxz_i) - iomega[i_mode] * mxz_i))
        mx = mxz_i[0::2]
        mz = mxz_i[1::2]
        C = np.sum(mz.conjugate() * mx * dx_idx * dz_idx).imag
        mx /= np.sqrt(abs(C))
        mz /= np.sqrt(abs(C))

        C_r =  np.sum(mz.conjugate() * mx * dx_idx * dz_idx * mask_r).imag
        V_r = np.sum(dx_idx * dz_idx * mask_r)
        V = np.sum(dx_idx * dz_idx)
        print ("participation: ", C_r, "thershold: ", V_r/V)
        if abs(C_r) < V_r / V: return

        #mx = np.zeros((Nx - 1, Nz - 1), dtype=complex)
        #mz = np.zeros((Nx - 1, Nz - 1), dtype=complex)
        #for i in range(Nx - 1):
        #    for j in range(Nz - 1):
        #        mx[i, j] = mx_flat[ij_to_idx(i, j)]
        #        mz[i, j] = mz_flat[ij_to_idx(i, j)]
        #print ("mx")
        #print (mx)
        #print ("mz")
        #print (mz)
        pl.figure()
        mxmax = np.max(np.abs(mx.real))
        mzmax = np.max(np.abs(mz.real))
        #pl.pcolormesh(X, Z, mx.real, cmap='bwr', vmin=-mxmax, vmax=mxmax)
        #pl.gca().set_aspect('equal', 'box')
        #pl.colorbar()
        #pl.title("mx for o = %g %g" % (iomega[i_mode].real, iomega[i_mode].imag))
        #pl.figure()
        #pl.pcolormesh(X, Z, mz.real, cmap='bwr', vmin=-mzmax, vmax=mzmax)
        #pl.gca().set_aspect('equal', 'box')
        #pl.colorbar()
        #pl.title("mz for o = %g %g"% (iomega[i_mode].real, iomega[i_mode].imag))
        dx = xrm[1] - xrm[0]
        dz = zrm[1] - zrm[0]
        mx_max = max(np.max(np.abs(mx.real)), np.max(np.abs(mx.imag)))
        mz_max = max(np.max(np.abs(mz.real)), np.max(np.abs(mz.imag)))
        q_scale = 0.8 * min(abs(dx), abs(dz)) / max(mx_max, mz_max)
        q_scale = 1.0/q_scale
        pl.figure()
        pl.quiver(xc_idx, zc_idx,
                  mx.real, mz.real, scale=q_scale, scale_units='x',
                  pivot='middle', color='red')
        pl.gca().set_aspect('equal', 'box')
        pl.title("Re mode %d o = %g %g" % (i_mode, iomega[i_mode].real,
                                        iomega[i_mode].imag))

        #pl.figure()
        pl.quiver(xc_idx, zc_idx,
                  mx.imag, mz.imag, scale=q_scale, scale_units='x',
                  pivot='middle',
                  color='blue')
        pl.gca().set_aspect('equal', 'box')
        pl.title("Im mode %d o = %g %g C_r = %g" % (i_mode,
                                                    iomega[i_mode].real,
                                                    iomega[i_mode].imag, C_r))

    modes = list(enumerate(iomega))
    modes_pos = list([t for t in modes if (t[1].imag > 0)])
    modes_neg = list([t for t in modes if (t[1].imag <= 0)])
    #modes_neg.sort(key = lambda t: -t[1].imag)
    modes_pos.sort(key = lambda t:  t[1].imag)
    omega_vals = []
    part_vals = []
    modes_mx_sv = []
    modes_mz_sv = []
    modes_o_sv = []
    for i_mod, omega_mod in modes_pos:
        mxz_mod = mxz[:, i_mod]
        mx_mod = mxz_mod[ ::2]
        mz_mod = mxz_mod[1::2]
        dV = dx_idx * dz_idx
        C = np.sum(mz_mod.conjugate() * mx_mod * dV).imag
        C_res = np.sum(mz_mod.conjugate() * mx_mod * mask_r * dV).imag
        part_ratio = np.abs(C_res/C)
        omega_vals.append(omega_mod)
        part_vals.append(part_ratio)
        V = np.sum(dV)
        V_res = np.sum(dV * mask_r)
        if part_ratio > V_res / V and abs(omega_mod) < 5.0 * constants.GHz_2pi:
            modes_o_sv.append(omega_mod)
            modes_mx_sv.append(mx_mod)
            modes_mz_sv.append(mz_mod)

    np.savez("modes-de-s=%g-a=%g b=%g-d=%g" % (s / constants.nm,
                                              a / constants.nm,
                                              b / constants.nm,
                                              d / constants.nm),
            omega = np.array(omega_vals),
            part_ratio = np.array(part_vals),
            omega_sv = np.array(modes_o_sv),
            mx_sv = np.array(modes_mx_sv),
            mz_sv = np.array(modes_mz_sv),
            xr = xr, zr = zr, xs = xs, zs = zs, 
            xc_idx = xc_idx, zc_idx = zc_idx,
            dx_idx = dx_idx, dz_idx = dz_idx,
            mask_r = mask_r, mask_s = mask_s,
            Jex = Jex, Bext = Bext, Ms = Ms,
            W = W, s = s, a = a, b = b, d = d,
            i_r = i_r, i_s = i_s, j_r = j_r, j_s = j_s)
    #show_mode(modes_pos[0][0])
    #show_mode(modes_neg[0][0])
    #pl.show()
    #for i_mod, omega_mod in modes_neg:
    #    show_mode(i_mod)
    #    pl.show()
    
    
def modes(Bext, Ms, Jex, a, b, Nx, Nz):
    N = (Nx - 1) * (Nz - 1)
    Hxx = np.zeros((N, N))
    Hxz = np.zeros((N, N))
    Hzx = np.zeros((N, N))
    Hzz = np.zeros((N, N))

    J = np.zeros((N, N))
    
    def idx_to_ij(ij):
        i = ij // (Nz - 1)
        j = ij %  (Nz - 1)
        return i, j
    def ij_to_idx(i, j):
        return i * (Nz - 1) + j

    x  = np.linspace(-a/2.0, a/2.0, Nx)
    z  = np.linspace(-b/2.0, b/2.0, Nz)
    xm = 0.5 * (x[1:] + x[:-1])
    zm = 0.5 * (z[1:] + z[:-1])
    dx = x[1:] - x[:-1]
    dz = z[1:] - z[:-1]

    dx1 = xm[1:] - xm[:-1]
    dz1 = zm[1:] - zm[:-1]

    x_ij = np.zeros((N))
    z_ij = np.zeros((N))
    
    for i in range(Nx - 1):
        for j in range(Nz - 1):
            idx_o = ij_to_idx(i, j)
            x_ij[idx_o] = xm[i]
            z_ij[idx_o] = zm[j]

    for i1 in range(Nx - 1):
        x1 = xm[i1]
        xa = x[i1]
        xb = x[i1 + 1]
        for j1 in range(Nz - 1):
            z1 = zm[j1]
            za = z[j1]
            zb = z[j1 + 1]
            idx1 = ij_to_idx(i1, j1)
            Hxx[:, idx1] += H_xx(x_ij, z_ij, xb, za, zb)
            Hxx[:, idx1] -= H_xx(x_ij, z_ij, xa, za, zb)
            Hzx[:, idx1] += H_zx(x_ij, z_ij, xb, za, zb)
            Hzx[:, idx1] -= H_zx(x_ij, z_ij, xa, za, zb)
            Hxz[:, idx1] += H_xz(x_ij, z_ij, xa, xb, zb)
            Hxz[:, idx1] -= H_xz(x_ij, z_ij, xa, xb, za)
            Hzz[:, idx1] += H_zz(x_ij, z_ij, xa, xb, zb)
            Hzz[:, idx1] -= H_zz(x_ij, z_ij, xa, xb, za)
            #dV = np.abs(dx[i1] * dz[j1])
            #for i2  in range(Nx - 1):
            #    x2 = xm[i2]
            #    for j2 in range(Nz - 1):
            #        idx2 = ij_to_idx(i2, j2)
            #        #if idx1 == idx2:
            #        #    Hxx[idx2, idx1] -= np.pi * 0.01
            #        #    Hzz[idx2, idx1] -= np.pi * 0.01
            #        #    continue
            #        z2 = zm[j2]
            #        Hxx[idx2, idx1] += H_xx(x2, z2, xb, za, zb) 
            #        Hxx[idx2, idx1] -= H_xx(x2, z2, xa, za, zb) 
            #        Hzx[idx2, idx1] += H_zx(x2, z2, xb, za, zb) 
            #        Hzx[idx2, idx1] -= H_zx(x2, z2, xa, za, zb) 
            #        Hxz[idx2, idx1] += H_xz(x2, z2, xa, xb, zb) 
            #        Hxz[idx2, idx1] -= H_xz(x2, z2, xa, xb, za) 
            #        Hzz[idx2, idx1] += H_zz(x2, z2, xa, xb, zb) 
            #        Hzz[idx2, idx1] -= H_zz(x2, z2, xa, xb, za) 
    Hxx /= 2.0 * np.pi 
    Hxz /= 2.0 * np.pi 
    Hzx /= 2.0 * np.pi 
    Hzz /= 2.0 * np.pi 
    demag_xx = -np.sum(Hxx, axis=(0, 1)) / N
    demag_xz = -np.sum(Hxz, axis=(0, 1)) / N
    demag_zx = -np.sum(Hzx, axis=(0, 1)) / N
    demag_zz = -np.sum(Hzz, axis=(0, 1)) / N

    for i in range(Nx - 2):
        for j in range(Nz - 1):
            idx_w = ij_to_idx(i, j)
            idx_e = ij_to_idx(i + 1, j    )
            J[idx_e, idx_e] +=  1.0 / dx1[i]**2
            J[idx_w, idx_e] += -1.0 / dx1[i]**2
            J[idx_e, idx_w] += -1.0 / dx1[i]**2
            J[idx_w, idx_w] +=  1.0 / dx1[i]**2
            
    for i in range(Nx - 1):
        for j in range(Nz - 2):
            idx_s = ij_to_idx(i, j)
            idx_n = ij_to_idx(i, j + 1)
            J[idx_s, idx_s] +=  1.0 / dz1[j]**2
            J[idx_n, idx_s] += -1.0 / dz1[j]**2
            J[idx_s, idx_n] += -1.0 / dz1[j]**2
            J[idx_n, idx_n] +=  1.0 / dz1[j]**2

    
            
    if False:
        Hx_uni_x = np.sum(Hxx, axis=1)
        Hz_uni_x = np.sum(Hzx, axis=1)
        Hx_uni_z = np.sum(Hxz, axis=1)
        Hz_uni_z = np.sum(Hzz, axis=1)
        Z, X = np.meshgrid(zm, xm)
        pl.figure()
        pl.quiver(X, Z, Hx_uni_x, Hz_uni_x, pivot='middle', color='red')
        pl.gca().set_aspect('equal', 'box')
        pl.title("H from uni x")
        pl.figure()
        pl.quiver(X, Z, Hx_uni_z, Hz_uni_z, pivot='middle', color='blue')
        pl.gca().set_aspect('equal', 'box')
        pl.title("H from uni z")
        idx0 = ij_to_idx(Nx//2, Nz//2)
        ds = 0.001
        H1xx = H_xx(X, Z, ds, -ds, ds) - H_xx(X, Z, -ds, -ds, ds)
        H1zx = H_zx(X, Z, ds, -ds, ds) - H_zx(X, Z, -ds, -ds, ds)
        H1xz = H_xz(X, Z, -ds, ds, ds) - H_xz(X, Z, -ds,  ds, -ds)
        H1zz = H_zz(X, Z, -ds, ds, ds) - H_zz(X, Z, -ds,  ds, -ds)
        pl.figure()
        pl.quiver(X, Z, H1xx, H1zx, pivot='middle', color='red')
        pl.gca().set_aspect('equal', 'box')
        pl.title("H from dipole || x")
        pl.figure()
        pl.quiver(X, Z, H1zx, H1zz, pivot='middle', color='blue')
        pl.gca().set_aspect('equal', 'box')
        pl.title("H from dipole || z")
        #pl.title("fields of a dipole")
        pl.show()
    
    print ("demag coefficients:", demag_xx, demag_xz, demag_zx, demag_zz)

    L = np.zeros((2 * N, 2 * N))
    I = np.eye(N)
    L[0::2, 1::2] += Bext * I + Jex * Ms * J 
    L[1::2, 0::2] += - Bext * I - Jex * Ms * J
    L[0::2, 0::2] += -  Ms * constants.mu_0 * Hzx 
    L[0::2, 1::2] += -  Ms * constants.mu_0 * Hzz 
    L[1::2, 0::2] += +  Ms * constants.mu_0 * Hxx 
    L[1::2, 1::2] += +  Ms * constants.mu_0 * Hxz
    L *= constants.gamma_s / constants.GHz_2pi 


    print("antisymmetry: ", linalg.norm(np.transpose(L) + L))
    if False:
        Q = np.zeros((2 * N, 2 * N))
        Q[0::2, 0::2] = Hxx
        Q[0::2, 1::2] = Hxz
        Q[1::2, 0::2] = Hzx
        Q[1::2, 1::2] = Hzz
        print("symmetry: ", linalg.norm(np.transpose(Q) - Q))
        lmb_Q, ev_Q = linalg.eig(Q)
        print ("eig(Q) = ", lmb_Q)
        mxz1 = ev_Q[:, 0]
        mxz2 = ev_Q[:, -1]
        mx1 = mxz1[::2]
        mz1 = mxz1[1::2]
        mx2 = mxz2[::2]
        mz2 = mxz2[1::2]
        pl.figure()
        pl.quiver(X, Z, mx1, mz1)
        pl.gca().set_aspect('equal', 'box')
        pl.figure()
        pl.quiver(X, Z, mx2, mz2)
        pl.gca().set_aspect('equal', 'box')
    iomega, mxz = linalg.eig(L)

    print ("eigenvals: ", iomega)
    print ("dangerous: ", [t for t in iomega if abs(t.real) > 1e-10])
    iomega1 = list(iomega)
    iomega1.sort(key = lambda z: np.abs(z.imag))
    #print (iomega1)

    
    
    om = iomega.imag
    pl.figure()
    pl.hist(om, bins=np.linspace(-10.0, 10.0, 201))


    import random
    Z, X = np.meshgrid(zm, xm)


    def show_mode(i_mode):
        mxz_i = mxz[:, i_mode]
        print ("mode ", i_mode, "o = ", iomega[i_mode], "check: ",
               linalg.norm(np.dot(L, mxz_i) - iomega[i_mode] * mxz_i))
        mx_flat = mxz_i[0::2]
        mz_flat = mxz_i[1::2]
        mx = np.zeros((Nx - 1, Nz - 1), dtype=complex)
        mz = np.zeros((Nx - 1, Nz - 1), dtype=complex)
        for i in range(Nx - 1):
            for j in range(Nz - 1):
                mx[i, j] = mx_flat[ij_to_idx(i, j)]
                mz[i, j] = mz_flat[ij_to_idx(i, j)]
        #print ("mx")
        #print (mx)
        #print ("mz")
        #print (mz)
        pl.figure()
        mxmax = np.max(np.abs(mx.real))
        mzmax = np.max(np.abs(mz.real))
        pl.pcolormesh(X, Z, mx.real, cmap='bwr', vmin=-mxmax, vmax=mxmax)
        pl.gca().set_aspect('equal', 'box')
        pl.colorbar()
        pl.title("mx for o = %g %g" % (iomega[i_mode].real, iomega[i_mode].imag))
        pl.figure()
        pl.pcolormesh(X, Z, mz.real, cmap='bwr', vmin=-mzmax, vmax=mzmax)
        pl.gca().set_aspect('equal', 'box')
        pl.colorbar()
        pl.title("mz for o = %g %g"% (iomega[i_mode].real, iomega[i_mode].imag))
        mx_max = max(np.max(np.abs(mx.real)), np.max(np.abs(mx.imag)))
        mz_max = max(np.max(np.abs(mz.real)), np.max(np.abs(mz.imag)))
        q_scale = 0.8 * min(abs(dx[0]), abs(dz[0])) / max(mx_max, mz_max)
        q_scale = 1.0/q_scale
        pl.figure()
        pl.quiver(X, Z, mx.real, mz.real, scale=q_scale, scale_units='x',
                  pivot='middle', color='red')
        pl.gca().set_aspect('equal', 'box')
        pl.title("Re mode %d o = %g %g" % (i_mode, iomega[i_mode].real,
                                        iomega[i_mode].imag))

        #pl.figure()
        pl.quiver(X, Z, mx.imag, mz.imag, scale=q_scale, scale_units='x',
                  pivot='middle',
                  color='blue')
        pl.gca().set_aspect('equal', 'box')
        pl.title("Im mode %d o = %g %g" % (i_mode, iomega[i_mode].real,
                                        iomega[i_mode].imag))

    if False:
        for i_rnd in range(5):
            i_mode = int(random.random() * len(iomega))
            show_mode(i_mode)
        i_max = np.argmax(np.abs(iomega.imag))
        i_min = np.argmin(np.abs(iomega.imag))
        show_mode(i_max)
        show_mode(i_min)
        pl.show()
    modes = list(enumerate(iomega))
    modes_pos = list([t for t in modes if (t[1].imag > 0)])
    modes_neg = list([t for t in modes if (t[1].imag <= 0)])
    modes_neg.sort(key = lambda t: -t[1].imag)
    modes_pos.sort(key = lambda t:  t[1].imag)
    show_mode(modes_pos[0][0])
    show_mode(modes_neg[0][0])
    pl.show()
    for i_mod, omega_mod in modes_neg:
        show_mode(i_mod)
        pl.show()
        
        

Bext = 5 * constants.mT
Ms = 140.0 * constants.kA_m
#a = 200 * constants.nm
a = 200 * constants.nm
b = 30 * constants.nm
Nx = 100
Nz = 15
Jex = 3.5 * constants.pJ_m / Ms**2 * 1.0

#def modes_big(Bext, Ms, Jex, a, b, Nrx, Nrz, s, W, d, Nsx, Nsz):
    
#modes(Bext, Ms, Jex, a, b, Nx, Nz)
#import sys
#sys.exit(-1)
#for s_nm in [100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20,
#             18, 16, 14, 12, 10, 9, 8, 7, 6, 5]:
for s_nm in [29, 28, 27, 26, 24, 23, 22, 21, 19, 17, 15, 13, 11, 110, 125, 150, 200, 250, 300]:

    s = s_nm * constants.nm
    W = a + 40 * b
    d = 20 * constants.nm
    Nrx = Nx
    Nrz = Nz
    Nsx = 200
    Nsz = 5
    modes_big(Bext, Ms, Jex, a, b, Nrx, Nrz, s, W, d, Nsx, Nsz)
    

    
