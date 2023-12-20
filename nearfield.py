import numpy as np
from scipy import linalg
from resonator2d import ResonantMode2D
from resonator2d import make_mode_collection
import constants

#
# There must be a way to do it faster and neater
#
def restrict(L, mask1, mask2):
    #return L[mask1 > 0, mask2 > 0]
    #out = []
    #N_L, M_L = np.shape(L)
    i_mask1 = [t for t in range (len(mask1)) if mask1[t] > 0]
    j_mask2 = [t for t in range (len(mask2)) if mask2[t] > 0]
    L_r = np.zeros((len(i_mask1), len(j_mask2)), dtype=L.dtype)
    for i_R, i_L in enumerate(i_mask1):
        for j_R, j_L in enumerate(j_mask2):
            L_r[i_R, j_R] = L[i_L, j_L]
    return L_r
    #for i in range(M_L):
    #    if mask1[i] < 0.5: continue
    #    line = []
    #    for j in range(N_L):
    #        if mask2[j] < 0.5: continue
    #        line.append(L[i, j])
    #    line = np.array(line)
    #    out.append(line)
    #return np.array(out)

def restrict2(L, mask1, mask2):
    Nr = len([t for t in range(len(mask1)) if mask1[t] > 0])
    Mr = len([t for t in range(len(mask2)) if mask2[t] > 0])
    Lr = np.zeros((2 * Nr, 2 * Mr), dtype=L.dtype)
    Lr[0::2, 0::2] = restrict(L[0::2, 0::2], mask1, mask2)
    Lr[0::2, 1::2] = restrict(L[0::2, 1::2], mask1, mask2)
    Lr[1::2, 0::2] = restrict(L[1::2, 0::2], mask1, mask2)
    Lr[1::2, 1::2] = restrict(L[1::2, 1::2], mask1, mask2)
    return Lr

def do_solveNearField(model,  *areas):
    L  = model.compute_LLG_operator()
    I  = model.get_I()
    IL = np.dot(I, L)

    res_mask = model.get_area_mask(areas[0])
    for area in areas[1:]:
        res_mask += model.get_area_mask(area)

    slab_mask = np.ones(np.shape(res_mask))
    slab_mask[res_mask > 0] = 0

    Nr = len([t for t in range(len(res_mask)) if res_mask[t] > 0])
    Ns = len([t for t in range(len(slab_mask)) if slab_mask[t] > 0])
    R  = np.zeros((2 * Nr, 2 * Nr))
    S  = np.zeros((2 * Ns, 2 * Ns))
    Q  = np.zeros((2 * Ns, 2 * Nr))
    K  = np.zeros((2 * Nr, 2 * Ns))
    Ir = np.zeros((2 * Nr, 2 * Nr)) 
    Is = np.zeros((2 * Ns, 2 * Ns))

    #Xr = np.zeros((Nr))
    #Zr = np.zeros((Nr))
    #Xr = model.box_array.X[res_mask > 0]
    #Zr = model.box_array.Z[res_mask > 0]
    #dV_r = model.box_array.dV()[res_mask > 0]

    R  = restrict2(IL, res_mask, res_mask)
    Ir = restrict2(I,  res_mask, res_mask)
    
    S = restrict2(IL, slab_mask, slab_mask)
    Is = restrict2(I, slab_mask, slab_mask)
    
    Q = restrict2(IL, slab_mask, res_mask)
    K = restrict2(IL, res_mask, slab_mask)

    Sinv  = linalg.inv(S)
    SinvQ = np.dot(Sinv, Q)
    R_nrf = R - np.dot(K, SinvQ)
    I_nrf = Ir + np.dot( np.dot(K, Sinv), np.dot(Is, SinvQ))
    I_nrf_inv = linalg.inv(I_nrf)
    L_nrf = np.dot(I_nrf_inv, R_nrf)

    iomega, mxz = linalg.eig(L_nrf)
    omega = 1j * iomega

    i_pos = [t for t in range(len(omega)) if omega[t].real > 0]
    i_pos.sort(key = lambda t: omega[t].real)
    return iomega, mxz, I_nrf, res_mask

def solveNearField(model, f_min, f_max, Lr, Wr, Hr, theta_or, *areas):

    
    iomega, mxz, I_nrf, res_mask = do_solveNearField(model, *areas)

    Nr = len([t for t in range(len(res_mask)) if res_mask[t] > 0])
    Xr = np.zeros((Nr))
    Zr = np.zeros((Nr))
    Xr = model.box_array.X[res_mask > 0]
    Zr = model.box_array.Z[res_mask > 0]
    dV_r = model.box_array.dV()[res_mask > 0]

    modes = make_mode_collection(iomega, mxz, I_nrf,
                                 f_min, f_max,
                                 Xr, Zr, dV_r,
                                 Lr, Wr, Hr, theta_or)

    if  False:
        for i in i_pos:
            print ("mode freq: ", omega[i] / constants.GHz_2pi)
            if omega[i].real < f_min * constants.GHz_2pi: continue
            if omega[i].real > f_max * constants.GHz_2pi: continue
            print ("append mode")
            mxz_i = np.array(mxz[:, i])
            C_norm = np.dot(mxz_i.conj(), np.dot(I_nrf, mxz_i))
            mxz_i *= np.sqrt(1.0 / np.abs(C_norm))
            #corr = np.dot(mxz_i.conj(), np.dot(Sigma_2, mxz_i)) * omega[i]**2
            #corr /= np.dot(mxz_i.conj(), np.dot(I_nrf, mxz_i)) * 1j
            #print ("correction: ", corr)
            i_max = np.argmax(np.abs(mxz_i))
            mxz_i *= np.abs(mxz_i[i_max]) / mxz_i[i_max]
            mx_i = mxz_i[0::2]
            mz_i = mxz_i[1::2]
            import pylab as pl
            mxz_i_slab = -np.dot(SinvQ, mxz_i)
            mx_s, mz_s = mxz_i_slab[0::2], mxz_i_slab[1::2]
            Xs = model.box_array.X[slab_mask > 0]
            Zs = model.box_array.Z[slab_mask > 0]
            mx_max = np.max(np.abs(mx_i))
            mz_max = np.max(np.abs(mz_i))
            pl.figure()
            pl.tripcolor(Xr / constants.nm, Zr / constants.nm,
                                np.abs(mx_i), vmin=0.0, vmax=mx_max,
                                cmap='magma')
            pl.tripcolor(Xs / constants.nm, Zs / constants.nm,
                                np.abs(mx_s), vmin=0.0, vmax=mx_max,
                                cmap='magma')
            pl.gca().set_aspect('equal', 'box')
            cb = pl.colorbar()
            cb.set_label(r"Horizontal magnetisation $|m_x|$, a.u.")
            pl.xlabel(r"Position $x$, nm")
            pl.xlabel(r"Position $z$, nm")

            pl.figure()
            pl.tripcolor(Xr / constants.nm, Zr / constants.nm,
                            np.abs(mz_i), vmin=0.0, vmax=mz_max,
                            cmap='magma')
            pl.tripcolor(Xs / constants.nm, Zs / constants.nm,
                            np.abs(mz_s), vmin=0.0, vmax=mz_max,
                            cmap='magma')
            pl.gca().set_aspect('equal', 'box')
            cb = pl.colorbar()
            cb.set_label(r"Vertical magnetisation, $|m_z|$, a.u.")
            pl.xlabel(r"Position $x$, nm")
            pl.xlabel(r"Position $z$, nm")
            pl.figure()
            pl.quiver(Xr / constants.nm, Zr / constants.nm,
                         5 * mx_i.real, 5 * mz_i.real, pivot='middle')
            pl.quiver(Xs / constants.nm, Zs / constants.nm,
                         5 * mx_s.real, 5 * mz_s.real, pivot='middle')
            pl.xlabel(r"Position $x$, nm")
            pl.xlabel(r"Position $z$, nm")
            pl.gca().set_aspect('equal', 'box')
            #I_nrf2 = I_nrf - 2.0 * Sigma_2 * omega[i] * 1j
            #R_nrf2 =  - Sigma_2 * omega[i]**2
            #iomega2, mxz2 = linalg.eig(np.dot(linalg.inv(I_nrf2), R_nrf2))
            #omega2 = iomega2 *  1j
            #i_closest = np.argmin(np.abs(omega2))
            #print ("closest ev: ", omega2[i_closest] / constants.GHz_2pi)
        pl.show()
            
    
    return modes
