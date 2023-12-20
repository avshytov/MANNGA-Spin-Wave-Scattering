import numpy as np
import pylab as pl
import constants

from modes2 import Material, Area, Grid, CellArrayModel, VariableGrid
from slab import Slab
from dispersion import Dispersion

a = 200 * constants.nm
b = 30  * constants.nm
d = 20  * constants.nm
s = 50  * constants.nm

Bext = 5.0 * constants.mT

Nx = 50
Nz = 7

Nsz = 3

YIG_Ms = 140 * constants.kA_m
YIG_alpha = 1 * 0.001
YIG_gamma_s = constants.gamma_s
Aex = 2 * 3.5 * constants.pJ_m
YIG_Jex = Aex / YIG_Ms**2

YIG = Material("YIG", YIG_Ms, YIG_Jex, YIG_alpha, YIG_gamma_s)

inf_slab = Slab(-d, 0.0, Bext, YIG_Ms, YIG_Jex, YIG_alpha, Nsz)

class SlabDispersion(Dispersion):
    def __init__(self, slab):
        self.slab = slab
        Dispersion.__init__ (self, self.omega_k)

    def omega_k(self, kx, ky):
        return self.mode(kx, ky).omega
    
    def mode(self, kx, ky):
        mode_plus_E, mode_minus_E = self.slab.make_modes(kx, ky)
        return mode_plus_E[0]

slab_dispersion = SlabDispersion(inf_slab)

def solve_scattering_problem(omega):
    k = slab_dispersion.k_omega(omega, 0.0)
    resonator_area = Area("resonator",
                          Grid(-a/2.0, a/2.0, 0, b, Nx, Nz),
                          YIG)
    model = CellArrayModel()
    model.add_area(resonator_area, Bext)

    lmb = 2.0 * np.pi / k
    W_right = a/2 + 5 * b + 2.5 * lmb
    W_left  = a/2 + 5 * b + 2.5 * lmb
    W_damp  = max(300 * constants.nm, 1.5 * lmb)
    #W_damp  = 300 * constants.nm
    W_total = W_right + W_left + 2 * W_damp
    dx_min  = min(lmb / 30, a/10, b/5, s/2)
    Nsx = int(W_total // dx_min + 10)
    Nsz = 3

    slab_z = np.linspace(-s - d, -s, Nsz + 1)
    if  False:
        dx0 = min(a/10, b/5, s/5)
        dx1 = lmb / 30
        N_r = int(2.0 * (W_right + W_damp) / (dx0 + dx1))
        N_l = int(2.0 * (W_right + W_damp) / (dx0 + dx1))
        print ("N_r, N_l = ", N_r, N_l)
        tvals_l = np.linspace(0.0, 1.0, N_l + 1)
        tvals_r = np.linspace(0.0, 1.0, N_r + 1)
        #tvals_l = tvals_l[1:]
        b_l = dx0 * N_l
        b_r = dx0 * N_r
        a_l = 0.5 * (dx1 * N_l - b_l)
        a_r = 0.5 * (dx1 * N_r - b_r)
        slab_x = np.zeros((N_l + N_r + 1))
        print ("np.shape(-Nr...)", np.shape(slab_x[-N_r::-1]),
               np.shape(slab_x))
        slab_x[N_l:] = a_r * tvals_r**2 + b_r * tvals_r
        slab_x[-N_r-1::-1] = - a_l * tvals_l**2 - b_l * tvals_l
        if  False:
            pl.figure()
            pl.plot(range(len(slab_x)), slab_x / constants.nm, '-')
            pl.plot(range(len(slab_x)), slab_x / constants.nm, 'o')
            pl.show()
    else:
        slab_x = np.linspace(-W_left - W_damp, W_left + W_damp, Nsx)

    slab_grid = VariableGrid(slab_x, slab_z)
        
    print ("Nsx = ", Nsx)

    YIG_d = Material("YIG", YIG_Ms, YIG_Jex, 0.0001, YIG_gamma_s)
    def alpha_slab(x, z):
         if x > W_right:
            return 0.0001 + 1.0 * ((x - W_right) / W_damp)**2
         if x < -W_left:
            return 0.0001 + 1.0 * ((-x - W_left) / W_damp)**2
         return 0.0001
    YIG_d.alpha_func = alpha_slab

    x0_fw = -W_left  + 0.125 * lmb 
    x0_bk =  W_right - 0.125 * lmb

    slab_area = Area("slab", slab_grid, YIG_d)
    model.add_area (slab_area, Bext)
    
    sigma = 5.0 * constants.nm
    h0 = 1.0
    
    def h_src_fwd(x, z):
         exp_x = np.exp(-(x - x0_fw)**2 / 2.0 / sigma**2)
         return h0 * exp_x, 0.0 * exp_x
         
    def h_src_bk(x, z):
         exp_x = np.exp(-(x - x0_bk)**2 / 2.0 / sigma**2)
         return h0 * exp_x, 0.0 * exp_x

    print ("solving: f = ", omega / constants.GHz_2pi)
    result_fw = model.solve_response(np.array([omega]), h_src_fwd)
    result_bk = model.solve_response(np.array([omega]), h_src_bk)
    
    xs = result_fw['coords']['slab'][0][:, -1]
    xr = result_fw['coords']['resonator'][0][:, -1]
    mxs_fw, mzs_fw = result_fw['modes'][0].m('slab')
    mxs_fw = mxs_fw[:, 0]; mzs_fw = mzs_fw[:, 0]
    mxr_fw, mzr_fw = result_fw['modes'][0].m('resonator')
    mxr_fw = mxr_fw[:, -1]; mzr_fw = mzr_fw[:, -1]
    mxs_bk, mzs_bk = result_bk['modes'][0].m('slab')
    mxs_bk = mxs_bk[:, 0]; mzs_bk = mzs_bk[:, 0]
    mxr_bk, mzr_bk = result_bk['modes'][0].m('resonator')
    mxr_bk = mxr_bk[:, -1]; mzr_bk = mzr_bk[:, -1]

    if  False:
        pl.plot(xs, mxs_fw.real, label='Re mx ->')
        pl.plot(xs, mxs_fw.imag, label='Im mx ->')
        pl.plot(xs, mxs_bk.real, label='Re mx <-')
        pl.plot(xs, mxs_bk.imag, label='Im mx <-')
        pl.plot(xs, np.abs(mxs_fw), label='|mx| ->')
        pl.plot(xs, np.abs(mxs_bk), label='|mx| <-')
        pl.plot(xs, np.vectorize(lambda x: YIG_d.alpha_func(x, 0.0))(xs),
                'k--')
        pl.legend()
        pl.show()
    
    return dict(k = k, omega = omega,
                xs = xs, xr = xr,
                mxs_fw = mxs_fw, mzs_fw = mzs_fw, 
                mxs_bk = mxs_bk, mzs_bk = mzs_bk, 
                mxr_fw = mxr_fw, mzr_fw = mzr_fw, 
                mxr_bk = mxr_bk, mzr_bk = mzr_bk,
                N = len(xs))

def extend_vector(v, N):
    #print ("extend: ", np.shape(v))
    v_new = np.zeros(N, dtype=v.dtype)
    v_new[:len(v)] = v
    return v_new

def extend_vectors_in_list(l_v, N):
    l_new = []
    for v in l_v:
        #print ("extend vector in list")
        l_new.append(extend_vector(v, N))
    return l_new

def solve_TR(omega_tab, fname):
    data = dict()
    Nmax = 0
    
    keys = ['mxs_fw', 'mzs_fw', 'mxs_bk', 'mzs_bk',
            'mxr_fw', 'mzr_fw', 'mxr_bk', 'mzr_bk',
            'xs', 'xr', 'k']

    do_extend = dict()
    for k in keys:
        do_extend[k] = True
    for k in ['k', 'xr', 'mxr_fw', 'mxr_bk', 'mzr_fw', 'mzr_bk']:
        do_extend[k] = False
        
    for k in keys:
        data[k] = []
        
    N = []

    omega_done = []
    for omega in omega_tab:

        result = solve_scattering_problem(omega)
        omega_done.append(omega)
        
        if result['N'] > Nmax:
           Nmax = result['N']
           for k in keys:
               if not do_extend[k]: continue
               #print ("extend list: ", k)
               data[k] = extend_vectors_in_list(data[k], Nmax)
        else:
           for k in keys:
               #print ("extend vector: ", k)
               if not do_extend[k]: continue
               result[k] = extend_vector(result[k], Nmax)
               
        for k in keys:
            data[k].append(result[k])
            
        N.append(result['N'])

        res_cur = dict()
        for k in keys:
            #print ("data[k]", k, type(data[k]))
            #print ("data:", data[k])
            res_cur[k] = np.array(data[k])

        np.savez(fname,
                 omega = np.array(omega_done),
                 a = a, b = b, s = s, d = d, Bext = Bext,
                 N = np.array(N, dtype=int),
                 **res_cur)

#omega_tab = np.linspace(1.7, 2.3, 301) * constants.GHz_2pi
#omega_tab = np.linspace(2.6, 3.0,  801) * constants.GHz_2pi
#omega_tab = np.linspace(3.7, 4.2, 251) * constants.GHz_2pi
#omega_tab = np.linspace(3.6, 4.2, 301) * constants.GHz_2pi
#omega_tab = np.linspace(2.7, 2.9, 401) * constants.GHz_2pi
omega_tab = np.linspace(3.8, 4.0, 401) * constants.GHz_2pi
#omega_tab = np.linspace(1.0, 4.2, 321) * constants.GHz_2pi
#omega_tab = np.linspace(1.7, 4.2, 250 + 1) * constants.GHz_2pi   
#omega_tab = np.linspace(1.5, 4.2, 270 + 1) * constants.GHz_2pi   
#omega_tab  = np.linspace(1.2, 4.2, 300 + 1) * constants.GHz_2pi   
#omega_tab  = np.linspace(1.0, 4.2, 320 + 1) * constants.GHz_2pi   
#omega_tab  = np.linspace(1.8, 2.1, 600 + 1) * constants.GHz_2pi   
#omega_tab = np.linspace(2.5, 3.2, 350 + 1) * constants.GHz_2pi   
#omega_tab = np.linspace(3.0, 3.3, 150 + 1) * constants.GHz_2pi
#omega_tab = np.linspace(3.9, 4.2, 150 + 1) * constants.GHz_2pi
#omega_tab = np.linspace(1.5, 2.5, 500 + 1) * constants.GHz_2pi

nm = constants.nm
GHz_2pi = constants.GHz_2pi
geom_vals = (a / nm, b/nm, s/nm, d/nm, Bext / constants.mT,
             omega_tab[0] / GHz_2pi, omega_tab[-1] / GHz_2pi)
fname = "trsolution-%gnmx%gnm-s=%gnm-d=%gnm-B=%gmT-%g-%gGHz.npz" % geom_vals

solve_TR(omega_tab, fname)
