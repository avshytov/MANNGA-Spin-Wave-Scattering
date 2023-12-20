import numpy as np
import pylab as pl
import sys
from scipy import interpolate, linalg

from resonator3d import ResonantMode3D
import constants
from scattering import ScatteringProblem

from slab import Slab
from sigma import Sigma, Sigma_Gamma

debug = False

def get_s(d, s_default = -1):
    if 's' in d.keys():
        s = d['s']
    elif s_default > 0:
        print ("Use default value for s = ", s_default)
        return s_default
    else:
        fields = fname.split('-')     # "-*-*-s=20nm-"
        for field in fields:
            if field[:2] == 's=':     # "s=20nm"
                kv = field.split('=') # "s=20nm"
                s = constants.nm * float(kv[1][:-2])    # "20nm"
    print ("s = ", s)
    return s 

def get_d_film(d, d_default = -1):
    if 'd_film' in d.keys():
        d_film = d['d_film']
    elif d_default  > 0:
        print ("use default value for d_film", d_default)
        return d_default
    else:
        fields = fname.split('-')     # "-*-*-d=20nm-"
        for field in fields:
            if field[:2] == 'd=':     # "d=20nm"
                kv = field.split('=') # "d=20nm"
                print (field, kv)
                d_film = constants.nm * float(kv[1][:-2])    # "20nm"
    print ("d_film = ", d_film)
    return d_film 

def setup_scattering_problem(d, s_default, d_default):
    X = d['X']
    Y = d['Y']
    Z = d['Z']
    ma = d['ma']
    mb = d['mb']
    mx = d['mx']
    my = d['my']
    mz = d['mz']
    dV = d['dV']
    L = d['a']
    W = d['b']
    H = d['c']
    omega = d['omega']
    d_film = get_d_film(d, d_default)
    s = get_s(d, s_default)
    Bext = d['Bext']
    mode = ResonantMode3D(L, W, H, X, Y, Z, dV, omega,
                          ma, mb, mx, my, mz, 0.0)

    YIG_Ms = 140 * constants.kA_m
    YIG_alpha = 1 * 0.001
    YIG_gamma_s = constants.gamma_s
    Aex = 2 * 3.5 * constants.pJ_m
    YIG_Jex = Aex / YIG_Ms**2

    if not debug: N_slab = 10
    else:         N_slab = 3
    
    slab = Slab(-s - d_film, -s, Bext, YIG_Ms, YIG_Jex, YIG_alpha, N_slab)

    sc_problem = ScatteringProblem(slab, mode, 0.0)

    return sc_problem

    
def tabulate_Gamma(sc_problem, omega_tab):
    Gamma_result = []
    Gamma_prime_result = []
    Gamma_tilde_result = []
    for omega in omega_tab:
        Gamma_omega       = sc_problem.Gamma_rad(omega)
        Gamma_prime_omega = sc_problem.Gamma_prime(omega)
        Gamma_tilde_omega = sc_problem.Gamma_tilde(omega)
        Gamma_result.append(Gamma_omega)
        Gamma_prime_result.append(Gamma_prime_omega)
        Gamma_tilde_result.append(Gamma_tilde_omega)
        print ("f = ", omega / constants.GHz_2pi,
               'Gamma = ',       Gamma_omega,
               'Gamma_prime = ', Gamma_prime_omega,
               "Gamma_tilde = ", Gamma_tilde_omega)

    Gamma_result = np.array(Gamma_result)
    Gamma_prime_result = np.array(Gamma_prime_result)
    Gamma_tilde_result = np.array(Gamma_tilde_result)
    
    return Gamma_result, Gamma_prime_result, Gamma_tilde_result


def tabulate_scattering(sc_problem, omega, theta_vals):
    k_theta = 0.0 * theta_vals 
    u_theta = 0.0 * theta_vals
    v_theta = 0.0 * theta_vals
    alpha_theta = 0.0 * theta_vals
    alpha_p_theta = 0.0 * theta_vals
    Delta_theta = 0.0 * theta_vals + 0.0j
    Delta1_theta = 0.0 * theta_vals + 0.0j
    print ("tabulating scattered waves")

    for i_theta, theta in enumerate(theta_vals):
        Delta_k, k, u, v, alpha, alpha_p = sc_problem.find_data_theta(omega, theta)
        k_theta[i_theta] = k
        Delta_theta[i_theta] = Delta_k
        u_theta[i_theta] = u
        v_theta[i_theta] = v
        alpha_theta[i_theta] = alpha
        alpha_p_theta[i_theta] = alpha_p
        kx = k * np.cos(theta)
        ky = k * np.sin(theta)
        Delta1_theta[i_theta] = sc_problem.Delta_prime(kx, ky)

    result = dict()
    result['omega'] = omega
    result['theta_vals'] = theta_vals
    result['Delta'] = Delta_theta
    result['Delta1'] = Delta1_theta
    result['k'] = k_theta
    result['u'] = u_theta
    result['v'] = v_theta
    result['alpha'] = alpha_theta
    result['alpha_p'] = alpha_p_theta

    print ("done")
    return result

def tabulate_inc(sc_problem, omega_tab, omega_min, theta_tab):
    Delta_inc  = np.zeros((len(omega_tab), len(theta_tab)), dtype=complex)
    Delta1_inc = np.zeros((len(omega_tab), len(theta_tab)), dtype=complex)
    k_inc      = np.zeros((len(omega_tab), len(theta_tab)))
    u_inc      = np.zeros((len(omega_tab), len(theta_tab)))
    v_inc      = np.zeros((len(omega_tab), len(theta_tab)))
    alpha_inc  = np.zeros((len(omega_tab), len(theta_tab)))
    print ("tabulating incident waves")
    for i_omega, omega in enumerate(omega_tab):
       if omega < omega_min + 0.000 * constants.GHz: continue
       for j_theta, theta in enumerate(theta_tab):
           #print ("i, j", i_omega, j_theta,
           #       omega_tab[i_omega] / constants.GHz_2pi,
           #       theta / np.pi * 180.0)
           #Delta_k, k, u, v, alpha, alpha_p
           sc_data  = sc_problem.find_data_theta(omega, theta)
           Delta_k, k, u, v, alpha, alpha_p = sc_data
           kx = k * np.cos(theta)
           ky = k * np.sin(theta)
           Delta1_k = sc_problem.Delta_prime(kx, ky)
           Delta_inc [i_omega, j_theta] = Delta_k
           Delta1_inc[i_omega, j_theta] = Delta1_k
           k_inc[i_omega, j_theta] = k
           u_inc[i_omega, j_theta] = u
           v_inc[i_omega, j_theta] = v
           alpha_inc[i_omega, j_theta]  = alpha

    result = dict()
    result['omega'] = omega_tab
    result['theta_inc'] = theta_tab
    result['Delta_inc'] = Delta_inc
    result['Delta_inc_prime'] = Delta1_inc
    result['k_inc'] = k_inc
    result['u_inc'] = u_inc
    result['v_inc'] = v_inc
    result['alpha_inc'] = alpha_inc
    print ("tabulating incident done")
    return result

def solve(f, a, b):
    fa = f(a)
    fb = f(b)
    while abs(a - b) > 1e-6:
        c = 0.5 * (a + b)
        fc = f(c)
        print ("a, c, b = ", a, c, b)
        print ("fa, fc, fb = ", fa, fc, fb)
        if (fc * fa) <= 0:
            b   = c
            fb = fc
        elif (fc * fb) <= 0:
            a = c
            fa = fc
        else:
            raise Exception("cannot find root")
    return 0.5 * (a + b)

def process_data(fname, fname_analysis, s = -1, d_film = -1, nrf=True):
    d = np.load(fname)
    for k in d.keys(): print (k)
    sc_problem   = setup_scattering_problem(d, s, d_film)
    Omega_0      =  d['omega'].real
    Gamma_0      = -d['omega'].imag
    if not debug: omega_max    = constants.GHz_2pi * 20
    else:         omega_max    = constants.GHz_2pi * 15
    #omega_max    = constants.GHz_2pi * 15
    #omega_min    = sc_problem.omega_k(1e-2, 0.0) + constants.GHz_2pi * 0.02
    omega_min    = sc_problem.omega_k(1e-6, 0.0)
    #+ constants.GHz_2pi * 0.0001
    omega_min    = omega_min.real
    if not debug: N_coarse = 200
    else:         N_coarse = 5 #50
    
    omega_coarse = np.linspace(omega_min, omega_max, N_coarse + 1)
    #omega_tab = constants.GHz_2pi * np.linspace(3.7, 20.0, 163 + 1)

    Gamma_coarse       = 0.0 * omega_coarse
    Gamma_prime_coarse = 0.0 * omega_coarse
    Gamma_tilde_coarse = 0.0 * omega_coarse + 0.0j
    Gammas_tabulated   = tabulate_Gamma(sc_problem, omega_coarse[1:])
    Gamma_coarse[1:]       = Gammas_tabulated[0]
    Gamma_prime_coarse[1:] = Gammas_tabulated[1]
    Gamma_tilde_coarse[1:] = Gammas_tabulated[2]
    
    Gamma_spl          = interpolate.splrep(omega_coarse,
                                            Gamma_coarse.real)
    Gamma_prime_spl    = interpolate.splrep(omega_coarse,
                                            Gamma_prime_coarse.real)
    Gamma_tilde_re_spl = interpolate.splrep(omega_coarse,
                                            Gamma_tilde_coarse.real)
    Gamma_tilde_im_spl = interpolate.splrep(omega_coarse,
                                            Gamma_tilde_coarse.imag)

    omega_fine        = np.linspace(omega_coarse[0],
                                    omega_coarse[-1],
                                    20 * (len(omega_coarse) - 1) + 1)
    Gamma_fine        = interpolate.splev(omega_fine, Gamma_spl)
    Gamma_prime_fine  = interpolate.splev(omega_fine, Gamma_prime_spl)
    Gamma_tilde_fine  = 0.0 * omega_fine + 0.0j
    Gamma_tilde_fine += interpolate.splev(omega_fine, Gamma_tilde_re_spl) 
    Gamma_tilde_fine += 1j * interpolate.splev(omega_fine, Gamma_tilde_im_spl)

    #omega_m    = 0.5 * (omega_fine[1:] + omega_fine[:-1])
    if not debug: N_tab = 4001
    else:         N_tab = 101
    do = 0.1 * (omega_coarse[-1] - omega_coarse[-2])
    omega_m    = np.linspace(-omega_max + do, omega_max - do, N_tab)
    omega_m_new = []
    for omega_i in omega_m:
        if np.min(np.abs(omega_i - omega_fine)) < 1e-6:
            print ("discard+: ", omega_i)
            continue
        if np.min(np.abs(omega_i + omega_fine)) < 1e-6:
            print ("discard-: ", omega_i)
            continue
        omega_m_new.append(omega_i)

    omega_m = np.array(omega_m_new)
    #k_max = 100.0
    #N_k = 1000
    #Sigma_m    = Sigma(omega_m, omega_fine, Gamma_fine,
    #                   sc_problem, k_max, N_k)
    
    def _Gamma_s(omega):
        if omega < omega_min: return 0.0
        return interpolate.splev(omega, Gamma_spl)

    def _Gamma_prime_s(omega):
        if omega < omega_min: return 0.0
        return interpolate.splev(omega, Gamma_prime_spl)

    def _Gamma_tilde_s(omega):
        if omega < omega_min: return 0.0 + 0.0j
        #print ("b2a")
        ret  = 0.0 * omega + 0.0j
        #print ("b2b")
        ret +=       interpolate.splev(omega, Gamma_tilde_re_spl)
        #print ("b2c")
        ret +=  1j * interpolate.splev(omega, Gamma_tilde_im_spl)
        #print ("b2d", ret)
        return ret

    #print ("a")
    Gamma_s       = np.vectorize(_Gamma_s)
    Gamma_prime_s = np.vectorize(_Gamma_prime_s)
    Gamma_tilde_s = np.vectorize(_Gamma_tilde_s)

    #print ("b")
    Gamma_m       = Gamma_s(omega_m)
    Gamma_m_neg   = Gamma_s(-omega_m)
    #print ("b1")
    Gamma_m_prime = Gamma_prime_s(omega_m)
    Gamma_m_prime_neg = Gamma_prime_s(-omega_m)
    #print ("b2")
    Gamma_m_tile = 0.0 * omega_m + 0.0j
    Gamma_m_tilde = Gamma_tilde_s(omega_m)
    Gamma_m_tilde_neg = Gamma_tilde_s(-omega_m)

    #print ("c")
    Sigma_m_pp    = Sigma(omega_m, omega_fine, Gamma_fine,
                          Gamma_prime_fine, nrf)
    Sigma_m_pp   += -1j * Gamma_m 
    Sigma_m_pp   +=  1j * Gamma_m_prime_neg
    
    #print ("d")
    Sigma_m_mm    = Sigma(omega_m, omega_fine, Gamma_prime_fine,
                          Gamma_fine, nrf)
    Sigma_m_mm   += -1j * Gamma_m_prime
    Sigma_m_mm   +=  1j * Gamma_m_neg

    #print ("e")
    Sigma_m_pm    = Sigma(omega_m, omega_fine, Gamma_tilde_fine.conj(),
                                               Gamma_tilde_fine.conj(), nrf)
    Sigma_m_pm   += -1j * Gamma_m_tilde.conj()  
    Sigma_m_pm   +=  1j * Gamma_m_tilde_neg.conj()  

    #print ("f")
    Sigma_m_mp    = Sigma(omega_m, omega_fine, Gamma_tilde_fine,
                                               Gamma_tilde_fine, nrf)
    Sigma_m_mp   += -1j * Gamma_m_tilde
    Sigma_m_mp   +=  1j * Gamma_m_tilde_neg

    #print ("g")
    Sigma_m       = Sigma(omega_m, omega_fine,
                          Gamma_fine, Gamma_prime_fine, nrf).real
    Sigma_m_Gamma = Sigma(omega_m, omega_fine, Gamma_fine,
                          0.0 * Gamma_prime_fine, nrf).real
    Sigma_m_prime    = Sigma_m - Sigma_m_Gamma
    Sigma_spl        = interpolate.splrep(omega_m, Sigma_m)
    
    def Sigma_s(omega):
        return interpolate.splev(omega, Sigma_spl)
    
    def dSigma_s(omega):
        return interpolate.splev(omega, Sigma_spl, 1)
    
    def f_res(omega_r):
        return omega_r - Omega_0 - Sigma_s(omega_r)
    
    print ("Omega_0 = ",   Omega_0   / constants.GHz_2pi)
    Omega_res = Omega_0
    
    try:
        Omega_res = solve (f_res, omega_min, omega_max)
    except:
        import traceback
        traceback.print_exc()
        print ("** Solving for resonance failed, bail out")

    print ("Gamma_0 = ", Gamma_0)
    print ("Omega_res = ", Omega_res / constants.GHz_2pi)
    Gamma_res = Gamma_s(Omega_res) + Gamma_0
    print ("Gamma_res_0 = ", Gamma_res)
    dSigma_res = dSigma_s(Omega_res)
    Z_res = 1.0 / (1.0 - dSigma_res)
    print ("Z_res = ", Z_res)
    print ("Gamma_res_Z = ", Z_res * Gamma_res)

    new_fname = fname[:-4] + "+analysis3.npz"
    d_new = dict()
    d_new.update(**d)
    d_new['s'] = get_s(d, s)
    d_new['d_film'] = get_d_film(d, d_film)
    d_new['Omega_0']       = Omega_0
    d_new['Gamma_0']       = Gamma_0
    d_new['omega_m']       = omega_m
    d_new['Gamma_m']       = Gamma_m
    d_new['Gamma_m_prime'] = Gamma_m_prime
    d_new['Gamma_m_tilde'] = Gamma_m_tilde
    d_new['Sigma_m']       = Sigma_m
    d_new['Sigma_m_prime'] = Sigma_m_prime
    d_new['Omega_res']     = Omega_res
    d_new['Gamma_res']     = Gamma_res
    d_new['omega_min']     = omega_min
    d_new['Z_res']         = Z_res
    #d_new['Sigma_m_prime'] = Sigma_m_prime
    d_new['Sigma_m_Gamma'] = Sigma_m_Gamma
    d_new['Sigma_m_pp'] = Sigma_m_pp
    d_new['Sigma_m_mm'] = Sigma_m_mm
    d_new['Sigma_m_mp'] = Sigma_m_mp
    d_new['Sigma_m_pm'] = Sigma_m_pm

    if not debug: theta_vals = np.linspace(0.0, 2.0 * np.pi, 361)
    else:         theta_vals = np.linspace(0.0, 2.0 * np.pi, 37)
    tab_scat = tabulate_scattering(sc_problem, Omega_res, theta_vals)
    for key in tab_scat.keys():
        new_key = 'scat:' + key
        d_new[new_key] = tab_scat[key]

    if not debug: N_theta = 12
    else:         N_theta = 2
    theta_inc = np.linspace(0.0, 2.0 * np.pi, N_theta + 1)
    #tab_inc = tabulate_inc(sc_problem, omega_m, omega_min, theta_inc)
    tab_inc = tabulate_inc(sc_problem, omega_coarse[1:], omega_min, theta_inc)
    for key in tab_inc.keys():
        if key == 'omega':
            tab_inc[key] = omega_coarse[0:]
            continue
        if key == 'theta_inc':
            continue
        N, M = np.shape(tab_inc[key])
        tab_res = []
        print ("insert zero into: ", key, np.shape(tab_inc[key]))
        for j in range(M):
            tab_list = list(tab_inc[key][:, j])
            tab_list.insert(0, 0.0)
            tab_res.append(tab_list)
        tab_inc[key] = np.transpose(np.array(tab_res))
        print ("result: ", np.shape(tab_inc[key]))

    def do_transfer(omega_coarse, data_coarse, omega_m, omega_min, do_cmplx):
        result = np.zeros((len(omega_m)), dtype=data_coarse.dtype)
        print ("data_coarse: ", np.shape(data_coarse), "omega_coarse:",
               np.shape(omega_coarse))
        data_spl_re = interpolate.splrep(omega_coarse, data_coarse.real)
        data_spl_im = interpolate.splrep(omega_coarse, data_coarse.imag)
        if do_cmplx:
          def _data_s(omega):
            if omega < omega_min: return 0.0 + 0.0j
            ret  = 0.0 + 0.0j
            ret +=      interpolate.splev(omega, data_spl_re)
            ret += 1j * interpolate.splev(omega, data_spl_im)
            return ret
        else :
          def _data_s(omega):
            if omega < omega_min: return 0.0
            return interpolate.splev(omega, data_spl_re)
        data_s = np.vectorize(_data_s)
        data_m = data_s(omega_m)
        return data_m
        
    def transfer_to_fine(omega_coarse, data_coarse, omega_m, omega_min):
        N, M = np.shape(data_coarse)
        result = np.zeros((len(omega_m), M), dtype=data_coarse.dtype)
        do_cmplx = (linalg.norm(data_coarse.imag) > 1e-10)
        for j in range(M):
            result[:, j] = do_transfer(omega_coarse, data_coarse[:, j],
                                       omega_m, omega_min, do_cmplx)
        print ("do_transfer:", data_coarse, result)
        return result
            
    for key in tab_inc.keys():
        new_key = 'inc:' + key
        if key == 'omega':
            d_new[new_key] = omega_m
            continue
        if key == 'theta_inc':
            d_new[new_key] = tab_inc[key]
            continue
        print ("handle", key, new_key)
        d_new[new_key] = transfer_to_fine(tab_inc['omega'],
                                          tab_inc[key], omega_m, omega_min)
        d_new['inc:omega'] = omega_m

    #np.savez(new_fname, **d_new)
    np.savez(fname_analysis, **d_new)

    #pl.figure()
    peak_0 = 1.0 / (omega_m - Omega_0 + 1j * Gamma_0 + 1j * Gamma_m - Sigma_m)
    M_pp = omega_m - Omega_0 + 1j * Gamma_0 - Sigma_m_pp
    M_mm = omega_m + Omega_0 + 1j * Gamma_0 + Sigma_m_mm
    M_pm = - Sigma_m_pm
    M_mp =   Sigma_m_mp
    det_M = M_pp * M_mm - M_pm * M_mp
    peak_1 = 1.0 / det_M
    #pl.plot(omega_m / constants.GHz_2pi, np.abs(peak_0)/np.max(np.abs(peak_0)))
    #pl.plot(omega_m / constants.GHz_2pi, np.abs(peak_1)/np.max(np.abs(peak_1)))
    #if debug: pl.show()
    

for fname in sys.argv[1:]:
    if fname.find('analysis') >=0: continue
    import os.path
    f_base = os.path.basename(fname)
    if f_base[:6] == 'BARE3d':
        print ("BARE MODE: tabulate for s and d")
        for s in [5.0 * constants.nm, 10.0 * constants.nm,
                  15.0 * constants.nm, 30.0 * constants.nm,
                  40.0 * constants.nm]:
                 #[20.0 * constants.nm]:
            for d_film in [20.0 * constants.nm, 50.0 * constants.nm]:
                base = fname[:-4]
                fname_analysis = "%s-s=%gnm-d=%gnm+analysis3.npz" % (base,
                                                         s / constants.nm,
                                                         d_film/constants.nm)
                print ("s = ", s / constants.nm, "d = ", d_film/constants.nm,
                       "->", fname_analysis)
                process_data(fname, fname_analysis, s, d_film, False)
    elif f_base[:5] == 'NRF3d':
        base = fname[:-4]
        fname_analysis = "%s+analysis3.npz" % base
        print ("NRF mode, extract s and d ->", fname_analysis)
        process_data(fname, fname_analysis, -1, -1, True)
        

#pl.show()
