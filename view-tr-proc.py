import numpy as np
import pylab as pl
import sys
from scipy import optimize
import random

from constants import GHz_2pi

def R_res(omega, Omega_0, Gamma_R, Gamma_L, Gamma_0):
    return 2.0 *np.sqrt(Gamma_R * Gamma_L) / (omega - Omega_0 + 1j * (Gamma_R + Gamma_L + Gamma_0))

def T_res(omega, Omega_0, Gamma_R, Gamma_0L):
    return 1.0 - 2.0j * Gamma_R / (omega - Omega_0 + 1j * (Gamma_R + Gamma_0L))

def T_res_phi(omega, Omega_0, Gamma_R, Gamma_0L, phi_0):
    return T_res(omega, Omega_0, Gamma_R, Gamma_0L) * np.exp(1j * phi_0)

def fit_T_resonance_with_phase(omega_min, omega_max, omega, T_omega):
    i_fit   = [t for t in range(len(omega))
               if omega[t] > omega_min and omega[t] < omega_max]
    o_fit   = np.array([omega[t]   for t in i_fit])
    T_fit = np.array([T_omega[t] for t in i_fit])
    def residue(p):
        Omega_0  = p[0]
        Gamma_R  = p[1]
        Gamma_0L = p[2]
        Phi_0    = p[3]
        print (p)
        r = T_fit - T_res_phi(o_fit, Omega_0, Gamma_R, Gamma_0L, Phi_0) 
        return np.sum(np.abs(r)**2)

    gamma_max = 2.0 * (omega_max - omega_min)
    p0 = np.array([0.5 * (omega_min + omega_max),
                   gamma_max / 2.0, gamma_max / 2.0, 0.0])
    result = optimize.minimize(residue, p0,
                               bounds=((omega_min, omega_max),
                                       (0, gamma_max), (0, gamma_max),
                                       (-np.pi, np.pi)),
                               options=dict(maxiter = 10000, maxfun = 10000))
    print ("result: ", result)
    p_res = result['x']
    Omega_0 = p_res[0]
    Gamma_R = p_res[1]
    Gamma_0L = p_res[2]
    Gamma_tot = Gamma_R + Gamma_0L
    phi_0 = p_res[3]

    return Omega_0, Gamma_R, Gamma_0L, Gamma_tot, phi_0

def fit_T_and_R (omega_min, omega_max, omega, T_omega, R_omega):
    i_fit   = [t for t in range(len(omega))
               if omega[t] > omega_min and omega[t] < omega_max]
    o_fit   = np.array([omega[t]   for t in i_fit])
    T_fit = np.array([T_omega[t] for t in i_fit])
    R_fit = np.array([R_omega[t] for t in i_fit])


    def fmin(p):
        Omega_0 = p[0]
        Gamma_R = p[1]
        Gamma_L = p[2]
        Gamma_0 = p[3]
        R_o = R_res(o_fit, Omega_0, Gamma_R, Gamma_L, Gamma_0)
        T_o = T_res(o_fit, Omega_0, Gamma_R, Gamma_0 + Gamma_L)
        r_R = np.sum(np.abs(R_o - R_fit)**2)
        r_T = np.sum(np.abs(T_o - T_fit)**2)
        print ("p = ", p, "r = ", r_R, r_T)
        return r_R + r_T 

    gamma_max = (omega_max - omega_min) * 1.5
    p0 = np.array([0.5 * (omega_max + omega_min), 0.1 * gamma_max,
                   0.1 * gamma_max, 0.1 * gamma_max])

    result = optimize.minimize(fmin, p0, bounds=((omega_min, omega_max),
                                                 (0.0, gamma_max),
                                                 (0.0, gamma_max),
                                                 (0.0, gamma_max)),
                               options=dict(maxiter=10000, maxfev=10000),
                               tol=1e-10, method='TNC')
    print ("fit result: ", result)
    p0 = result['x']
    Omega_0 = p0[0]
    Gamma_R = p0[1]
    Gamma_L = p0[2]
    Gamma_0 = p0[3]
    Gamma_tot = Gamma_0 + Gamma_L + Gamma_R
    
    return Omega_0, Gamma_R, Gamma_L, Gamma_0, Gamma_tot

def fit_T_and_T (omega_min, omega_max, omega, T1_omega, T2_omega):
    i_fit   = [t for t in range(len(omega))
               if omega[t] > omega_min and omega[t] < omega_max]
    o_fit   = np.array([omega[t]   for t in i_fit])
    T1_fit = np.array([T1_omega[t] for t in i_fit])
    T2_fit = np.array([T2_omega[t] for t in i_fit])


    def fmin(p):
        Omega_0 = p[0]
        Gamma_R = p[1]
        Gamma_L = p[2]
        Gamma_0 = p[3]
        T1_o = T_res(o_fit, Omega_0, Gamma_R, Gamma_L + Gamma_0)
        T2_o = T_res(o_fit, Omega_0, Gamma_L, Gamma_R + Gamma_0)
        r_1 = np.sum(np.abs(T1_o - T1_fit)**2)
        r_2 = np.sum(np.abs(T2_o - T2_fit)**2)
        #print ("p = ", p, "r = ", r_R, r_T)
        return r_1 + r_2

    gamma_max = (omega_max - omega_min) * 2
    p0 = np.array([0.5 * (omega_max + omega_min), 0.1 * gamma_max,
                   0.1 * gamma_max, 0.1 * gamma_max])

    result = optimize.minimize(fmin, p0, bounds=((omega_min, omega_max),
                                                 (0.0, gamma_max),
                                                 (0.0, gamma_max),
                                                 (0.0, gamma_max)),
                               options=dict(maxiter=10000, maxfev=10000),
                               tol=1e-10, method='TNC')
    print ("fit result: ", result)
    p0 = result['x']
    Omega_0 = p0[0]
    Gamma_R = p0[1]
    Gamma_L = p0[2]
    Gamma_0 = p0[3]
    Gamma_tot = Gamma_0 + Gamma_L + Gamma_R
    
    return Omega_0, Gamma_R, Gamma_L, Gamma_0, Gamma_tot

def fit_T2_resonance(omega_min, omega_max, omega, T_omega):

    i_fit   = [t for t in range(len(omega))
               if omega[t] > omega_min and omega[t] < omega_max]
    o_fit   = np.array([omega[t]   for t in i_fit])
    T_fit = np.array([T_omega[t] for t in i_fit])

    gamma_max = 2 * (omega_max - omega_min)

    def T2(o, Omega_0, Gamma_R, Gamma_0L):
        return np.abs(T_res(o, Omega_0, Gamma_R, Gamma_0L))**2
    p_opt, p_cov = optimize.curve_fit(T2, o_fit, np.abs(T_fit)**2,
                                      bounds=[(omega_min, 0.0, 0.0),
                                        (omega_max, gamma_max, gamma_max)])
    print ("curve_fit: ", p_opt, p_cov)
    Omega_0  = p_opt[0]
    Gamma_R  = p_opt[1]
    Gamma_0L = p_opt[2]

    Gamma_tot = Gamma_R + Gamma_0L

    return Omega_0, Gamma_R, Gamma_0L, Gamma_tot


def fit_R2_resonance(omega_min, omega_max, omega, R_omega):

    i_fit   = [t for t in range(len(omega))
               if omega[t] > omega_min and omega[t] < omega_max]
    o_fit   = np.array([omega[t]   for t in i_fit])
    R_fit = np.array([R_omega[t] for t in i_fit])

    gamma_max = 2 * (omega_max - omega_min)

    def R2(o, Omega_0, Gamma_R, Gamma_L, Gamma_0):
        return np.abs(R_res(o, Omega_0, Gamma_R, Gamma_L, Gamma_0))**2
    p_opt, p_cov = optimize.curve_fit(R2, o_fit, np.abs(R_fit)**2,
                                      bounds=[(omega_min, 0.0, 0.0, 0.0),
                                        (omega_max, gamma_max, gamma_max,
                                         gamma_max)])
    print ("curve_fit: ", p_opt, p_cov)
    Omega_0 = p_opt[0]
    Gamma_R = p_opt[1]
    Gamma_L = p_opt[2]
    Gamma_0 = p_opt[3]

    Gamma_tot = Gamma_R + Gamma_L + Gamma_0

    return Omega_0, Gamma_R, Gamma_L, Gamma_0, Gamma_tot


if False:
    omega = np.linspace(0.0, 2.0, 1001)
    gamma_0L_test = 0.01
    gamma_R_test  = 0.07
    omega_0_test =  1.03
    T_test = T_res(omega, omega_0_test, gamma_R_test, gamma_0L_test)

    Omega_0, Gamma_R, Gamma_0L, Gamma_tot = fit_T2_resonance(0.8, 1.2,
                                                             omega, T_test)
    print ("Test: o   = ", Omega_0, omega_0_test)
    print ("      G_R = ", Gamma_R, gamma_R_test)
    print ("      G_R = ", Gamma_0L, gamma_0L_test)

    pl.figure()
    pl.plot(omega, np.abs(T_test), label='test')
    pl.plot(omega, np.abs(T_res(omega, Omega_0, Gamma_R, Gamma_0L)), '--',
            label='fit')
    pl.legend()

    Omega_1, Gamma_R1, Gamma_1L, Gamma_tot1, phase_1 = fit_T_resonance_with_phase(0.7, 1.3, omega, T_test)

    print ("with phase: Omega1 = ", Omega_1)
    print ("            Gamma_R  = ", Gamma_R1)
    print ("            Gamma_0L = ", Gamma_1L)
    print ("            phi_0   = ", phase_1)

    T_fit1 = T_res_phi(omega, Omega_1, Gamma_R1, Gamma_1L, phase_1)
    pl.figure()
    pl.plot(omega, T_test.real, label='Re test')
    pl.plot(omega, T_test.imag, label='Im test')
    pl.plot(omega, T_fit1.real, '--', label='Re fit')
    pl.plot(omega, T_fit1.imag, '--', label='Im fit')
    pl.legend()

    pl.show()


def readData(fname):
    d = np.load(fname)
    for k in d.keys(): print (k)
    for k in ['a', 'b', 's', 'd']:
        if k in d.keys(): print (k, " = ", d[k])
    omega = d['omega']
    T = d['T1']
    R = d['R1']
    k = 0.0 * omega
    if 'k' in d.keys():
        k = d['k'] #0.0 * omega + 0.0
    k_bk = 0.0 * omega + 0.0
    err = d['err'] #0.0 * np.abs(T)
    if 'T1_bk' in d.keys():
       T_bk = d['T1_bk']
       R_bk = d['R1_bk']
       #err_bk = 0.0 * np.abs(T)
       err_bk = d['err_bk']
       if 'k_bk' in d.keys(): k_bk = d['k_bk']
    pl.figure()
    p_t = pl.plot(omega / GHz_2pi, np.abs(T), label='|T| forw')
    pl.plot(omega / GHz_2pi, err, label='err fw')
    p_r = pl.plot(omega / GHz_2pi, np.abs(R), label='|R| forw')
    #pl.legend()
    if 'T_bk' in d.keys():
        pl.plot(omega / GHz_2pi, np.abs(T_bk), '--', color=p_t[0].get_color())
        pl.plot(omega / GHz_2pi, np.abs(R_bk), '--', color=p_r[0].get_color())
        pl.plot(omega / GHz_2pi, err, label='err bk')
    pl.legend()
    ax_angle = pl.gca().twinx()
    ax_angle.plot (omega / GHz_2pi, np.angle(T) * 180.0/np.pi,
                   color='m', label='arg T forw')
    if 'T_bk' in d.keys():
        ax_angle.plot (omega / GHz_2pi, np.angle(T_bk) * 180.0/np.pi,
                       'c')
    ax_angle.set_ylim(-180.0, 180.0)
    pl.legend()
    pl.xlabel("Frequency $f$")

    pl.figure()
    p_t = pl.plot(omega / GHz_2pi, np.abs(T), label='|T| forw')
    pl.plot(omega / GHz_2pi, err, label='err fw')
    p_r = pl.plot(omega / GHz_2pi, np.abs(R), label='|R| forw')
    pl.legend()
    ax_T = pl.gca()
    ax_angle = pl.gca().twinx()
    ax_angle.plot (omega / GHz_2pi, np.angle(T) * 180.0/np.pi,
                   color='m', label='arg T forw')
    ax_angle.set_ylim(-180.0, 180.0)
    pl.legend()
    pl.xlabel("Frequency $f$")
    pl.title("Forward")

    #res_min = 3.10 * GHz_2pi
    #res_max = 3.25 * GHz_2pi
    res_min = 3.02 * GHz_2pi
    res_max = 3.15 * GHz_2pi
    #res_min = 4.05 * GHz_2pi
    #res_max = 4.10 * GHz_2pi
    #res_min = 3.90 * GHz_2pi
    #res_max = 4.10 * GHz_2pi

    show_omega = np.array([1.6, 2.34, 3.1, 4.01]) * GHz_2pi

    for o in show_omega:
        i_omega = np.argmin(np.abs(o - d['omega']))
        pl.figure()
        pl.plot(d['xr'], d['mode_r_mx'][i_omega][:].real, label='Re mx')
        pl.plot(d['xr'], d['mode_r_mx'][i_omega][:].imag, label='Im mx')
        pl.plot(d['xr'], d['mode_r_mz'][i_omega][:].real, label='Re mz')
        pl.plot(d['xr'], d['mode_r_mz'][i_omega][:].imag, label='Im mz')
        pl.plot(d['xr'], np.abs(d['mode_r_mx'][i_omega]), label='|m_x|')
        pl.plot(d['xr'], np.abs(d['mode_r_mz'][i_omega]), label='|m_z|')
        pl.legend()
        pl.title("f = %g ---> " % (o / GHz_2pi))
    if 'T_bk' in d.keys():
      for o in show_omega:
        i_omega = np.argmin(np.abs(o - d['omega']))
        pl.figure()
        pl.plot(d['xr'], d['mode_r_mx_bk'][i_omega][:].real, label='Re mx')
        pl.plot(d['xr'], d['mode_r_mx_bk'][i_omega][:].imag, label='Im mx')
        pl.plot(d['xr'], d['mode_r_mz_bk'][i_omega][:].real, label='Re mz')
        pl.plot(d['xr'], d['mode_r_mz_bk'][i_omega][:].imag, label='Im mz')
        pl.plot(d['xr'], np.abs(d['mode_r_mx_bk'][i_omega]), label='|m_x|')
        pl.plot(d['xr'], np.abs(d['mode_r_mz_bk'][i_omega]), label='|m_z|')
        pl.legend()
        pl.title("f = %g <-- " % (o / GHz_2pi))
        
    
    if True:
        Omega_0, Gamma_R, Gamma_0L, Gamma_tot = fit_T2_resonance(res_min,
                                                                 res_max,
                                                                 omega,
                                                                 T)
        print ("fit transmission: freq = ", Omega_0 / GHz_2pi,
               "width = ", Gamma_tot)
        print ("Gamma_R = ", Gamma_R, "left: ", Gamma_0L)
        T_fit = T_res(omega, Omega_0, Gamma_R, Gamma_0L)
        ax_T.plot(omega / GHz_2pi, np.abs(T_fit), '--', label='T fit')
        Omega_0, Gamma_R, Gamma_0L, Gamma_tot, phase_R = fit_T_resonance_with_phase(res_min,
                                                                 res_max,
                                                                 omega,
                                                                 T)
        print ("fit transmission: freq = ", Omega_0 / GHz_2pi,
               "width = ", Gamma_tot)
        print ("Gamma_R = ", Gamma_R, "left: ", Gamma_0L)
        T_fit = T_res(omega, Omega_0, Gamma_R, Gamma_0L)
        ax_T.plot(omega / GHz_2pi, np.abs(T_fit), '--', label='T fit')
    if True:
        Omega_1, Gamma_R1, Gamma_L1, Gamma_1, Gamma_tot1 = fit_R2_resonance(res_min,
                                                                 res_max,
                                                                 omega, R)
        print ("fit reflection: Omega = ", Omega_1 / GHz_2pi,
               "width = ", Gamma_tot1)
        print ("GammaR = ", Gamma_R1, "Gamma_L = ",
               Gamma_L1, "Gamma_0 = ", Gamma_1)

        R_fit = R_res(omega, Omega_1, Gamma_R1, Gamma_L1, Gamma_1)
        ax_T.plot(omega / GHz_2pi, np.abs(R_fit), '--', label='R fit')
        
    if False:
        Omega_2, Gamma_R2, Gamma_L2, Gamma_2, Gamma_tot2 = fit_T_and_R(
               res_min, res_max, omega, T, R)
        print ("fit T and R: Omega = ", Omega_2 / GHz_2pi,
               "width = ", Gamma_tot2)
        print ("GammaR = ", Gamma_R2, "Gamma_L = ",
               Gamma_L2, "Gamma_0 = ", Gamma_2)

        R2_fit = R_res(omega, Omega_2, Gamma_R2, Gamma_L2, Gamma_2)
        ax_T.plot(omega / GHz_2pi, np.abs(R2_fit), '--', label='R fit')
        T2_fit = T_res(omega, Omega_2, Gamma_R2, Gamma_L2 + Gamma_2)
        ax_T.plot(omega / GHz_2pi, np.abs(T2_fit), '--', label='T fit')

    
    
    if 'T_bk' in d.keys():
        pl.figure()
        p_t = pl.plot(omega / GHz_2pi, np.abs(T_bk), label='|T| back')
        pl.plot(omega / GHz_2pi, err_bk, label='err back')
        p_r = pl.plot(omega / GHz_2pi, np.abs(R_bk), label='|R| back')
        pl.legend()
        ax_Tbk = pl.gca()
        ax_angle = pl.gca().twinx()
        ax_angle.plot (omega / GHz_2pi, np.angle(T_bk) * 180.0/np.pi,
                       color='m', label='arg T back')
        ax_angle.set_ylim(-180.0, 180.0)
        pl.legend()
        pl.xlabel("Frequency $f$")
        pl.title("Backward")
        if False:
          Omega_0, Gamma_R, Gamma_0L, Gamma_tot = fit_T2_resonance(res_min,
                                                                 res_max,
                                                                 omega,
                                                                   T_bk)
          print ("fit transmission: freq = ", Omega_0 / GHz_2pi,
               "width = ", Gamma_tot)
          print ("Gamma_R = ", Gamma_R, "left: ", Gamma_0L)
          T_fit = T_res(omega, Omega_0, Gamma_R, Gamma_0L)
          ax_Tbk.plot(omega / GHz_2pi, np.abs(T_fit), '--', label='T fit')
          Omega_0, Gamma_R, Gamma_0L, Gamma_tot, phase_L = fit_T_resonance_with_phase(res_min,
                                                                 res_max,
                                                                 omega,
                                                                                      T_bk)
        print ("fit transmission: freq = ", Omega_0 / GHz_2pi,
               "width = ", Gamma_tot)
        print ("Gamma_R = ", Gamma_R, "left: ", Gamma_0L)
        T_fit = T_res(omega, Omega_0, Gamma_R, Gamma_0L)
        ax_Tbk.plot(omega / GHz_2pi, np.abs(T_fit), '--', label='T fit')
        if True:
          Omega_1, Gamma_R1, Gamma_L1, Gamma_1, Gamma_tot1 = fit_R2_resonance(
              res_min,
                                                                 res_max,
                                                                omega, R_bk)
          print ("fit reflection: Omega = ", Omega_1 / GHz_2pi,
               "width = ", Gamma_tot1)
          print ("GammaR = ", Gamma_R1, "Gamma_L = ",
               Gamma_L1, "Gamma_0 = ", Gamma_1)

          R_fit = R_res(omega, Omega_1, Gamma_R1, Gamma_L1, Gamma_1)
          ax_Tbk.plot(omega / GHz_2pi, np.abs(R_fit), '--', label='R fit')
        if True:
          Omega_3, Gamma_R3, Gamma_L3, Gamma_3, Gamma_tot3 = fit_T_and_T(
            res_min, res_max, omega, T, T_bk)
          print ("T and T fit: freq = ", Omega_3 / GHz_2pi, "width = ",
               Gamma_tot3)
          print ("Gamma_R = ", Gamma_R3, "Gamma_L = ", Gamma_L3,
               "Gamma_0 = ", Gamma_3)
          T1_fit = T_res(omega, Omega_3, Gamma_R3, Gamma_L3 + Gamma_3)
          T2_fit = T_res(omega, Omega_3, Gamma_L3, Gamma_R3 + Gamma_3)
          R1_fit = R_res(omega, Omega_3, Gamma_R3, Gamma_L3, Gamma_3)
          R2_fit = R_res(omega, Omega_3, Gamma_L3, Gamma_R3, Gamma_3)
          ax_T.plot(omega / GHz_2pi, np.abs(T1_fit), label='T+T fit T')
          ax_T.plot(omega / GHz_2pi, np.abs(R1_fit), label='T+T fit R')
          ax_Tbk.plot(omega / GHz_2pi, np.abs(T2_fit), label='T+T fit T')
          ax_Tbk.plot(omega / GHz_2pi, np.abs(R2_fit), label='T+T fit R')
          ax_T.legend()
          ax_Tbk.legend()

    pl.figure()
    pl.plot(k, omega / GHz_2pi)
    pl.plot(k_bk, omega / GHz_2pi, '--')
    pl.xlabel(r"wavenumber $k$")
    pl.ylabel(r"Frequency $f$")
    #pl.figure()
    #pl.plot(omega / GHz_2pi, np.angle(T))

    pl.figure()
    pl.plot(T.real, T.imag)
    pl.gca().set_aspect('equal', 'box')
    if 'T1_bk' in d.keys():
        pl.plot(T_bk.real, T_bk.imag)
        
    pl.show()

    


for fname in sys.argv[1:]:
    readData(fname)
