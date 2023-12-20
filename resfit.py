import numpy as np
import pylab as pl
import sys
from scipy import optimize
import random

from constants import GHz_2pi

def R_res(omega, Omega_0, Gamma_R, Gamma_L, Gamma_0):
    return 2.0*np.sqrt(Gamma_R * Gamma_L) / (omega - Omega_0 + 1j * (Gamma_R + Gamma_L + Gamma_0))

def T_res(omega, Omega_0, Gamma_R, Gamma_0L):
    return 1.0 - 2.0j * Gamma_R / (omega - Omega_0 + 1j * (Gamma_R + Gamma_0L))

def T_res_phi(omega, Omega_0, Gamma_R, Gamma_0L, phi_0):
    return T_res(omega, Omega_0, Gamma_R, Gamma_0L) * np.exp(1j * phi_0)

def R_Fano(omega, Omega_0, Gamma_R, Gamma_L, Gamma_0, R0):
    print (np.shape(R0), np.shape(omega))
    return R0 + R_res(omega, Omega_0, Gamma_R, Gamma_L, Gamma_0)

def T_Fano(omega, Omega_0, Gamma_R, Gamma_L, Gamma_0, T0):
    print (np.shape(R0), np.shape(omega))
    return T0 - 1.0 + T_res(omega, Omega_0, Gamma_R, Gamma_L, Gamma_0)

def fit_R_Fano(omega_min, omega_max, omega, R_omega, Omega_0, Gamma_R, Gamma_L, Gamma_0):
    i_fit   = [t for t in range(len(omega))
               if omega[t] > omega_min and omega[t] < omega_max]
    o_fit   = np.array([omega[t]   for t in i_fit])
    R_fit = np.array([R_omega[t] for t in i_fit])

    def f_Fano(o, R0_re, R0_im):
        return np.abs(R_Fano(o, Omega_0, Gamma_R, Gamma_L, Gamma_0, R0_re + 1j * R0_im))

    p_opt, p_cov = optimize.curve_fit(f_Fano, o_fit, np.abs(R_fit), bounds = [(-0.1, -0.1), (0.1, 0.1)])
    print ("fit r fano: ", p_opt, p_cov)
    return p_opt[0] + 1j * p_opt[1]

def fit_T_Fano(omega_min, omega_max, omega, T_omega):
    i_fit   = [t for t in range(len(omega))
               if omega[t] > omega_min and omega[t] < omega_max]
    o_fit   = np.array([omega[t]   for t in i_fit])
    T_fit = np.array([T_omega[t] for t in i_fit])

    def f_Fano(o, R0_re, R0_im, Omega_0, Gamma_R, Gamma_L, Gamma_0):
        print ("f_fano: ", R0_re, R0_im)
        return np.abs(T_Fano(o, Omega_0, Gamma_R, Gamma_L, Gamma_0, R0_re + 1j * R0_im))
    print (np.shape(o_fit), np.shape(R_fit))

    p_opt, p_cov = optimize.curve_fit(f_Fano, o_fit, np.abs(T_fit))
    T_0 =  p_opt[0] + 1j * p_opt[1]
    Omega_0 = p_opt[2]
    Gamma_R = p_opt[3]
    Gamma_L = p_opt[3]
    Gamma_0 = p_opt[4]
    Gamma_tot = Gamma_R + Gamma_L + Gamma_0

    return Omega_0, Gamma_R, Gamma_L, Gamma_0, Gamma_tot
    

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
        r_1 = np.sum(np.abs(np.abs(T1_o) - np.abs(T1_fit))**2)
        r_2 = np.sum(np.abs(np.abs(T2_o) - np.abs(T2_fit))**2)
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

    if len(o_fit) < 5: return 0.0, 0.0, 0.0, 0.0
    
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

    if len(o_fit) < 5: return 0.0, 0.0, 0.0, 0.0, 0.0

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

