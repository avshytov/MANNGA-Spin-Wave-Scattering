import numpy as np
import pylab as pl
import sys
from scipy import optimize
import random

from constants import GHz_2pi

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
    Omega_0 = p_opt[0]
    Gamma_R = p_opt[1]
    Gamma_0L = p_opt[2]

    Gamma_tot = Gamma_R + Gamma_0L

    return Omega_0, Gamma_R, Gamma_0L, Gamma_tot


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

def f_wave_fit(x, k, A, B, k1):
    return A * np.exp(1j * k * x - k1 * x) + B * np.exp(-1j * k * x + k1 * x)

def get_scat_solution(x_in_a, x_in_b, x_out_a, x_out_b, x, psi,
                      k_min = 0.0, k_max = 2.0, k_approx = 1.0):
    if x_in_a > x_in_b:
        x_in_a, x_in_b = x_in_b, x_in_a
    if x_out_a > x_out_b:
        x_out_a, x_out_b = x_out_b, x_out_a
    
    in_fit   = [t for t in range(len(x)) if x[t] > x_in_a and x[t] < x_in_b]
    out_fit   = [t for t in range(len(x)) if x[t] > x_out_a and x[t] < x_out_b]
    x_in   = np.array([x[t]   for t in in_fit])
    psi_in = np.array([psi[t] for t in in_fit])
    x_out   = np.array([x[t]   for t in out_fit])
    psi_out = np.array([psi[t] for t in out_fit])

    C_in = np.sum(np.abs(psi_in)**2)
    C_out = np.sum(np.abs(psi_out)**2)

    print ("len(psi_in), len(psi_out)", len(psi_in), len(psi_out))

    def f_opt(x):
        k = x[0];
        A = x[1] + 1j * x[2]
        B = x[3] + 1j * x[4]
        C = x[5] + 1j * x[6]
        k1 = x[7]
        #psi_test = A * np.exp(1j * k * x_fit) + B * np.exp(-1j * k * x_fit)
        psi_in_fit  = f_wave_fit(x_in,  k, A, B,   k1)
        psi_out_fit = f_wave_fit(x_out, k, C, 0.0, k1)
        r_in = np.sum(np.abs(psi_in_fit - psi_in)**2) / C_in
        r_out = np.sum(np.abs(psi_out_fit - psi_out)**2) / C_out
        return r_in + r_out

    print ("k min, max", k_min, k_max)
    #k_approx = 0.5 * (k_min + k_max)
    results = []
    max_it = 10
    n_it = 0
    while n_it < max_it:
      n_it += 1
      rnd_0 = 0.5 * (2 * random.random() - 1.0) * (k_max - k_min)
      if n_it == 1: rnd_0 = 0.0
      rnd_1 = 0.5 * (2 * random.random() - 1.0)
      rnd_2 = 0.5 * (2 * random.random() - 1.0)
      rnd_3 = 0.5 * (2 * random.random() - 1.0)
      rnd_4 = 0.5 * (2 * random.random() - 1.0)
      x0 = np.array([k_approx + rnd_0,
                               1.0 + rnd_1, 0.0 + rnd_2,
                               1.0 + rnd_3, 0.0 + rnd_4, 
                               0.0 + rnd_3, 1.0 + rnd_4,
                               0.0001])
      result = optimize.minimize(f_opt, x0, tol=1e-13, method='Powell',
            bounds=((k_min, k_max),
                    (-4, 4), (-4, 4), (-4, 4), (-4, 4),  (-4, 4),  (-4, 4),
                    (0.0, 0.01)))
      print ("fit: ", result)
      results.append(result)
      if result['fun'] < 0.005: break
    results.sort(key = lambda x: x['fun'])
    result = results[0]
    print ("best: ", result)
    k_fit, A_fit_re, A_fit_im, B_fit_re, B_fit_im, C_fit_re, C_fit_im, k1 = result['x']
    A_fit = A_fit_re + 1j * A_fit_im
    B_fit = B_fit_re + 1j * B_fit_im
    C_fit = C_fit_re + 1j * C_fit_im

    #if k_fit < 0:
    #    A_fit, B_fit = B_fit, A_fit
    #    k_fit *= -1
    len_fit = len(x_in) + len(x_out)
    return k_fit, A_fit, B_fit, C_fit, k1, np.sqrt(result['fun'])


def get_amplitudes(x_a, x_b, x, psi, k_min = 0.0, k_max = 2.0, k_approx = 1.0):
    i_fit   = [t for t in range(len(x)) if x[t] > x_a and x[t] < x_b]
    x_fit   = np.array([x[t]   for t in i_fit])
    psi_fit = np.array([psi[t] for t in i_fit])

    if False:
        from scipy.fft import fft, fftfreq, fftshift
        fft_psi  = fft(psi_fit)
        fft_freq = fftfreq( len(fft_psi), x_fit[1] - x_fit[0] )
        fft_freq = fftshift( fft_freq )
        fft_psi  = fftshift( fft_psi )
        pl.figure()
        pl.semilogy(fft_freq * 2.0 * np.pi, np.abs(fft_psi)**2)
        fft_max = np.argmax(np.abs(fft_psi))
        k_fft_max = fft_freq[fft_max] * 2.0 * np.pi
        print ("k_fft_max = ", k_fft_max)
        if k_fft_max < 0: k_fft_max *= -1
        print ("k_fft_max = ", k_fft_max)


    C = 1.0/np.sum(np.abs(psi_fit)**2)
    def f_opt(x):
        k = x[0]; A = x[1] + 1j * x[2]; B = x[3] + 1j * x[4]
        k1 = x[5]
        #psi_test = A * np.exp(1j * k * x_fit) + B * np.exp(-1j * k * x_fit)
        return C * np.sum(np.abs(f_wave_fit(x_fit, k, A, B, k1) - psi_fit)**2)

    print ("k min, max", k_min, k_max)
    #k_approx = 0.5 * (k_min + k_max)
    results = []
    max_it = 10
    n_it = 0
    while n_it < max_it:
      n_it += 1
      rnd_0 = 0.5 * (2 * random.random() - 1.0) * (k_max - k_min)
      if n_it == 1: rnd_0 = 0.0
      rnd_1 = 0.5 * (2 * random.random() - 1.0)
      rnd_2 = 0.5 * (2 * random.random() - 1.0)
      rnd_3 = 0.5 * (2 * random.random() - 1.0)
      rnd_4 = 0.5 * (2 * random.random() - 1.0)
      x0 = np.array([k_approx + rnd_0, 1.0 + rnd_1, 0.0 + rnd_2,
                               0.0 + rnd_3, 1.0 + rnd_4, 0.0001])
      result = optimize.minimize(f_opt, x0, tol=1e-13, method='Powell',
                                 bounds=((k_min, k_max),
                                         (-2, 2), (-2, 2), (-2, 2), (-2, 2), (0.0, 0.01)))
      print ("fit: ", result)
      results.append(result)
      if result['fun'] < 0.01: break
    results.sort(key = lambda x: x['fun'])
    result = results[0]
    k_fit, A_fit_re, A_fit_im, B_fit_re, B_fit_im, k1 = result['x']
    A_fit = A_fit_re + 1j * A_fit_im
    B_fit = B_fit_re + 1j * B_fit_im

    if k_fit < 0:
        A_fit, B_fit = B_fit, A_fit
        k_fit *= -1
    return k_fit, A_fit, B_fit, k1, np.sqrt(result['fun'] / len(x_fit))

def get_T_and_R(omega, xs, mx, k_approx):
    f = omega / GHz_2pi
    phase_change = 0.0
    phases = []
    n_changes = 0
    print ("mx: ", np.shape(mx))
    i_zero = np.argmin(np.abs(xs))
    i_max = len(xs) - 5
    i_min = i_zero  + 5
    xvals = []
    for i in range(i_min, i_max - 1):
        dphi = np.angle(mx[i + 1] / mx[i])
        while dphi > np.pi/2.0:
            dphi -= 2.0 * np.pi
        while dphi < -np.pi/2.0:
            dphi += 2.0 * np.pi
        phase_change += dphi
        phases.append(phase_change)
        xvals.append(xs[i])
    phases = np.array(phases)
    xvals = np.array(xvals)
    dx = xs[i_max] - xs[i_min]
    k_approx = phase_change / dx
    print ("approx k from phase change: ", k_approx)
    print ("approximate wave number ", k_approx)
    def f_lin(x, a, b):
        return a * x + b
    p_opt, p_cov = optimize.curve_fit(f_lin, xvals, phases)
    print ("linear fit: ", p_opt, p_cov)
    print ("slope: ", p_opt[0])
    if phase_change > 2.0 * np.pi:
        k_approx = np.abs(p_opt[0])
    #for i in range(i_min, i_max):
    #    if mx[i].real * mx[i + 1].real < 0: n_changes += 1
    #if n_changes > 0:
    #    k_min = np.pi * (n_changes - 1) / (xs[i_max] - xs[i_min])
    #else:
    #    k_min = 0.0

    #k_max =  np.pi * (n_changes + 1) / (xs[i_max] - xs[i_min])
    k_min = max((abs(phase_change) - np.pi) / abs(dx), 0.0)
    k_max = (abs(phase_change) + np.pi) / abs(dx)

    print ("K: approx = ", k_approx, "min = ", k_min, "max = ", k_max)
    if k_approx < k_min:
        print ("shift k_min")
        k_min = max(2.2 * k_approx - k_min, 0.0)
    if k_approx > k_max:
        print ("shift k_max")
        k_max = 2.2 * k_approx - k_max
    
    print ("k_approx: ", k_approx)
    x_in_a = -0.6 * abs(xs[0])
    x_in_b = -0.2 * abs(xs[0])
    x_out_a = 0.2 * abs(xs[0])
    x_out_b = 0.7 * abs(xs[0])
    k_scat, A_inc, B_inc, A_out, k1, err = get_scat_solution(x_in_a, x_in_b,
                                                             x_out_a, x_out_b,
                                                             xs, mx,
                                                             k_min, k_max,
                                                             k_approx)
    k_inc = k_out = k_scat
    k1_inc = k1_out = k1
    err_inc = err_out = err
    B_out = 0.0
    #k_out, A_out, B_out, k1_out, err_out = get_amplitudes(x_out_a, x_out_b,
    #                                                      xs, mx,
    #                                                      k_min, k_max,
    #                                                      k_approx)
    #k_approx = k_out
    #k_inc, A_inc, B_inc, k1_inc, err_inc = get_amplitudes(x_in_a, x_in_b,
    #                                                      xs, mx,
    #                                                      k_min, k_max,
    #                                                      k_approx) 
    #k_inc, A_inc, B_inc, k1_inc, err_inc = get_amplitudes(x_in_a, x_in_b,
    #                                                      xs, mx,
    #                                                      0.99*k_approx,
    #                                                      1.01*k_approx,
    #                                                      k_approx) 
    print ("incident: ", k_inc, A_inc, B_inc, k1_inc);
    print ("transmitted: ", k_out, A_out, B_out, k1_out)
    print ("Transmission: ", A_out / A_inc, "reflection: ", B_inc / A_inc)
    T =  A_out / A_inc
    R = B_inc / A_inc
    if  False or err_inc > 0.08 or err_out > 0.08:
        print ("f = ", f, "errs = ", err_inc, err_out)
        pl.figure()
        pl.plot(xs, np.abs(mx))
        pl.plot(xs, mx.real)
        pl.plot(xs, mx.imag)
        x_in = np.linspace(x_in_a, x_in_b, 100)
        x_out = np.linspace(x_out_a, x_out_b, 100)
        psi_inc = f_wave_fit(x_in,  k_inc, A_inc, B_inc, k1_inc)
        psi_out = f_wave_fit(x_out, k_out, A_out, B_out, k1_out)
        pl.plot(x_in, psi_inc.real, '--', label='Re inc')
        pl.plot(x_in, psi_inc.imag, '--', label='Im Inc')
        pl.plot(x_out, psi_out.real, '--', label='Re out')
        pl.plot(x_out, psi_out.imag, '--', label='Im out')
        pl.legend()
        pl.show()

    k_new_approx = (err_out * k_inc + err_inc * k_out) / (err_inc + err_out)
    return T, R, err_inc + err_out, k_new_approx

def readData(fname):
    d = np.load(fname)
    for k in d.keys(): print (k)
    omega = d['omega']
    T = d['T']
    R = d['R']
    k = 0.0 * omega + 0.0
    k_bk = 0.0 * omega + 0.0
    err = 0.0 * np.abs(T)
    if 'T_bk' in d.keys():
       T_bk = d['T_bk']
       R_bk = d['R_bk']
       err_bk = 0.0 * np.abs(T)
        

    #pl.figure()
    xs = d['xs']
    if False:
        pl.plot(xs, d['mode_s_mx'][1].real)
        pl.plot(xs, d['mode_s_mx'][1].imag)
        pl.plot(xs, d['mode_s_mz'][1].real)
        pl.plot(xs, d['mode_s_mz'][1].imag)

    k_approx_frw = 1.0
    k_approx_bk = 1.0
    for i in range(len(R)):
        #if abs(R[i]) > 1.0: 
           T_i, R_i, err_i, k_approx_frw = get_T_and_R(omega[i],
                                                   d['xs'], d['mode_s_mx'][i],
                                                       k_approx_frw)
           T[i] = T_i
           R[i] = R_i
           err[i] = err_i
           k[i] = k_approx_frw
           if 'mode_s_mx_bk' in d.keys():
               T_i, R_i, err_i, k_approx_bk = get_T_and_R(omega[i],
                                                -d['xs'][-1::-1],
                                                d['mode_s_mx_bk'][i][-1::-1],
                                                k_approx_bk)
               T_bk[i] = T_i
               R_bk[i] = R_i
               err_bk[i] = err_i
               k_bk[i] = k_approx_bk

    fname_new = fname[:-4] + "-proc.npz"
    d_new = dict()
    d_new.update(d)
    d_new['T1'] = T
    d_new['R1'] = R
    d_new['err'] = err
    d_new['k'] = k
    if 'T_bk' in d.keys():
        d_new['T1_bk'] = T_bk
        d_new['R1_bk'] = R_bk
        d_new['err_bk'] = err_bk
        d_new['k_bk'] = k_bk
    np.savez(fname_new, **d_new)
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
    ax_angle = pl.gca().twinx()
    ax_angle.plot (omega / GHz_2pi, np.angle(T) * 180.0/np.pi,
                   color='m', label='arg T forw')
    ax_angle.set_ylim(-180.0, 180.0)
    pl.legend()
    pl.xlabel("Frequency $f$")
    pl.title("Forward")
    if 'T_bk' in d.keys():
        pl.figure()
        p_t = pl.plot(omega / GHz_2pi, np.abs(T_bk), label='|T| back')
        pl.plot(omega / GHz_2pi, err_bk, label='err back')
        p_r = pl.plot(omega / GHz_2pi, np.abs(R_bk), label='|R| back')
        pl.legend()
        ax_angle = pl.gca().twinx()
        ax_angle.plot (omega / GHz_2pi, np.angle(T_bk) * 180.0/np.pi,
                       color='m', label='arg T back')
        ax_angle.set_ylim(-180.0, 180.0)
        pl.legend()
        pl.xlabel("Frequency $f$")
        pl.title("Backward")

    pl.figure()
    pl.plot(k, omega / GHz_2pi)
    pl.plot(k_bk, omega / GHz_2pi, '--')
    pl.xlabel(r"wavenumber $k$")
    pl.ylabel(r"Frequency $f$")
    #pl.figure()
    #pl.plot(omega / GHz_2pi, np.angle(T))

    pl.show()

    


for fname in sys.argv[1:]:
    readData(fname)
