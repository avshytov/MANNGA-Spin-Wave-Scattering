import numpy as np
import pylab as pl
import sys
from scipy import optimize
import random
from scipy import linalg

from constants import GHz_2pi


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
        r_in = np.sum(np.abs(psi_in_fit - psi_in)**2)    #/ C_in
        r_out = np.sum(np.abs(psi_out_fit - psi_out)**2) #/ C_out
        return r_in / len(psi_in)  + r_out / len(psi_out)

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
      if result['fun'] < 0.001: break
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

def fit_wave_ampl(xs, psi1, psi2, k_approx, x_l1_a, x_l1_b,
                  x_r1_a, x_r1_b, x_l2_a, x_l2_b, x_r2_a, x_r2_b):
    
    l1_fit = [t for t in range(len(xs)) if xs[t]>x_l1_a and xs[t]<x_l1_b]
    l2_fit = [t for t in range(len(xs)) if xs[t]>x_l2_a and xs[t]<x_l2_b]
    r1_fit = [t for t in range(len(xs)) if xs[t]>x_r1_a and xs[t]<x_r1_b]
    r2_fit = [t for t in range(len(xs)) if xs[t]>x_r2_a and xs[t]<x_r2_b]

    #psi1_max = np.max(np.abs(psi1))
    #psi2_max = np.max(np.abs(psi2))
    
    psi_l1  = np.array([psi1[t] for t in l1_fit]) #/ psi1_max
    psi_r1  = np.array([psi1[t] for t in r1_fit]) #/ psi1_max
    psi_l2  = np.array([psi2[t] for t in l2_fit]) #/ psi2_max
    psi_r2  = np.array([psi2[t] for t in r2_fit]) #/ psi2_max
    
    x_l1  = np.array([xs[t] for t in l1_fit])
    x_r1  = np.array([xs[t] for t in r1_fit])
    x_l2  = np.array([xs[t] for t in l2_fit])
    x_r2  = np.array([xs[t] for t in r2_fit])
    
    def f_mismatch(p):
        k = p[0]; k1 = p[1]
        A1 = p[2] + 1j * p[3]
        B1 = p[4] + 1j * p[5]
        C1 = p[6] + 1j * p[7]
        D1 = p[8] + 1j * p[9]
        A2 = p[10] + 1j * p[11]
        B2 = p[12] + 1j * p[13]
        C2 = p[14] + 1j * p[15]
        D2 = p[16] + 1j * p[17]

        psi_fit_l1 = f_wave_fit(x_l1, k, A1, B1, k1)
        psi_fit_l2 = f_wave_fit(x_l2, k, A2, B2, k1)
        psi_fit_r1 = f_wave_fit(x_r1, k, C1, D1, k1)
        psi_fit_r2 = f_wave_fit(x_r2, k, C2, D2, k1)
        
        err2  = linalg.norm(psi_fit_l1 - psi_l1)**2 / len(psi_l1)
        err2 += linalg.norm(psi_fit_l2 - psi_l2)**2 / len(psi_l2)
        err2 += linalg.norm(psi_fit_r1 - psi_r1)**2 / len(psi_r1)
        err2 += linalg.norm(psi_fit_r2 - psi_r2)**2 / len(psi_r2)

        return err2


    k_min = k_approx * 0.95
    k_max = k_approx * 1.05
    
    bounds = [(k_min, k_max), (0.0, 0.001),
              (-3.0, 3.0), (-3.0, 3.0),
              (-3.0, 3.0), (-3.0, 3.0),
              (-3.0, 3.0), (-3.0, 3.0),
              (-3.0, 3.0), (-3.0, 3.0),
              (-3.0, 3.0), (-3.0, 3.0),
              (-3.0, 3.0), (-3.0, 3.0),
              (-3.0, 3.0), (-3.0, 3.0),
              (-3.0, 3.0), (-3.0, 3.0)
              ]

    results = []
    max_it = 10
    n_it = 0
    while n_it < max_it:
      n_it += 1

      rnd = np.zeros((18))
      rnd[0] = 0.5 * (2 * random.random() - 1.0) * (k_max - k_min)
      if n_it == 1: rnd[0] = 0.0
      for i in range(2, 18):
          rnd[i] = 0.5 * (2 * random.random() - 1.0)

      p0 = np.array([k_approx + rnd[0], 0.0001,
                   1.0 + rnd[2],  0.0 + rnd[3],
                   1.0 + rnd[4],  0.0 + rnd[5],
                   0.0 + rnd[6],  1.0 + rnd[7],
                   1.0 + rnd[8],  0.0 + rnd[9],
                   0.0 + rnd[10], 1.0 + rnd[11],
                   1.0 + rnd[12], 1.0 + rnd[13],
                   0.0 + rnd[14], 0.0 + rnd[15],
                   1.0 + rnd[16], 0.0 + rnd[17]])
      
      result = optimize.minimize(f_mismatch, p0, tol=1e-13, method='Powell',
                                 bounds=bounds)
      print ("fit: ", result)
      results.append(result)
      if result['fun'] < 0.001: break
      
    results.sort(key = lambda x: x['fun'])
    result = results[0]
    print ("best: ", result)
    fit_res = dict()
    p = result['x']
    fit_res['k'] = p[0]
    fit_res['k1'] = p[1]
    fit_res['A1'] = p[2] + 1j * p[3]
    fit_res['B1'] = p[4] + 1j * p[5]
    fit_res['C1'] = p[6] + 1j * p[7]
    fit_res['D1'] = p[8] + 1j * p[9]
    fit_res['A2'] = p[10] + 1j * p[11]
    fit_res['B2'] = p[12] + 1j * p[13]
    fit_res['C2'] = p[14] + 1j * p[15]
    fit_res['D2'] = p[16] + 1j * p[17]
    fit_res['err'] = np.sqrt(result['fun'])
    fit_res['status'] = result['status']
    fit_res['message'] = result['message']
    return fit_res
    

def get_T_and_R_matrix(omega, xs, psi1, psi2, k_approx):

    psi1 /= np.max(np.abs(psi1))
    psi2 /= np.max(np.abs(psi2))
    lmb = 2.0 * np.pi / k_approx
    
    x_l1_b = - lmb * 0.8
    x_l1_a = x_l1_b - 1 * lmb
    x_r1_a = lmb * 0.8
    x_r1_b = x_r1_a + 1.5 * lmb
    x_l2_b = - x_r1_a
    x_l2_a = - x_r1_b
    x_r2_a = - x_l1_b
    x_r2_b = - x_l2_a

    wave_ampl_fit = fit_wave_ampl(xs, psi1, psi2, k_approx,
                                  x_l1_a, x_l1_b, x_r1_a, x_r1_b,
                                  x_l2_a, x_l2_b, x_r2_a, x_r2_b)
    A1 = wave_ampl_fit['A1']
    A2 = wave_ampl_fit['A2']
    B1 = wave_ampl_fit['B1']
    B2 = wave_ampl_fit['B2']
    C1 = wave_ampl_fit['C1']
    C2 = wave_ampl_fit['C2']
    D1 = wave_ampl_fit['D1']
    D2 = wave_ampl_fit['D2']
    k  = wave_ampl_fit['k']
    k1 = wave_ampl_fit['k1']

    err = wave_ampl_fit['err']
    
    Det = A1 * D2 - D1 * A2
    T1 = (C1 * D2 - D1 * C2) / Det
    T2 = (B2 * A1 - B1 * A2) / Det
    R1 = (B1 * D2 - D1 * B2) / Det
    R2 = (C2 * A1 - C1 * A2) / Det


    print ("o = ", omega / GHz_2pi, "k = ", k, "k1 = ", k1, "err = ", err)
    print ("Transmission: ", T1, T2, np.abs(T1), np.abs(T2))
    print ("reflection: ", R1, R2, np.abs(R1), np.abs(R2))
    print ("unitarity: ", 1.0 - np.abs(T1)**2 - np.abs(R1)**2,
           1.0 - np.abs(T2)**2 - np.abs(R2)**2)
    if  False or err > 0.05:
        xl1 = np.linspace(x_l1_a, x_l1_b, 101)
        xl2 = np.linspace(x_l2_a, x_l2_b, 101)
        xr1 = np.linspace(x_r1_a, x_r1_b, 101)
        xr2 = np.linspace(x_r2_a, x_r2_b, 101)
        pl.figure()
        pl.plot(xs, np.abs(psi1), label=r'|psi1|')
        pl.plot(xl1, np.abs(f_wave_fit(xl1, k, A1, B1, k1)), '--')
        pl.plot(xr1, np.abs(f_wave_fit(xr1, k, C1, D1, k1)), '--')
        pl.plot(xs, psi1.real, label='Re psi1')
        pl.plot(xl1, f_wave_fit(xl1, k, A1, B1, k1).real, '--')
        pl.plot(xr1, f_wave_fit(xr1, k, C1, D1, k1).real, '--')
        pl.plot(xs, psi1.imag, label='Im psi1')
        pl.plot(xl1, f_wave_fit(xl1, k, A1, B1, k1).imag, '--')
        pl.plot(xr1, f_wave_fit(xr1, k, C1, D1, k1).imag, '--')
        pl.legend()
        pl.figure()
        pl.plot(xs, np.abs(psi2), label=r'|psi2|')
        pl.plot(xl2, np.abs(f_wave_fit(xl2, k, A2, B2, k1)), '--')
        pl.plot(xr2, np.abs(f_wave_fit(xr2, k, C2, D2, k1)), '--')
        pl.plot(xs, psi2.real, label='Re psi2')
        pl.plot(xl2, f_wave_fit(xl2, k, A2, B2, k1).real, '--')
        pl.plot(xr2, f_wave_fit(xr2, k, C2, D2, k1).real, '--')
        pl.plot(xs, psi2.imag, label='Im psi2')
        pl.plot(xl2, f_wave_fit(xl2, k, A2, B2, k1).imag, '--')
        pl.plot(xr2, f_wave_fit(xr2, k, C2, D2, k1).imag, '--')
        pl.legend()
        pl.show()
    
    return k, T1, T2, R1, R2, err
    
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
    lmb = 2.0 * np.pi / k_approx
    #x_in_a = -2.2 * lmb - a/2
    x_in_b = - lmb * 0.8
    x_in_a = x_in_b - lmb
    x_out_a = lmb * 0.8
    x_out_b = x_out_a + 1.5 * lmb
    #x_in_a = -0.6 * abs(xs[0])
    #x_in_b = -0.2 * abs(xs[0])
    #x_out_a = 0.2 * abs(xs[0])
    #x_out_b = 0.7 * abs(xs[0])
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
    if  False or err_inc > 0.05 or err_out > 0.05:
        print ("f = ", f, "errs = ", err_inc, err_out)
        pl.figure()
        pl.plot(xvals, phases)
        pl.plot(xvals, p_opt[0] * xvals + p_opt[1])
        pl.plot(xvals, k_max * xvals, '--')
        pl.plot(xvals, k_min * xvals, '--')
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
    for kd in d.keys(): print (kd)
    omega = d['omega']
    xs = d['xs']
    Ns = d['N']
    k_disp = d['k']
    print ("Ns = ", Ns)
    mx_fw = d['mxs_fw']
    mx_bk = d['mxs_bk']
    k_approx_frw = 1.0
    k_approx_bk = 1.0
    T_fw = 0.0 * omega + 0.0j
    R_fw = 0.0 * omega + 0.0j
    T_bk = 0.0 * omega + 0.0j
    R_bk = 0.0 * omega + 0.0j
    err_fw = 0.0 * omega
    err_bk = 0.0 * omega
    k_fit_fw = 0.0 * omega
    k_fit_bk = 0.0 * omega
    for i in range(len(Ns)):
           N_i = Ns[i]
           xs_i = xs[i,  :N_i]
           mx_fw_i = mx_fw[i, :N_i]
           mx_bk_i = mx_bk[i, :N_i]
           print ("orig k: ", d['k'][i])
           k_approx = k_disp[i]
           result = get_T_and_R_matrix(omega[i],
                                       xs_i, mx_fw_i, mx_bk_i, k_approx)
           k_i, T_fw_i, T_bk_i, R_fw_i, R_bk_i, err_i = result
           err_fw[i] = err_i
           err_bk[i] = err_i
           k_fit_fw[i] = k_i
           k_fit_bk[i] = k_i
           T_fw[i] = T_fw_i
           T_bk[i] = T_bk_i
           R_fw[i] = R_fw_i
           R_bk[i] = R_bk_i
           
           #T_i, R_i, err_i, k_approx_frw = get_T_and_R(omega[i],
           #                                     xs_i, mx_fw_i, k_approx_frw)
           #T_fw[i] = T_i
           #R_fw[i] = R_i
           #err_fw[i] = err_i
           #k_fit_fw[i] = k_approx_frw
           #T_i, R_i, err_i, k_approx_bk = get_T_and_R(omega[i],
           #                                    - xs_i[-1::-1],
           #                                       mx_bk_i[-1::-1],
           #                                       k_approx_bk)
           #T_bk[i] = T_i
           #R_bk[i] = R_i
           #err_bk[i] = err_i
           #k_fit_bk[i] = k_approx_bk
           

    fname_new = fname[:-4] + "-proc.npz"
    d_new = dict()
    d_new.update(d)
    d_new['T1_fw']  = T_fw
    d_new['R1_fw']  = R_fw
    d_new['T1_bk']  = T_bk
    d_new['R1_bk']  = R_bk
    d_new['k_fw']   = k_fit_fw
    d_new['k_bk']   = k_fit_bk
    d_new['err_fw'] = err_fw
    d_new['err_bk'] = err_bk
    np.savez(fname_new, **d_new)
    
for fname in sys.argv[1:]:
    readData(fname)
