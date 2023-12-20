import numpy as np
import pylab as pl
import sys
from scipy import optimize
import random

from constants import GHz_2pi

from resfit import R_res, T_res, T_res_phi, R_Fano
from resfit import fit_T_resonance_with_phase, fit_T_and_R, fit_T_and_T
from resfit import fit_T2_resonance, fit_R2_resonance, fit_R_Fano



def readData(fname):
    d = np.load(fname)
    #for k in d.keys(): print (k)
    #for k in ['a', 'b', 's', 'd']:
    #    if k in d.keys(): print (k, " = ", d[k])
    omega = d['omega']
    T = d['T1_fw']
    R = d['R1_fw']
    k = 0.0 * omega
    #if 'k' in d.keys():
    k = d['k_fw'] #0.0 * omega + 0.0
    k_bk = 0.0 * omega + 0.0
    err = d['err_fw'] #0.0 * np.abs(T)
    T_bk = d['T1_bk']
    R_bk = d['R1_bk']
    #err_bk = 0.0 * np.abs(T)
    err_bk = d['err_bk']
    k_bk = d['k_bk']
    k_orig = d['k']

    result = []
    for i in range(len(omega)):
        item = dict(
            omega = omega[i], k_orig = k[i], k = k[i], k_bk = k_bk[i],
            err = err[i], err_bk = err_bk[i],
            T = T[i], T_bk = T_bk[i], R = R[i], R_bk = R_bk[i],
            fname = fname)
        result.append(item)
    return result


def joinData(fnames):
    result = []
    for fname in fnames:
        dataset = readData(fname)
        result.extend(dataset)
    result.sort(key = lambda item: item['omega'])
    res_filtered = []
    last_group = []
    for item in result:
        print ("item", item['omega'] / GHz_2pi, item['fname'])
        if len(last_group) == 0:
            last_group.append(item)
            print ('start new group')
        elif abs(last_group[-1]['omega'] - item['omega']) < 1e-6:
            last_group.append(item)
            print ('add to the group')
        else:
            last_group.sort(key = lambda item: item['err'])
            res_filtered.append(last_group[0])
            last_group = []
            print('add to the filtered result')
    if len(last_group):
        last_group.sort(key = lambda item: item['err'])
        res_filtered.append(last_group[0])
        print('flush the group')
    print ("total: ", len(res_filtered))
    res_data = dict()
    keys_to_copy = [
        'omega', 'T', 'T_bk', 'R', 'R_bk',
        'k_orig', 'k', 'k_bk', 'err', 'err_bk'
    ]
    for k in keys_to_copy:
        res_data[k] = np.array([t[k] for t in res_filtered])
    print ("res_data:", res_data)
    return res_data
        

def plotData(d):
    for k in d.keys(): print (k)
    omega = d['omega']
    T = d['T']
    R = d['R']
    k = 0.0 * omega
    #if 'k' in d.keys():
    k = d['k'] #0.0 * omega + 0.0
    k_bk = 0.0 * omega + 0.0
    err = d['err'] #0.0 * np.abs(T)
    T_bk = d['T_bk']
    R_bk = d['R_bk']
    #err_bk = 0.0 * np.abs(T)
    err_bk = d['err_bk']
    k_bk = d['k_bk']
    k_orig = d['k']
    pl.figure()
    p_t = pl.plot(omega / GHz_2pi, np.abs(T), label='|T| forw')
    pl.plot(omega / GHz_2pi, err, label='err fw')
    p_r = pl.plot(omega / GHz_2pi, np.abs(R), label='|R| forw')
    pl.plot(omega / GHz_2pi, np.abs(T_bk), '--', color=p_t[0].get_color())
    pl.plot(omega / GHz_2pi, np.abs(R_bk), '--', color=p_r[0].get_color())
    pl.plot(omega / GHz_2pi, err, label='err bk')
    pl.legend()
    ax_angle = pl.gca().twinx()
    ax_angle.plot (omega / GHz_2pi, np.angle(T) * 180.0/np.pi,
                   color='m', label='arg T forw')
    ax_angle.plot (omega / GHz_2pi, np.angle(T_bk) * 180.0/np.pi,
                       'c')
    ax_angle.set_ylim(-180.0, 180.0)
    pl.legend()
    pl.xlabel("Frequency $f$")

    pl.figure()
    p_t = pl.plot(omega / GHz_2pi, np.abs(T), label=r'$|T|$ forw')
    pl.plot(omega / GHz_2pi, err, label='err fw')
    p_r = pl.plot(omega / GHz_2pi, np.abs(R), label=r'$|R|$ forw')
    pl.legend()
    ax_T = pl.gca()
    ax_angle = pl.gca().twinx()
    ax_angle.plot (omega / GHz_2pi, np.angle(T) * 180.0/np.pi,
                   color='m', label=r'arg $T {forw}$')
    ax_angle.set_ylim(-180.0, 180.0)
    pl.legend()
    pl.ylabel(r"Transmissivity/Reflectivity")
    ax_angle.set_ylabel("Phase")
    pl.xlabel(r"Frequency $f$, GHz")
    pl.title("Forward")

    #res_min = 3.10 * GHz_2pi
    #res_max = 3.25 * GHz_2pi
    #res_min = 3.15 * GHz_2pi
    #res_max = 3.25 * GHz_2pi
    #res_min = 4.02 * GHz_2pi
    #res_max = 4.07 * GHz_2pi
    #res_min = 4.05 * GHz_2pi
    #res_max = 4.10 * GHz_2pi
    #res_min = 3.90 * GHz_2pi
    #res_max = 4.10 * GHz_2pi

    #res_min = 1.75 * GHz_2pi
    #res_max = 2.10 * GHz_2pi
    #res_min = 1.90 * GHz_2pi
    #res_max = 2.00 * GHz_2pi

    res_min = 2.72 * GHz_2pi
    res_max = 2.85 * GHz_2pi

    #res_min = 3.8 * GHz_2pi
    #res_max = 4.0 * GHz_2pi
    #res_min = 0.8 * GHz_2pi
    #res_max = 5.15 * GHz_2pi
    #res_min = 2.5 * GHz_2pi
    #res_max = 3.2 * GHz_2pi
    #res_min = 3.85 * GHz_2pi
    #res_max = 3.95 * GHz_2pi
    #res_min = 3.75 * GHz_2pi
    #res_max = 3.95 * GHz_2pi

    #show_omega = np.array([1.6, 2.34, 3.1, 4.01]) * GHz_2pi
    #show_omega = np.array([1.97, 2.784]) * GHz_2pi
    #show_omega = np.array([1.75, 2.64, 3.84]) * GHz_2pi
    show_omega = []

    for o in show_omega:
        i_omega = np.argmin(np.abs(o - d['omega']))
        pl.figure()
        xr_i = d['xr'][i_omega][:]
        pl.plot(xr_i, d['mxr_fw'][i_omega][:].real, label='Re mx')
        pl.plot(xr_i, d['mxr_fw'][i_omega][:].imag, label='Im mx')
        pl.plot(xr_i, d['mzr_fw'][i_omega][:].real, label='Re mz')
        pl.plot(xr_i, d['mzr_fw'][i_omega][:].imag, label='Im mz')
        pl.plot(xr_i, np.abs(d['mxr_fw'][i_omega]), label='|m_x|')
        pl.plot(xr_i, np.abs(d['mzr_fw'][i_omega]), label='|m_z|')
        pl.legend()
        pl.title("f = %g ---> " % (d['omega'][i_omega] / GHz_2pi))
        
    for o in show_omega:
        i_omega = np.argmin(np.abs(o - d['omega']))
        xr_i = d['xr'][i_omega][:]
        pl.figure()
        pl.plot(xr_i, d['mxr_bk'][i_omega][:].real, label='Re mx')
        pl.plot(xr_i, d['mxr_bk'][i_omega][:].imag, label='Im mx')
        pl.plot(xr_i, d['mzr_bk'][i_omega][:].real, label='Re mz')
        pl.plot(xr_i, d['mzr_bk'][i_omega][:].imag, label='Im mz')
        pl.plot(xr_i, np.abs(d['mxr_bk'][i_omega]), label='|m_x|')
        pl.plot(xr_i, np.abs(d['mzr_bk'][i_omega]), label='|m_z|')
        pl.legend()
        pl.title("f = %g <-- " % (d['omega'][i_omega] / GHz_2pi))

    pl.figure()
    theta = np.linspace(-np.pi, np.pi, 1001)
    exp_th = np.exp(1j * theta)
    for o in show_omega:
        i_omega = np.argmin(np.abs(o - d['omega']))
        xr_i = d['xr'][i_omega][:]
        mx_fw = d['mxr_fw'][i_omega][:]
        mz_fw = d['mzr_fw'][i_omega][:]
        mx_bk = d['mxr_bk'][i_omega][:]
        mz_bk = d['mzr_bk'][i_omega][:]
        i_fw_max = np.argmax(np.abs(mx_fw))
        mx_fw_max = mx_fw[i_fw_max] 
        mz_fw_max = mz_fw[i_fw_max]
        n_fw = np.abs(mx_fw_max) + np.abs(mz_fw_max)
        i_bk_max = np.argmax(np.abs(mx_bk))
        mx_bk_max = mx_bk[i_bk_max]
        mz_bk_max = mz_bk[i_bk_max]
        n_bk = np.abs(mx_bk_max) + np.abs(mz_bk_max)
        p = pl.plot((mx_fw_max / n_fw * exp_th).real,
                    (mz_fw_max / n_fw * exp_th).real,
                label='@ %g -->' % (d['omega'][i_omega] / GHz_2pi))
        pl.plot((mx_bk_max / n_bk * exp_th).real,
                (mz_bk_max / n_bk * exp_th).real,
                '--', color=p[0].get_color(),
                label='@ %g <--' % (d['omega'][i_omega] / GHz_2pi))
    pl.gca().set_aspect('equal', 'box')
    pl.legend()
    
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
    if False:
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

    
    
    pl.figure()
    p_t = pl.plot(omega / GHz_2pi, np.abs(T_bk), label='|T| back')
    pl.plot(omega / GHz_2pi, err_bk, label='err back')
    p_r = pl.plot(omega / GHz_2pi, np.abs(R_bk), label='|R| back')
    pl.legend()
    ax_Tbk = pl.gca()
    ax_angle = pl.gca().twinx()
    ax_angle.plot (omega / GHz_2pi, np.angle(T_bk) * 180.0/np.pi,
                   color='m', label=r'arg $T_{back}$')
    ax_angle.set_ylim(-180.0, 180.0)
    pl.legend()
    pl.xlabel("Frequency $f$")
    pl.title("Backward")
    pl.ylabel(r"Transmissivity/Reflectivity")
    ax_angle.set_ylabel(r"Phase")
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
    if False:
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
      R1_0 = 0.0
      R2_0 = 0.0
      try:
         R1_0 = fit_R_Fano(res_min, res_max, omega, R, Omega_3, Gamma_R3,
                           Gamma_L3, Gamma_3)
         print ("Fano amplitude: ", R1_0)
         R2_0 = fit_R_Fano(res_min, res_max, omega, R_bk, Omega_3, Gamma_L3,
                           Gamma_R3, Gamma_3)
         print ("Fano amplitude: ", R2_0)
      except:
          import traceback
          traceback.print_exc()
      R1_fit = R_Fano(omega, Omega_3, Gamma_R3, Gamma_L3, Gamma_3, R1_0 * 1)
      R2_fit = R_Fano(omega, Omega_3, Gamma_L3, Gamma_R3, Gamma_3, R2_0 * 1)
      ax_T.plot(omega / GHz_2pi, np.abs(T1_fit), label='T+T fit T')
      ax_T.plot(omega / GHz_2pi, np.abs(R1_fit), label='T+T fit R')
      ax_Tbk.plot(omega / GHz_2pi, np.abs(T2_fit), label='T+T fit T')
      ax_Tbk.plot(omega / GHz_2pi, np.abs(R2_fit), label='T+T fit R')
      ax_T.legend()
      ax_Tbk.legend()

    pl.figure()
    pl.plot(k, omega / GHz_2pi)
    pl.plot(k_bk, omega / GHz_2pi, '--')
    pl.plot(k_orig, omega / GHz_2pi, '--')
    pl.xlabel(r"wavenumber $k$")
    pl.ylabel(r"Frequency $f$")
    #pl.figure()
    #pl.plot(omega / GHz_2pi, np.angle(T))

    pl.figure()
    pl.plot(T.real, T.imag, label='forward')
    pl.xlabel(r"Re $T(\omega)$")
    pl.ylabel(r"Im $T(\omega)$")
    pl.gca().set_aspect('equal', 'box')
    #if 'T_bk' in d.keys():
    pl.plot(T_bk.real, T_bk.imag, label='backward')
    pl.legend()

    pl.figure()
    ax_abs = pl.gca()
    ax_angle = pl.gca().twinx()
    p_fw = ax_abs.plot(omega / GHz_2pi, np.abs(T), label=r'$|T_{fw}(\omega)|$')
    p_bk = ax_abs.plot(omega / GHz_2pi, np.abs(T_bk), label=r'$|T_{bk}(\omega)|$')
    ax_angle.plot(omega / GHz_2pi, np.angle(T) * 180.0/np.pi, '--', 
                  color=p_fw[0].get_color())
    ax_angle.plot(omega / GHz_2pi, np.angle(T_bk) * 180.0/np.pi, '--', 
                  color=p_bk[0].get_color())
    ax_angle.set_ylim(-180.0, 180.0)
    ax_abs.plot(omega / GHz_2pi, np.abs(R), label='|R_{fw}(\omega)|')
    ax_abs.plot(omega / GHz_2pi, np.abs(R_bk), label='|R_{bk}(\omega)|')
    pl.legend()
    pl.xlabel(r"Frequency $\omega/2\pi$, GHz")
    pl.ylabel(r"Transmissivity $|T(\omega)|$")
    ax_angle.set_ylabel(r"Phase arg $T(\omega)$")
    ax_angle.set_yticks([-180.0, -90.0, 0.0, 90.0, 180.0])
    pl.figure()
    pl.plot(omega / GHz_2pi, T.real, label=r'Re $T_{fw}(\omega)$')
    pl.plot(omega / GHz_2pi, T_bk.real, label=r'Re $T_{bk}(\omega)$')
    pl.xlabel(r"Frequency $\omega/2\pi$, GHz")
    pl.legend()
    pl.figure()
    pl.plot(omega / GHz_2pi, T.imag, label=r'Im $T_{fw}(\omega)$')
    pl.plot(omega / GHz_2pi, T_bk.imag, label=r'Im $T_{bk}(\omega)$')
    pl.xlabel(r"Frequency $\omega/2\pi$, GHz")
    pl.legend()
    pl.show()

    

res_data = joinData(sys.argv[1:])
plotData(res_data)
#for fname in sys.argv[1:]:
#    readData(fname)
