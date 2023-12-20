import numpy as np
import pylab as pl
import sys
import resfit
from scipy import linalg

from resfit import T_res, R_res

from printconfig import print_config

from constants import ns, GHz_2pi

def readData(fname):
    d = np.load(fname)
    for k in d.keys():
        print(k)

    omega = d['omega']
    T1 = d['T1']
    if len(T1) < len(omega):
        omega = omega[:len(T1)]
    R1 = d['R1']
    A1 = d['A1']
    T2 = d['T2']
    R2 = d['R2']
    A2 = d['A2']
    T1_nrf = d['T1_nrf']
    R1_nrf = d['R1_nrf']
    A1_nrf = d['A1_nrf']
    T2_nrf = d['T2_nrf']
    R2_nrf = d['R2_nrf']
    A2_nrf = d['A2_nrf']
    if 'Gamma_rad' in d.keys():
        Gamma_rad = d['Gamma_rad_nrf']
    else:
        Gamma_rad = 0.0 * omega
    if 'Gamma_R' in d.keys():
        Gamma_R = d['Gamma_R_nrf']
        Gamma_L = d['Gamma_L_nrf']
    else:
        Gamma_R = 0.0 * omega
        Gamma_L = 0.0 * omega

    if 'res_modes_freq_nrf' in d.keys():
        print ("nrf freqs: ", d['res_modes_freq_nrf'] / GHz_2pi)

    if False and 'res_modes_freq' in d.keys():
        mode_freq = d['res_modes_freq_nrf']
        mx = d['res_modes_mx_nrf']
        mz = d['res_modes_mz_nrf']
        res_X = d['res_mode_X_nrf']
        res_Z = d['res_mode_Z_nrf']
        for i_mode in range(len(mode_freq)):
            pl.figure()
            pl.tripcolor(res_X, res_Z, np.abs(mx[i_mode, :]), cmap='magma')
            pl.colorbar()
            pl.title("f = %g" % (mode_freq[i_mode] / GHz_2pi))
            pl.gca().set_aspect('equal', 'box')
        pl.figure()
        theta = np.linspace(0.0, 2.0 * np.pi, 1001)
        exp_theta = np.exp(1j * theta)
        for i_mode in range(len(mode_freq)):
            i_max = np.argmax(np.abs(mx[i_mode, :]))
            mx_max = mx[i_mode, i_max]
            mz_max = mz[i_mode, i_max]
            print ("mx max: ", mx_max, "mz max: ", mz_max)
            x_theta = (mx_max * exp_theta).real
            y_theta = (mz_max * exp_theta).real
            pl.plot(x_theta, y_theta, label='mode %d' % i_mode)
        pl.gca().set_aspect('equal', 'box')
        pl.legend()
        

        
    pl.figure()
    pl.plot(omega / GHz_2pi, np.abs(T1_nrf), label=r'$|T_R(\omega)|$')
    pl.plot(omega / GHz_2pi, np.abs(T2_nrf), '--', label=r'$|T_L(\omega)|$')
    pl.plot(omega / GHz_2pi, np.abs(R1_nrf), label=r'$|R_R(\omega)|$')
    pl.plot(omega / GHz_2pi, np.abs(R2_nrf), '--', label=r'$|R_L(\omega)|$')
    pl.plot(omega / GHz_2pi, A1_nrf, label=r'$|A_R(\omega)|$')
    pl.plot(omega / GHz_2pi, A2_nrf, '--', label=r'$|A_L(\omega)|$')
    ax_T = pl.gca()
    pl.xlabel(r"Frequency $\omega/2\pi$, GHz")
    pl.legend()
    pl.title("with near fields")
    pl.figure()
    pl.plot(omega / GHz_2pi, np.abs(T1), label=r'$|T_R(\omega)|$')
    pl.plot(omega / GHz_2pi, np.abs(T2), '--', label=r'$|T_L(\omega)|$')
    pl.plot(omega / GHz_2pi, np.abs(R1), label=r'$|R_R(\omega)|$')
    pl.plot(omega / GHz_2pi, np.abs(R2), '--', label=r'$|R_L(\omega)|$')
    pl.plot(omega / GHz_2pi, A1, label=r'$|A_R(\omega)|$')
    pl.plot(omega / GHz_2pi, A2, '--', label=r'$|A_L(\omega)|$')
    #ax_T = pl.gca()
    pl.xlabel(r"Frequency $\omega/2\pi$, GHz")
    pl.legend()
    pl.title("no near fields")


    Gamma_shape = np.shape(Gamma_rad)
    if False:
      pl.figure()
      if len(Gamma_shape) < 2:
       pl.plot(omega / GHz_2pi, Gamma_rad, label=r'$\Gamma_{rad}(\omega)$')
      else:
         for i in range(Gamma_shape[1]):
           pl.plot(omega / GHz_2pi, Gamma_rad[:, i, i],
                   label=r'$\Gamma_{rad}(\omega)$ mode %d' % i)
           pl.plot(omega / GHz_2pi, Gamma_R[:, i, i],
                   label=r'$\Gamma_{R}(\omega)$ mode %d' % i)
           pl.plot(omega / GHz_2pi, Gamma_L[:, i, i],
                   label=r'$\Gamma_{L}(\omega)$ mode %d' % i)
      pl.xlabel(r"Frequency $\omega/2\pi$, GHz")
      pl.ylabel(r'Radiative linewidth $\Gamma_{rad}(\omega)$, ${ns}^{-1}$')
      pl.legend()

    if len(Gamma_shape) > 2:
       for i in range(Gamma_shape[1]):
           continue
           pl.figure()
           pl.plot(omega / GHz_2pi, Gamma_rad[:, i, i],
                   label=r'$\Gamma_{rad}(\omega)$ mode %d' % i)
           pl.plot(omega / GHz_2pi, Gamma_R[:, i, i],
                   label=r'$\Gamma_{R}(\omega)$ mode %d' % i)
           pl.plot(omega / GHz_2pi, Gamma_L[:, i, i],
                   label=r'$\Gamma_{L}(\omega)$ mode %d' % i)
           pl.legend()
           pl.xlabel(r"Frequency $\omega/2\pi$, GHz")
           pl.ylabel(r'Radiative linewidth $\Gamma_{rad}(\omega)$, ${ns}^{-1}$')
           
    print_config(d)
    pl.legend()

    res_min = 3.8 * GHz_2pi
    res_max = 4.0 * GHz_2pi

    Omega_3, Gamma_R3, Gamma_L3, Gamma_3, Gamma_tot3 = resfit.fit_T_and_T(
         res_min, res_max, omega, T1_nrf, T2_nrf)

    print ("fit: ", Omega_3 / GHz_2pi, "width = ", Gamma_tot3)
    print ("   Gamma_R = ", Gamma_R3, "Gamma_L = ", Gamma_L3,
           "Gamma_0", Gamma_3)

    T1_fit = T_res(omega, Omega_3, Gamma_R3, Gamma_L3 + Gamma_3)
    T2_fit = T_res(omega, Omega_3, Gamma_L3, Gamma_R3 + Gamma_3)
    R1_fit = R_res(omega, Omega_3, Gamma_R3, Gamma_L3, Gamma_3)
    R2_fit = R_res(omega, Omega_3, Gamma_L3, Gamma_R3, Gamma_3)
    pl.figure()
    pl.plot(omega/ GHz_2pi, np.abs(T1_nrf), label='T1')
    pl.plot(omega/ GHz_2pi, np.abs(R1_nrf), label='R1')
    pl.plot(omega/ GHz_2pi, np.abs(T2_nrf), label='T2')
    pl.plot(omega/ GHz_2pi, np.abs(R2_nrf), label='R2')
    pl.plot(omega / GHz_2pi, np.abs(T1_fit), '--', label='T1 fit')
    pl.plot(omega / GHz_2pi, np.abs(T2_fit), '--', label='T2 fit')
    pl.plot(omega / GHz_2pi, np.abs(R1_fit), '--', label='R1 fit')
    pl.plot(omega / GHz_2pi, np.abs(R2_fit), '--', label='R2 fit')
    pl.legend()

    Omega_0 = np.diag(d['omega_0']) * np.eye((Gamma_shape[1]))
    #print ("Omega_0 = ", Omega_0)
    Gamma_0 = np.diag(d['Gamma_0']) * np.eye((Gamma_shape[1]))
    if  Gamma_shape[1] > 1:
        Gamma_rad_eig = np.zeros ((Gamma_shape[0], Gamma_shape[1]))
        dOmega_eig = np.zeros ((Gamma_shape[0], Gamma_shape[1]))
        #Gamma_R_eig   = np.zeros ((Gamma_shape[0], Gamma_shape[1]))
        #Gamma_L_eig   = np.zeros ((Gamma_shape[0], Gamma_shape[1]))
        for i in range(Gamma_shape[0]):
            Gamma_rad_i = Gamma_rad[i, :, :]
            Gamma_R_i = Gamma_R[i, :, :]
            Gamma_L_i = Gamma_L[i, :, :]
            #print ("f = ", omega[i] / GHz_2pi, "herm rad",
            #  linalg.norm(Gamma_rad_i - np.transpose(Gamma_rad_i.conjugate())))
            #print ("   herm R",
            #  linalg.norm(Gamma_R_i - np.transpose(Gamma_R_i.conjugate())))
            #print ("   herm L",
            #  linalg.norm(Gamma_L_i - np.transpose(Gamma_L_i.conjugate())))
            H = Omega_0 - 1j * Gamma_0 - 1j * Gamma_rad[i, :, :]
            #print ("H = ", H)
            ev_rad_i, psi_i = linalg.eig(H)
            #print ("ev = ", ev_rad_i)
            for j in range(Gamma_shape[1]):
                i_near = np.argmin(np.abs(np.diag(Omega_0) - ev_rad_i[j].real))
                #print (i, j, ev_rad_i[j], i_near)
                dOmega_eig[i, i_near] = ev_rad_i[j].real - Omega_0[i_near, i_near]
                Gamma_rad_eig[i, i_near] = -ev_rad_i[j].imag

            #ev_rad_i, psi_i = linalg.eigh(Gamma_rad_i)
            #Gamma_rad_eig[i, :] = ev_rad_i
            #ev_R_i, psi_R_i = linalg.eigh(Gamma_R_i)
            #Gamma_R_eig[i, :] = ev_R_i
            #ev_L_i, psi_L_i = linalg.eigh(Gamma_L_i)
            #Gamma_L_eig[i, :] = ev_L_i

        for i in range(Gamma_shape[1]):
            continue
            pl.figure()
            pl.plot(omega / GHz_2pi, Gamma_rad_eig[:, i],
                    label='ev Gamma_rad %d' % i)
            #pl.plot(omega / GHz_2pi, Gamma_R_eig[:, i],
            #        label='ev Gamma_R %d' % i)
            #pl.plot(omega / GHz_2pi, Gamma_L_eig[:, i],
            #        label='ev Gamma_L %d' % i)
            pl.plot(omega / GHz_2pi, Gamma_rad[:, i, i], '--',
                    label='diag, Gamma_rad %d' % i)
            pl.plot(omega / GHz_2pi, Gamma_R[:, i, i], '--',
                    label='diag, Gamma_R %d' % i)
            pl.plot(omega / GHz_2pi, Gamma_L[:, i, i], '--',
                    label='diag, Gamma_L %d' % i)
            pl.plot(omega / GHz_2pi, dOmega_eig[:, i], label='dOmega %d' % i)
            pl.legend()
        for i in range(Gamma_shape[1]):
            print ("mode", i)
            i_nr = np.argmin(np.abs(omega - Omega_0[i, i]))
            omega_res = Omega_0[i, i] + dOmega_eig[i_nr, i]
            print ("  omega_res_1 = ", omega_res / GHz_2pi)
            i_nr = np.argmin(np.abs(omega_res - Omega_0[i, i]))            
            omega_res = Omega_0[i, i] + dOmega_eig[i_nr, i]
            print ("  omega_res_2 = ", omega_res / GHz_2pi)
            i_nr = np.argmin(np.abs(omega_res - Omega_0[i, i]))            
            print ("  dOmega  = ", dOmega_eig[i_nr, i])
            print ("  Gamma_R = ", Gamma_R[i_nr, i, i])
            print ("  Gamma_L = ", Gamma_L[i_nr, i, i])
            print ("  Gamma_R in full", Gamma_R[i_nr, :])
            print ("  Gamma_L in full", Gamma_L[i_nr, :])
            print ("  Gamma_0 = ", Gamma_0[i, i])
            print ("  Gamma_rad = ", Gamma_rad[i_nr, i, i])
            print ("  Gamma_tot = ", Gamma_rad[i_nr, i, i] + Gamma_0[i, i])
            print ("  Gamma_eig = ", Gamma_rad_eig[i_nr, i])
            

    pl.show()


for fname in sys.argv[1:]:
    readData(fname)
