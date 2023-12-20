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
    T1_bare = d['T1_bare']
    R1_bare = d['R1_bare']
    A1_bare = d['A1_bare']
    T2_bare = d['T2_bare']
    R2_bare = d['R2_bare']
    A2_bare = d['A2_bare']
    
    T1_nrf = d['T1_bare']
    R1_nrf = d['R1_bare']
    A1_nrf = d['A1_bare']
    T2_nrf = d['T2_bare']
    R2_nrf = d['R2_bare']
    A2_nrf = d['A2_bare']

    show_phi = False
    if 'phi_R_bare' in d.keys():
        show_phi = True
        phi_R_bare = d['phi_R_bare']
        phibar_R_bare = d['phibar_R_bare']
        phi_L_bare = d['phi_L_bare']
        phibar_L_bare = d['phibar_L_bare']
        phi_R_nrf = d['phi_R_bare']
        phibar_R_nrf = d['phibar_R_bare']
        phi_L_nrf = d['phi_L_bare']
        phibar_L_nrf = d['phibar_L_bare']

    freq_split = [(0.5, 2.5), (2.5, 3.2), (3.2, 5.0)]
    pl.figure()
    #ax_1 = pl.gca()
    ax_1 = pl.axes([0.15, 0.1, 0.25, 0.8])
    ax_2 = pl.axes([0.65, 0.1, 0.25, 0.8])
    ax_1.set_aspect('equal', 'box')
    ax_1.set_xlabel(r"Re $T_R(\omega)$")
    ax_1.set_ylabel(r"Im $T_R(\omega)$")
    #pl.figure()
    #ax_2 = pl.gca()
    ax_2.set_aspect('equal', 'box')
    ax_2.set_xlabel(r"Re $T_L(\omega)$")
    ax_2.set_ylabel(r"Im $T_L(\omega)$")
    #pl.show()
    for f_min, f_max in freq_split:
        omega_min = f_min * GHz_2pi
        omega_max = f_max * GHz_2pi
        i_filt = [t for t in range(len(omega))
                  if omega[t] >= omega_min and omega[t] <= omega_max]
        o_filt = np.array([omega[t] for t in i_filt])
        T1_filt = np.array([T1_nrf[t] for t in i_filt])
        T2_filt = np.array([T2_nrf[t] for t in i_filt])
        ax_1.plot(T1_filt.real, T1_filt.imag,
                label=r'$%g$ GHz $\leq f \leq$ $%g$GHz'
                % (f_min, f_max))
        ax_2.plot(T2_filt.real, T2_filt.imag,
                label=r'$%g$ GHz $\leq f \leq$ $%g$GHz'
                % (f_min, f_max))
    ax_1.legend(loc='upper right', bbox_to_anchor=(2.0, 1.7))
    #ax_2.legend()
    
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
        


    if False:
        pl.figure()
        pl.title("comparison: bare vs nrf")
        p = pl.plot(omega / GHz_2pi, np.abs(T1_nrf), label='|T1_nrf|')
        p = pl.plot(omega / GHz_2pi, np.abs(T1_bare), '--', label='|T1_bare|')
        p = pl.plot(omega / GHz_2pi, np.abs(T2_nrf), label='|T2_nrf|')
        p = pl.plot(omega / GHz_2pi, np.abs(T2_bare), '--', label='|T2_bare|')
        pl.legend()
        

    if True:
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



    if False:
        pl.figure()
        #ax_abs = pl.gca()
        ax_abs = pl.axes([0.1, 0.1, 0.75, 0.8])
        pl.title("nrf")
        ax_phase = pl.gca().twinx()
        p1 = ax_abs.plot(omega/ GHz_2pi, np.abs(T1_nrf), label='|T1|')
        p2 = ax_abs.plot(omega/ GHz_2pi, np.abs(T2_nrf), label='|T2|')
        ax_abs.plot(omega/ GHz_2pi, np.abs(R1_nrf), label='|R1|')
        #ax_abs.plot(omega/ GHz_2pi, np.abs(R2_nrf), label='|R2|')
        ax_phase.plot(omega/ GHz_2pi, np.angle(T1_nrf)*180.0/np.pi,
                      '--', color=p1[0].get_color(), label='arg T1')
        ax_phase.plot(omega/ GHz_2pi, np.angle(T2_nrf)*180.0/np.pi,
                      '--', color=p2[0].get_color(), label='arg T2')
        ax_abs.set_ylim(0.0, 1.01)
        ax_phase.set_ylim(-180.0, 180.0)
        ax_phase.set_yticks([-180.0, -90.0, 0.0, 90.0, 180.0])
        #pl.plot(omega / GHz_2pi, np.abs(T1_fit), '--', label='T1 fit')
        #pl.plot(omega / GHz_2pi, np.abs(T2_fit), '--', label='T2 fit')
        #pl.plot(omega / GHz_2pi, np.abs(R1_fit), '--', label='R1 fit')
        #pl.plot(omega / GHz_2pi, np.abs(R2_fit), '--', label='R2 fit')
        ax_abs.legend()
        pl.xlabel(r"Frequency $\omega /2\pi$, GHz")
        #pl.ylabel(r"Transmissivigy, reflectivity $|T(\omega)|$, $|R(\omega)|$")
        ax_abs.set_ylabel(r"Transmissivigy, reflectivity $|T(\omega)|$, $|R(\omega)|$")
        ax_phase.set_ylabel(r"Phase of transmissitivyt ${\rm arg} T(\omega)$")

    pl.figure()
    #ax_abs = pl.gca()
    ax_abs = pl.axes([0.1, 0.1, 0.75, 0.8])
    pl.title("bare")
    ax_phase = pl.gca().twinx()
    p1 = ax_abs.plot(omega/ GHz_2pi, np.abs(T1_bare), label='|T1|')
    p2 = ax_abs.plot(omega/ GHz_2pi, np.abs(T2_bare), label='|T2|')
    #ax_abs.plot(omega/ GHz_2pi, np.abs(R1_bare), label='|R1|')
    #ax_abs.plot(omega/ GHz_2pi, np.abs(R2_bare), label='|R2|')
    ax_phase.plot(omega/ GHz_2pi, np.angle(T1_bare)*180.0/np.pi,
                  '--', color=p1[0].get_color(), label='arg T1')
    ax_phase.plot(omega/ GHz_2pi, np.angle(T2_bare)*180.0/np.pi,
                  '--', color=p2[0].get_color(), label='arg T2')
    ax_abs.set_ylim(0.0, 1.01)
    ax_phase.set_ylim(-180.0, 180.0)
    ax_phase.set_yticks([-180.0, -90.0, 0.0, 90.0, 180.0])
    #pl.plot(omega / GHz_2pi, np.abs(T1_fit), '--', label='T1 fit')
    #pl.plot(omega / GHz_2pi, np.abs(T2_fit), '--', label='T2 fit')
    #pl.plot(omega / GHz_2pi, np.abs(R1_fit), '--', label='R1 fit')
    #pl.plot(omega / GHz_2pi, np.abs(R2_fit), '--', label='R2 fit')
    ax_abs.legend()
    pl.xlabel(r"Frequency $\omega /2\pi$, GHz")
    #pl.ylabel(r"Transmissivigy, reflectivity $|T(\omega)|$, $|R(\omega)|$")
    ax_abs.set_ylabel(r"Transmissivigy, reflectivity $|T(\omega)|$, $|R(\omega)|$")
    ax_phase.set_ylabel(r"Phase of transmissitivyt ${\rm arg} T(\omega)$")

    def do_smear(T, df):
        from scipy import fft
        T_fft  = fft.fftshift(fft.fft(T))
        T_fft_old = np.copy(T_fft)
        o_fft  = fft.fftshift(fft.fftfreq(len(omega), omega[1] - omega[0]))
        do = df * GHz_2pi
        T_fft *= np.exp( - o_fft**2  * do**2 / 2.0)
        T_new = fft.ifft(fft.ifftshift(T_fft))
        if False:
            pl.figure()
            pl.plot(o_fft, np.abs(T_fft_old))
            pl.plot(o_fft, np.abs(T_fft))
            pl.plot(o_fft,  np.exp( - o_fft**2  * do**2 / 2.0))
            pl.figure()
            pl.plot(o_fft, np.abs(T))
            pl.plot(o_fft, np.abs(T_new))
            pl.show()
        return T_new

    df = 0.2
    T1_nrf_av = do_smear(T1_nrf, df)
    T2_nrf_av = do_smear(T2_nrf, df)
    R1_nrf_av = do_smear(R1_nrf, df)
    
    if False:
        pl.figure()
        pl.title("smeared transmissivity")
        #ax_abs = pl.gca()
        ax_abs = pl.axes([0.1, 0.1, 0.75, 0.8])
        ax_phase = pl.gca().twinx()
        p1 = ax_abs.plot(omega/ GHz_2pi, np.abs(T1_nrf_av), label='|T1|')
        p2 = ax_abs.plot(omega/ GHz_2pi, np.abs(T2_nrf_av), label='|T2|')
        ax_abs.plot(omega/ GHz_2pi, np.abs(R1_nrf_av), label='|R1|')
        #ax_abs.plot(omega/ GHz_2pi, np.abs(R2_nrf), label='|R2|')
        ax_phase.plot(omega/ GHz_2pi, np.angle(T1_nrf_av)*180.0/np.pi,
                      '--', color=p1[0].get_color(), label='arg T1')
        ax_phase.plot(omega/ GHz_2pi, np.angle(T2_nrf_av)*180.0/np.pi,
                      '--', color=p2[0].get_color(), label='arg T2')
        ax_abs.set_ylim(0.0, 1.01)
        ax_phase.set_ylim(-180.0, 180.0)
        ax_phase.set_yticks([-180.0, -90.0, 0.0, 90.0, 180.0])
        #pl.plot(omega / GHz_2pi, np.abs(T1_fit), '--', label='T1 fit')
        #pl.plot(omega / GHz_2pi, np.abs(T2_fit), '--', label='T2 fit')
        #pl.plot(omega / GHz_2pi, np.abs(R1_fit), '--', label='R1 fit')
        #pl.plot(omega / GHz_2pi, np.abs(R2_fit), '--', label='R2 fit')
        ax_abs.legend()
        pl.xlabel(r"Frequency $\omega /2\pi$, GHz")
        #pl.ylabel(r"Transmissivigy, reflectivity $|T(\omega)|$, $|R(\omega)|$")
        ax_abs.set_ylabel(r"Transmissivigy, reflectivity $|T(\omega)|$, $|R(\omega)|$")
        ax_phase.set_ylabel(r"Phase of transmissitivyt ${\rm arg} T(\omega)$")

    if False:
        pl.figure()
        pl.plot(omega/ GHz_2pi, np.angle(T1_nrf) * 180.0/np.pi, label='T1')
        #pl.plot(omega/ GHz_2pi, np.angle(R1_nrf), label='R1')
        pl.plot(omega/ GHz_2pi, np.angle(T2_nrf) * 180.0/np.pi, label='T2')
        #pl.plot(omega/ GHz_2pi, np.angle(R2_nrf), label='R2')
        pl.ylim(-180.0, 180.0)
        pl.gca().set_yticks([-180.0, -90.0, 0.0, 90.0, 180.0])
        pl.ylabel(r"Phase of transmissitivyt ${\rm arg} T(\omega)$")
        pl.legend()

    if show_phi:
        Nphi, Mphi = np.shape(phi_R_nrf)
        if False:
            pl.figure()
            pl.title("phi nrf")
            print ("Nphi, Mphi = ", Nphi, Mphi)
            for i in range(Mphi):
                p = pl.plot(omega / GHz_2pi, np.abs(phi_R_nrf[:, i]),
                            label='mode %d' % i)
                pl.plot(omega / GHz_2pi, np.abs(phi_L_nrf[:, i]), '--',
                        color = p[0].get_color())
            pl.legend()
            pl.figure()
            pl.title("phibar nrf")
            for i in range(Mphi):
                p = pl.plot(omega / GHz_2pi, np.abs(phibar_R_nrf[:, i]),
                            label='mode %d' % i)
                pl.plot(omega / GHz_2pi, np.abs(phibar_L_nrf[:, i]), '--',
                        color = p[0].get_color())
            pl.legend()
            pl.figure()
            pl.title("phi bare")
            Nphi, Mphi = np.shape(phi_R_bare)
            print ("Nphi, Mphi = ", Nphi, Mphi)
            for i in range(Mphi):
                p = pl.plot(omega / GHz_2pi, np.abs(phi_R_bare[:, i]),
                            label='mode %d' % i)
                pl.plot(omega / GHz_2pi, np.abs(phi_L_bare[:, i]), '--',
                        color = p[0].get_color())
            pl.legend()
        Nphi, Mphi = np.shape(phi_R_bare)
        pl.figure()
        pl.title("phibar bare")
        for i in range(Mphi):
            p = pl.plot(omega / GHz_2pi, np.abs(phibar_R_bare[:, i]),
                        label='mode %d' % i)
            pl.plot(omega / GHz_2pi, np.abs(phibar_L_bare[:, i]), '--',
                    color = p[0].get_color())
        pl.legend()

        #Y, X = np.meshgrid(np.array(range(Mphi)), omega / GHz_2pi)

        mode_freq = d['res_modes_freq']
        Y, X = np.meshgrid(mode_freq.real / GHz_2pi, omega / GHz_2pi)

        from matplotlib.colors import LogNorm
        pl.figure()
        pl.pcolormesh(X, Y, np.abs(phi_R_bare[:-1, :-1]), shading='auto',
                  norm=LogNorm(vmin=1e-3, vmax=0.5))
        pl.colorbar()
        pl.title("phi_R")
        
        pl.figure()
        pl.pcolormesh(X, Y, np.abs(phibar_R_bare[:-1, :-1]), shading='auto',
                  norm=LogNorm(vmin=1e-3, vmax=0.5))
        pl.colorbar()
        pl.title("phi_R bar")

        pl.figure()
        pl.pcolormesh(X, Y, np.abs(phi_L_bare[:-1, :-1]), shading='auto',
                  norm=LogNorm(vmin=1e-3, vmax=0.5))
        pl.colorbar()
        pl.title("phi_L")
        pl.figure()
        pl.pcolormesh(X, Y, np.abs(phibar_L_bare[:-1, :-1]), shading='auto',
                      norm=LogNorm(vmin=1e-3, vmax=0.5))
        pl.colorbar()
        pl.title("phi_L bar")
        
    pl.show()


for fname in sys.argv[1:]:
    readData(fname)
