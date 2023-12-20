import numpy as np
import pylab as pl
from scipy import interpolate, linalg
import sys

import constants

def show_analysis_old(fname):
    d = np.load(fname)
    for k in d.keys():
        print (k)

    omega_m = d['omega_m']
    Gamma_m = d['Gamma_m']
    Sigma_m = d['Sigma_m']

    Omega_res = d['Omega_res']
    Gamma_res = d['Gamma_res']
    Z_res     = d['Z_res']
    Omega_0   = d['omega'].real
    Gamma_0   = -d['omega'].imag
    omega_min = d['omega_min']


    Gamma_spl = interpolate.splrep(omega_m, Gamma_m)
    Sigma_spl = interpolate.splrep(omega_m, Sigma_m)

    if 'Sigma_m_Gamma' in d.keys():
        Sigma_m_Gamma = d['Sigma_m_Gamma']
        Sigma_m_prime = d['Sigma_m_prime']
    else:
        Sigma_m_Gamma = Sigma_m
        Sigma_m_prime = 0.0 * omega_m

    def _Gamma_s(omega):
        if omega < omega_min: return 0.0
        return interpolate.splev(omega, Gamma_spl)

    Gamma_s = np.vectorize(_Gamma_s)

    def Sigma_s(omega):
        return interpolate.splev(omega, Sigma_spl)

    print ("Omega_0 = ",   Omega_0   / constants.GHz_2pi)
    print ("Gamma_0 = ", Gamma_0)
    print ("Omega_res = ", Omega_res / constants.GHz_2pi)
    print ("Gamma_res = ", Gamma_res)
    #print ("Sigma\' = ", Sigma_prime)
    print ("Z_res = ", Z_res)

    #pl.figure()
    #pl.plot(omega_m / constants.GHz_2pi,
    #        Gamma_m, label=r'$\Gamma_{\rm rad}(\omega)$')
    #pl.plot(omega_m / constants.GHz_2pi,
    #        Sigma_m, label=r'Re $\Sigma (\omega)$')

    p = pl.plot(omega_m / constants.GHz_2pi, Gamma_s(omega_m),
            label=r'$\Gamma_{\rm rad}(\omega)$')
    #p = pl.plot(omega_coarse / constants.GHz_2pi, Gamma_coarse, 'o',
    #            label=r'Tabulation points',
    #            markeredgecolor = p[0].get_color(),
    #            markerfacecolor = 'yellow')
    pl.plot(omega_m / constants.GHz_2pi, Sigma_s(omega_m),
            label=r'Re $\Sigma(\omega)$')
    pl.plot(omega_m / constants.GHz_2pi, Sigma_m_Gamma, '--',
            label=r'Re $\Sigma_\Gamma(\omega)$')
    pl.plot(omega_m / constants.GHz_2pi, Sigma_m_prime.real, '--',
            label=r'Re $\Sigma\'(\omega)$')
    pl.plot(omega_m / constants.GHz_2pi, Sigma_m_prime.imag, '--',
            label=r'Im $\Sigma\'(\omega)$')
    pl.plot(np.array([Omega_res, Omega_res]) / constants.GHz_2pi,
            np.array([Gamma_s(Omega_res), Sigma_s(Omega_res)]),
            'r--', linestyle='dotted')
    pl.plot(np.array([Omega_res]) / constants.GHz_2pi,
            np.array([Gamma_s(Omega_res)]), 'o',
            markeredgecolor='red', markerfacecolor='yellow')
    pl.plot(np.array([Omega_res]) / constants.GHz_2pi,
            np.array([Sigma_s(Omega_res)]), 'o',
            markeredgecolor='red', markerfacecolor='yellow')
    #pl.plot(np.array([Omega_res, Omega_res]) / constants.GHz_2pi,
    #        np.array([Gamma_s(Omega_0), Sigma_s(Omega_0)]),
    #        'k--')
    pl.plot(np.array([Omega_0]) / constants.GHz_2pi,
            np.array([Gamma_s(Omega_0)]), 'o',
            markeredgecolor='black', markerfacecolor='white')
    pl.plot(np.array([Omega_0]) / constants.GHz_2pi,
            np.array([Sigma_s(Omega_0)]), 'o',
            markeredgecolor='black', markerfacecolor='white')
    #pl.plot(omega_m / constants.GHz_2pi, 0.0, 'k--')
    xmin, xmax = pl.xlim()
    pl.plot([xmin, xmax], [0.0, 0.0], 'k--')
    pl.xlim(xmin, xmax)
    pl.xlabel(r"Frequency $\omega/2\pi$, GHz")
    pl.ylabel(r"Self-energy $\Sigma(\omega)$, ${\rm ns}^{-1}$")
    #pl.plot(np.array([Omega_res, Omega_res]) / constants.GHz_2pi, [0.0, np.max(Gamma_coarse)], 'r--')
    #pl.plot(np.array([Omega_0, Omega_0]) / constants.GHz_2pi, [0.0, np.max(Gamma_coarse)], 'k--')
    pl.legend()
    
    pl.show()

def show_analysis_new2(fname):
    d = np.load(fname)
    for k in d.keys():
        print (k)

    omega_m = d['omega_m']
    Gamma_m = d['Gamma_m']
    Gamma_m_prime = d['Gamma_m_prime']
    Sigma_m = d['Sigma_m']
    Sigma_m_prime = d['Sigma_m_prime']
    Sigma_m_Gamma = d['Sigma_m_Gamma']

    Omega_res = d['Omega_res']
    Gamma_res = d['Gamma_res']
    Z_res     = d['Z_res']
    Omega_0   = d['omega'].real
    Gamma_0   = -d['omega'].imag
    omega_min = d['omega_min']

    Gamma_spl = interpolate.splrep(omega_m, Gamma_m)
    Gamma_prime_spl = interpolate.splrep(omega_m, Gamma_m)
    Sigma_spl = interpolate.splrep(omega_m, Sigma_m)
    Sigma_prime_spl = interpolate.splrep(omega_m, Sigma_m_prime)

    #if 'Sigma_m_Gamma' in d.keys():
    #    Sigma_m_Gamma = d['Sigma_m_Gamma']
    #    Sigma_m_prime = d['Sigma_m_prime']
    #else:
    #    Sigma_m_Gamma = Sigma_m
    #    Sigma_m_prime = 0.0 * omega_m

    def _Gamma_s(omega):
        if omega < omega_min: return 0.0
        return interpolate.splev(omega, Gamma_spl)
    
    def _Gamma_prime_s(omega):
        if omega < omega_min: return 0.0
        return interpolate.splev(omega, Gamma_spl)

    Gamma_s = np.vectorize(_Gamma_s)
    Gamma_prime_s = np.vectorize(_Gamma_prime_s)

    def Sigma_s(omega):
        return interpolate.splev(omega, Sigma_spl)

    def Sigma_prime_s(omega):
        return interpolate.splev(omega, Sigma_prime_spl)

    def Sigma_Gamma_s(omega):
        return Sigma_s(omega) - Sigma_prime_s(omega)

    print ("Omega_0 = ",   Omega_0   / constants.GHz_2pi)
    print ("Gamma_0 = ", Gamma_0)
    print ("Omega_res = ", Omega_res / constants.GHz_2pi)
    print ("Gamma_res = ", Gamma_res)
    #print ("Sigma\' = ", Sigma_prime)
    print ("Z_res = ", Z_res)

    #pl.figure()
    #pl.plot(omega_m / constants.GHz_2pi,
    #        Gamma_m, label=r'$\Gamma_{\rm rad}(\omega)$')
    #pl.plot(omega_m / constants.GHz_2pi,
    #        Sigma_m, label=r'Re $\Sigma (\omega)$')

    p = pl.plot(omega_m / constants.GHz_2pi, Gamma_s(omega_m),
            label=r'$\Gamma_{\rm rad}(\omega)$')
    #p = pl.plot(omega_coarse / constants.GHz_2pi, Gamma_coarse, 'o',
    #            label=r'Tabulation points',
    #            markeredgecolor = p[0].get_color(),
    #            markerfacecolor = 'yellow')
    pl.plot(omega_m / constants.GHz_2pi, Sigma_s(omega_m),
            label=r'Re $\Sigma(\omega)$')
    pl.plot(omega_m / constants.GHz_2pi, Sigma_m_Gamma, '--',
            label=r'Re $\Sigma_\Gamma(\omega)$')
    pl.plot(omega_m / constants.GHz_2pi, Sigma_m_prime, '--',
            label=r'Re $\Sigma\'(\omega)$')
    pl.plot(-omega_m / constants.GHz_2pi, -Gamma_m_prime, '--',
            label=r'Im $\Sigma\'(\omega)$')
    ##pl.plot(omega_m / constants.GHz_2pi, Sigma_m_prime.real, '--',
    ##        label=r'Re $\Sigma\'(\omega)$')
    ##pl.plot(omega_m / constants.GHz_2pi, Sigma_m_prime.imag, '--',
    ##        label=r'Im $\Sigma\'(\omega)$')
    pl.plot(np.array([Omega_res, Omega_res]) / constants.GHz_2pi,
            np.array([Gamma_s(Omega_res), Sigma_s(Omega_res)]),
            'r:', linestyle='dotted')
    pl.plot(np.array([Omega_res]) / constants.GHz_2pi,
            np.array([Gamma_s(Omega_res)]), 'o',
            markeredgecolor='red', markerfacecolor='yellow')
    pl.plot(np.array([Omega_res]) / constants.GHz_2pi,
            np.array([Sigma_s(Omega_res)]), 'o',
            markeredgecolor='red', markerfacecolor='yellow')
    #pl.plot(np.array([Omega_res, Omega_res]) / constants.GHz_2pi,
    #        np.array([Gamma_s(Omega_0), Sigma_s(Omega_0)]),
    #        'k--')
    pl.plot(np.array([Omega_0]) / constants.GHz_2pi,
            np.array([Gamma_s(Omega_0)]), 'o',
            markeredgecolor='black', markerfacecolor='white')
    pl.plot(np.array([Omega_0]) / constants.GHz_2pi,
            np.array([Sigma_s(Omega_0)]), 'o',
            markeredgecolor='black', markerfacecolor='white')
    #pl.plot(omega_m / constants.GHz_2pi, 0.0, 'k--')
    xmin, xmax = pl.xlim()
    pl.plot([xmin, xmax], [0.0, 0.0], 'k--')
    pl.xlim(xmin, xmax)
    pl.xlabel(r"Frequency $\omega/2\pi$, GHz")
    pl.ylabel(r"Self-energy $\Sigma(\omega)$, ${\rm ns}^{-1}$")
    #pl.plot(np.array([Omega_res, Omega_res]) / constants.GHz_2pi, [0.0, np.max(Gamma_coarse)], 'r--')
    #pl.plot(np.array([Omega_0, Omega_0]) / constants.GHz_2pi, [0.0, np.max(Gamma_coarse)], 'k--')
    pl.legend(loc=1)
    
    pl.show()


def show_analysis_new3(fname):
    d = np.load(fname)
    for k in d.keys():
        print (k)

    omega_m = d['omega_m']
    Gamma_m = d['Gamma_m']
    Gamma_m_prime = d['Gamma_m_prime']
    Gamma_m_tilde = d['Gamma_m_tilde']
    #Gamma_m_tilde = -d['Sigma_m_mp'].imag
    Sigma_m = d['Sigma_m']
    Sigma_m_prime = d['Sigma_m_prime']
    Sigma_m_Gamma = d['Sigma_m_Gamma']

    Omega_res = d['Omega_res']
    Gamma_res = d['Gamma_res']
    Z_res     = d['Z_res']
    Omega_0   = d['omega'].real
    Gamma_0   = -d['omega'].imag
    print ("d[omega] = ", d['omega'])
    print ("d[Gamma_0]", d['Gamma_0'])
    omega_min = d['omega_min']

    print ("a - 2")
    Delta_m_inc       = d['inc:Delta_inc']
    Delta_m_inc_prime = d['inc:Delta_inc_prime']
    theta_inc = d['inc:theta_inc']
    v_m_inc = d['inc:v_inc']
    v_m_0 = v_m_inc[:, 0]
    Delta_m_0 = Delta_m_inc[:, 0]
    Delta_m_prime_0 = Delta_m_inc_prime[:, 0]

    print ("a-1.9")
    Gamma_spl = interpolate.splrep(omega_m, Gamma_m)
    print ("a-1.8")
    print ("Gamma_m_prime = ", Gamma_m_prime)
    Gamma_prime_spl = interpolate.splrep(omega_m, Gamma_m_prime)
    print ("a-1.7")
    Gamma_tilde_spl_re = interpolate.splrep(omega_m, Gamma_m_tilde.real)
    print ("a-1.6")
    Gamma_tilde_spl_im = interpolate.splrep(omega_m, Gamma_m_tilde.imag)
    print ("a-1.5")
    Sigma_spl = interpolate.splrep(omega_m, Sigma_m)
    #Sigma_prime_spl = interpolate.splrep(omega_m, Sigma_m_prime)

    #if 'Sigma_m_Gamma' in d.keys():
    #    Sigma_m_Gamma = d['Sigma_m_Gamma']
    #    Sigma_m_prime = d['Sigma_m_prime']
    #else:
    #    Sigma_m_Gamma = Sigma_m
    #    Sigma_m_prime = 0.0 * omega_m

    print ("a-1")
    def _Gamma_s(omega):
        if omega < omega_min: return 0.0
        return interpolate.splev(omega, Gamma_spl)
    
    def _Gamma_prime_s(omega):
        if omega < omega_min: return 0.0
        return interpolate.splev(omega, Gamma_prime_spl)
    
    def _Gamma_tilde_s(omega):
        if omega < omega_min: return 0.0 + 0.0j
        ret = interpolate.splev(omega, Gamma_tilde_spl_re) + 0.0j
        ret += 1j * interpolate.splev(omega, Gamma_tilde_spl_im)
        return ret

    def make_cmplx_spl(omega_m, F_m):
        F_re_spl = interpolate.splrep(omega_m, F_m.real)
        F_im_spl = interpolate.splrep(omega_m, F_m.imag)
        def F_s(omega):
            ret  = interpolate.splev(omega, F_re_spl)
            ret += 1j * interpolate.splev(omega, F_im_spl)
        return F_s

    Gamma_s = np.vectorize(_Gamma_s)
    Gamma_prime_s = np.vectorize(_Gamma_prime_s)
    Gamma_tilde_s = np.vectorize(_Gamma_tilde_s)
    print ("a")

    Sigma_m_pp = d['Sigma_m_pp']
    Sigma_m_mm = d['Sigma_m_mm']
    Sigma_m_pm = d['Sigma_m_pm']
    Sigma_m_mp = d['Sigma_m_mp']

    Sigma_s_pp = make_cmplx_spl(omega_m, Sigma_m_pp)
    Sigma_s_mm = make_cmplx_spl(omega_m, Sigma_m_mm)
    Sigma_s_pm = make_cmplx_spl(omega_m, Sigma_m_pm)
    Sigma_s_mp = make_cmplx_spl(omega_m, Sigma_m_mp)

    def Sigma_s(omega):
        return interpolate.splev(omega, Sigma_spl)

    def Sigma_prime_s(omega):
        return interpolate.splev(omega, Sigma_prime_spl)

    def Sigma_Gamma_s(omega):
        return Sigma_s(omega) - Sigma_prime_s(omega)

    print ("b")
    print ("Omega_0 = ",   Omega_0   / constants.GHz_2pi)
    print ("Gamma_0 = ", Gamma_0)
    print ("Omega_res = ", Omega_res / constants.GHz_2pi)
    print ("Gamma_res = ", Gamma_res)
    #print ("Sigma\' = ", Sigma_prime)
    print ("Z_res = ", Z_res)


    GHz_2pi = constants.GHz_2pi
    pl.figure()
    pl.plot(omega_m / GHz_2pi, Gamma_m, label=r'$\Gamma_{\rm rad}(\omega)$')
    pl.plot(omega_m / GHz_2pi, Gamma_m_prime, label=r'$\Gamma\'(\omega)$')
    pl.plot(omega_m / GHz_2pi, np.abs(Gamma_m_tilde),
            label=r'$|{\tilde \Gamma}(\omega)|$')
    pl.plot([Omega_res / GHz_2pi], Gamma_s(Omega_res), 'o',
            markerfacecolor='yellow', markeredgecolor='red')
    pl.plot([Omega_res / GHz_2pi], Gamma_prime_s(Omega_res), 'o',
            markerfacecolor='yellow', markeredgecolor='red')
    pl.plot([Omega_res / GHz_2pi], np.abs(Gamma_tilde_s(Omega_res)), 'o',
            markerfacecolor='yellow', markeredgecolor='red')
    pl.legend()
    pl.xlabel(r'Frequency $\omega/2\pi$, GHz')
    pl.ylabel(r'Rate $\Gamma$, ${\rm ns}^{-1}$')

    pl.figure()
    pl.plot(omega_m / GHz_2pi, Sigma_m, label=r'Re $\Sigma(\omega)$')
    pl.plot(omega_m / GHz_2pi, Sigma_m_Gamma, '--',
            label=r'Re $\Sigma_\Gamma(\omega)$')
    pl.plot(omega_m / GHz_2pi, Sigma_m_prime, '--',
            label=r'Re $\Sigma\'(\omega)$')
    pl.plot(omega_m / GHz_2pi, Gamma_s(omega_m),
            label=r'-Im $\Sigma(\omega)$')
    pl.plot([Omega_res / GHz_2pi], [Sigma_s(Omega_res)], 'o',
            markerfacecolor='yellow', markeredgecolor='red')
    pl.plot([Omega_res / GHz_2pi], [Gamma_s(Omega_res)], 'o',
            markerfacecolor='yellow', markeredgecolor='red')
    pl.legend()
    pl.xlabel(r'Frequency $\omega/2\pi$, GHz')
    pl.ylabel(r'Rate $\Gamma$, ${\rm ns}^{-1}$')

    pl.figure()
    M_pp = omega_m - Omega_0 + 1j * Gamma_0 - Sigma_m_pp
    M_mm = omega_m + Omega_0 + 1j * Gamma_0 + Sigma_m_mm
    M_pm = - Sigma_m_pm
    M_mp =   Sigma_m_mp
    det_M = M_pp * M_mm - M_mp * M_pm
    src_p = -1j
    src_m = -1j
    phi_p = (src_p * M_mm - src_m * M_pm) / det_M
    phi_m = (src_m * M_pp - src_p * M_mp) / det_M
    m_z = (phi_p - phi_m)/1j
    #M2_z = Gamma_m * np.abs(phi_p)**2
    M2_z = Gamma_m * np.abs(phi_p)**2 + Gamma_m_prime * np.abs(phi_m)**2
    M2_z += (Gamma_m_tilde.conj() * phi_p.conj() * phi_m).real
    M2_z += (Gamma_m_tilde        * phi_p        * phi_m.conj()).real
    m0_z = -1j / M_pp
    M0_z = np.abs(m0_z)**2 * Gamma_m
    m0_max = np.max(np.abs(m0_z)**2)
    M0_max = np.max(np.abs(M0_z))
    m_max = np.max(np.abs(m_z)**2)
    M_max = np.max(M2_z)
    pl.plot(omega_m / GHz_2pi, np.abs(m0_z)**2 / m0_max,
            label=r'$|m_{z, 0}|(\omega)|^2$')
    pl.plot(omega_m / GHz_2pi, np.abs(M0_z) / M0_max,
            label=r'$|M_{z, 0}|(\omega)|^2$')
    pl.plot(omega_m / GHz_2pi, np.abs(m_z)**2 / m_max,
            label=r'$|m_z|(\omega)|^2$')
    pl.plot(omega_m / GHz_2pi, M2_z / M_max,
            label=r'$|M_z|(\omega)|^2$')
    pl.xlabel(r"Frequency $\omega/2\pi$, GHz")
    pl.ylabel(r"$|m_z(\omega)|^2$, a.u")
    pl.xlim(1.0, 5.0)
    pl.ylim(0.0, 1.1)
    pl.legend()
    
    pl.figure()
    peak_0 = 1.0 / (omega_m - Omega_0 + 1j * Gamma_0 - Sigma_m + 1j * Gamma_m)
    peak_0 /= np.max(np.abs(peak_0))
    M_pp = omega_m - Omega_0 + 1j * Gamma_0 - Sigma_m_pp
    M_mm = omega_m + Omega_0 + 1j * Gamma_0 + Sigma_m_mm
    M_pm = - Sigma_m_pm
    M_mp =   Sigma_m_mp
    det_M = M_pp * M_mm - M_mp * M_pm
    peak_1 = 1.0 / det_M
    peak_1 /= np.max(np.abs(peak_1))
    b_p = Delta_m_0.conj()
    b_m = -Delta_m_prime_0
    print ("Delta_m_0, Delta_m_prime_0", Delta_m_0, Delta_m_prime_0)
    print (d['inc:Delta_inc'], d['inc:Delta_inc_prime'])
    phi     = (b_p * M_mm - b_m * M_pm) / det_M
    phi_bar = (b_m * M_pp - b_p * M_mp) / det_M
    psi2  = Gamma_m * np.abs(phi)**2 + Gamma_m_prime * np.abs(phi_bar)**2
    psi2 += 2.0 * (Gamma_m_tilde * phi * phi_bar.conj()).real
    peak_2 = psi2
    peak_2 /= np.max(np.abs(psi2))
    pl.plot(omega_m / GHz_2pi, np.abs(peak_0),
            label=r'resonant peak, no $\bar{\varphi}$')
    pl.plot(omega_m / GHz_2pi, np.abs(peak_1),
            label=r'resonant peak, with $\bar{\varphi}$')
    pl.plot(omega_m / GHz_2pi, np.abs(peak_2),
            label=r'resonant peak in total cross-section')

    pl.xlabel(r'Frequency $\omega/2\pi$, GHz')
    pl.ylabel(r'Response, a.u.')
    pl.legend()

    pl.figure()
    pl.polar(d['scat:theta_vals'], d['scat:k'])
    pl.title(r'Constant-frequency contour $k_\omega(\theta)$')
    pl.figure()
    pl.polar(d['scat:theta_vals'], np.abs(d['scat:Delta']))
    pl.title(r'k-space directivity $|\Delta_\theta|$')
    pl.figure()
    pl.polar(d['scat:alpha'], np.abs(d['scat:Delta']))
    pl.title(r'Real-space directivity $|\Delta_\alpha|$')


    M_pp = omega_m - Omega_0 + 1j * Gamma_0 - Sigma_m_pp
    M_mm = omega_m + Omega_0 + 1j * Gamma_0 + Sigma_m_mm
    M_pm = - Sigma_m_pm
    M_mp =   Sigma_m_mp
    det_M = M_pp * M_mm - M_mp * M_pm
    nm = constants.nm
    Gamma_tot = Gamma_0 + Gamma_m
    for theta_cur in [0, np.pi/2.0, np.pi, -np.pi/2.0]:
      j_inc = np.argmin(np.abs(np.exp(1j * theta_cur)
                               - np.exp(1j * theta_inc)))
      b_p = Delta_m_inc[:, j_inc].conj()
      b_m = -Delta_m_inc_prime[:, j_inc]
      v_in = v_m_inc[:, j_inc]
      phi     = (b_p * M_mm - b_m * M_pm) / det_M
      phi_bar = (b_m * M_pp - b_p * M_mp) / det_M
      print ("M_pp = ", M_pp, "M_pm = ", M_pm, "M_mp = ",
             M_mp, "M_mm = ", M_mm)
      print ("b_p = ", b_p, "b_m = ", b_m)
      print ("check phi+", linalg.norm(M_pp * phi + M_pm * phi_bar - b_p))
      print ("check phi+", linalg.norm(M_mp * phi + M_mm * phi_bar - b_m))
      psi2  = Gamma_m * np.abs(phi)**2 + Gamma_m_prime * np.abs(phi_bar)**2
      psi2 += 2.0 * (Gamma_m_tilde * phi * phi_bar.conj()).real
      sigma_tot = 0.0
      v_in += 1e-10
      print ('v_in = ', v_in)
      sigma_tot = psi2 * 2.0 / v_in
      
      sigma_abs = sigma_tot * Gamma_0 / (Gamma_m + 1e-10)
      sigma_abs[Gamma_m < 1e-6] = 0.0
      
      simple_denom =  omega_m - Omega_0 + 1j * Gamma_0
      simple_denom += - Sigma_m + 1j * Gamma_m
      phi_simple = Delta_m_inc[:, j_inc].conj () / simple_denom
      sigma_tot_simple = 2.0 / v_in * np.abs(phi_simple)**2 * Gamma_m 
      sigma_abs_simple = Gamma_0 / (Gamma_m + 1e-10) * sigma_tot_simple
      sigma_abs_simple[Gamma_m < 1e-6] = 0.0

      pl.figure()
      pl.title(r"cross-section for $\theta_{\rm inc} = %g$"
               % (theta_cur * 180.0/np.pi))
      p = pl.plot(omega_m / GHz_2pi,
              sigma_tot / nm, label=r'$\sigma_{\rm tot}(\omega)$')
      p = pl.plot(omega_m / GHz_2pi,
                  sigma_tot_simple / nm, '--',
                  label=r'$\sigma_{\rm tot}(\omega)$ simplified',
                  color=p[0].get_color())
      p = pl.plot(omega_m / GHz_2pi,
                  sigma_abs / nm,
                  label=r'$\sigma_{\rm abs}(\omega)$')
                  #color = p[0].get_color())
      p = pl.plot(omega_m / GHz_2pi,
                  sigma_abs_simple / nm, '--',
                  label=r'$\sigma_{\rm abs}(\omega)$ simplified',
                  color=p[0].get_color())
      #ratio = sigma_tot_simple / sigma_tot
      #ratio[omega_m < omega_min] = 0.0
      #pl.plot(omega_m / GHz_2pi, ratio * 1000, 'k-')
      #pl.plot(omega_m / GHz_2pi, np.abs(phi_bar/phi)*1000)
      #pl.plot(omega_m / GHz_2pi, np.abs(phi_simple/phi)*1000)
      pl.legend()
      pl.xlabel(r'Frequency $\omega/2\pi$, GHz')
      pl.ylabel(r'Cross-section $\sigma(\omega)$, nm')
    

    
    pl.figure()
    #pl.plot(omega_m / constants.GHz_2pi,
    #        Gamma_m, label=r'$\Gamma_{\rm rad}(\omega)$')
    #pl.plot(omega_m / constants.GHz_2pi,
    #        Sigma_m, label=r'Re $\Sigma (\omega)$')
    p = pl.plot(omega_m / constants.GHz_2pi, Gamma_s(omega_m),
            label=r'$\Gamma_{\rm rad}(\omega)$')
    #p = pl.plot(omega_coarse / constants.GHz_2pi, Gamma_coarse, 'o',
    #            label=r'Tabulation points',
    #            markeredgecolor = p[0].get_color(),
    #            markerfacecolor = 'yellow')
    pl.plot(omega_m / constants.GHz_2pi, Sigma_s(omega_m),
            label=r'Re $\Sigma(\omega)$')
    pl.plot(omega_m / constants.GHz_2pi, Sigma_m_Gamma, '--',
            label=r'Re $\Sigma_\Gamma(\omega)$')
    pl.plot(omega_m / constants.GHz_2pi, Sigma_m_prime, '--',
            label=r'Re $\Sigma\'(\omega)$')
    pl.plot(-omega_m / constants.GHz_2pi, -Gamma_m_prime, '--',
            label=r'Im $\Sigma\'(\omega)$')
    ##pl.plot(omega_m / constants.GHz_2pi, Sigma_m_prime.real, '--',
    ##        label=r'Re $\Sigma\'(\omega)$')
    ##pl.plot(omega_m / constants.GHz_2pi, Sigma_m_prime.imag, '--',
    ##        label=r'Im $\Sigma\'(\omega)$')
    pl.plot(np.array([Omega_res, Omega_res]) / constants.GHz_2pi,
            np.array([Gamma_s(Omega_res), Sigma_s(Omega_res)]),
            'r:', linestyle='dotted')
    pl.plot(np.array([Omega_res]) / constants.GHz_2pi,
            np.array([Gamma_s(Omega_res)]), 'o',
            markeredgecolor='red', markerfacecolor='yellow')
    pl.plot(np.array([Omega_res]) / constants.GHz_2pi,
            np.array([Sigma_s(Omega_res)]), 'o',
            markeredgecolor='red', markerfacecolor='yellow')
    #pl.plot(np.array([Omega_res, Omega_res]) / constants.GHz_2pi,
    #        np.array([Gamma_s(Omega_0), Sigma_s(Omega_0)]),
    #        'k--')
    pl.plot(np.array([Omega_0]) / constants.GHz_2pi,
            np.array([Gamma_s(Omega_0)]), 'o',
            markeredgecolor='black', markerfacecolor='white')
    pl.plot(np.array([Omega_0]) / constants.GHz_2pi,
            np.array([Sigma_s(Omega_0)]), 'o',
            markeredgecolor='black', markerfacecolor='white')
    #pl.plot(omega_m / constants.GHz_2pi, 0.0, 'k--')
    xmin, xmax = pl.xlim()
    pl.plot([xmin, xmax], [0.0, 0.0], 'k--')
    pl.xlim(xmin, xmax)
    pl.xlabel(r"Frequency $\omega/2\pi$, GHz")
    pl.ylabel(r"Self-energy $\Sigma(\omega)$, ${\rm ns}^{-1}$")
    #pl.plot(np.array([Omega_res, Omega_res]) / constants.GHz_2pi, [0.0, np.max(Gamma_coarse)], 'r--')
    #pl.plot(np.array([Omega_0, Omega_0]) / constants.GHz_2pi, [0.0, np.max(Gamma_coarse)], 'k--')
    pl.legend(loc=1)
    
    pl.show()


for fname in sys.argv[1:]:
    if fname.find('+analysis') < 0:
        print ("no +analysis in the file name, skip", fname)
        continue
    if fname.find('+analysis2') > 0:
        print ("use new2 format: ", fname)
        show_analysis_new2(fname)
    elif fname.find('+analysis3') > 0:
        print ("use new3 format: ", fname)
        show_analysis_new3(fname)
    else:
        print ("use old format: ", fname)
        show_analysis_old(fname)
