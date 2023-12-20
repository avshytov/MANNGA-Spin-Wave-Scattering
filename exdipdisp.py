import numpy as np
import pylab as pl
from scipy import linalg
from scipy import integrate

from slab import Slab, Mode
from thinslab import ThinSlab, UniformMode
from constants import um, nm, kA_m, mT, pJ_m, GHz, GHz_2pi, ns


d = 85 * nm
Bext = 3 * mT
Ms = 120 * kA_m
N = 50
alpha = 0.001
Aex = 2 * 3.5 * pJ_m 
Jex = Aex / Ms**2

slab = Slab(-d, 0.0, Bext, Ms, Jex, alpha, N)
slab_t = ThinSlab(-d, 0.0, Bext, Ms, Jex, alpha)

from dispersion import Dispersion
from scattering import ScatteringProblem
        
            
def get_omega(kx, ky):
    #omega, z, mx, mz, phi
    mode_plus, mode_minus = slab.make_modes(kx, ky)
    mode, E = mode_plus
    z = mode.z
    omega = mode.omega
    if  False:
        pl.figure()
        pl.plot(z, phi.real, label='Re phi')
        pl.plot(z, phi.imag, label='Im phi')
        pl.plot(z, mx.real, label='Re mx')
        pl.plot(z, mx.imag, label='Im mx')
        pl.plot(z, mz.real, label='Re mz')
        pl.plot(z, mz.imag, label='Im mz')
        pl.legend()
        pl.show()
    return omega

def get_omega_n(kx, ky, n):
    #omega, z, mx, mz, phi
    modes_plus, modes_minus = slab.make_modes(kx, ky, n + 1)
    mode, E = modes_plus[n]
    z = mode.z
    omega = mode.omega
    if  False:
        pl.figure()
        pl.plot(z, phi.real, label='Re phi')
        pl.plot(z, phi.imag, label='Im phi')
        pl.plot(z, mx.real, label='Re mx')
        pl.plot(z, mx.imag, label='Im mx')
        pl.plot(z, mz.real, label='Re mz')
        pl.plot(z, mz.imag, label='Im mz')
        pl.legend()
        pl.show()
    return omega

def get_omega_t(kx, ky):
    #omega, z, mx, mz, phi
    mode_plus, mode_minus = slab_t.make_modes(kx, ky)
    mode, E = mode_plus
    omega = mode.omega
    return omega

def show_dispersion():
    #kx = np.linspace(-5.0, 5.0, 500)
    #ky = np.linspace(-5.0, 5.0, 500)
    kx_vals = ky_vals = np.linspace(-100.0, 100.0, 2000)
    #kx_vals = ky_vals = np.linspace(-50.0, 50.0, 50)
    #kx = ky = np.linspace(0.01, 10.0, 100)

    omega_x = np.vectorize(lambda k: get_omega(k,   0.0))(kx_vals)
    omega_x1 = np.vectorize(lambda k: get_omega_n(k,   0.0, 1))(kx_vals)
    omega_x2 = np.vectorize(lambda k: get_omega_n(k,   0.0, 2))(kx_vals)
    omega_x3 = np.vectorize(lambda k: get_omega_n(k,   0.0, 3))(kx_vals)
    omega_y = np.vectorize(lambda k: get_omega(0.0, k))(ky_vals)
    omega_y1 = np.vectorize(lambda k: get_omega_n(0.0, k, 1))(ky_vals)
    omega_y2 = np.vectorize(lambda k: get_omega_n(0.0, k, 2))(ky_vals)
    omega_y3 = np.vectorize(lambda k: get_omega_n(0.0, k, 3))(ky_vals)
    omega_x_t = np.vectorize(lambda k: get_omega_t(k,   0.0))(kx_vals)
    omega_y_t = np.vectorize(lambda k: get_omega_t(0.0, k))(ky_vals)

    from constants import gamma_s, mu_0
    omega_b = gamma_s * (Bext + Jex *  Ms * kx_vals**2)
    omega_M = gamma_s * mu_0 * Ms 
    ex  = np.exp(-np.abs(kx_vals)*d)
    ex2 = np.exp(-np.abs(kx_vals)*2.0*d)
    omega_x_naive = np.sqrt(omega_b**2 + omega_M * omega_b
                            + 0.25 * omega_M**2 * (1.0 - ex2))
    Nx = 1.0 -  (1.0 - ex) / np.abs(kx_vals) / d 
    #Nx = np.clip(np.abs(kx_vals) * d / 2.0, 0.0, 1.0)
    Ny = 1.0 - Nx
    omega_x_n = np.sqrt((omega_b + Nx * omega_M) * (omega_b + Ny * omega_M))
    gamma_x_n = alpha / 2.0 * (omega_b + Nx * omega_M + omega_b + Ny * omega_M)

    omega_b = gamma_s * (Bext + Jex * Ms * ky_vals**2)
    omega_M = gamma_s * mu_0 * Ms
    ex = np.exp(-np.abs(ky_vals)*d)
    omega_Mx = omega_M * (1.0 - ex) / np.abs(ky_vals) / d
    omega_y_naive = np.sqrt(omega_b * (omega_b + omega_Mx))
    Nx = 0.0
    Ny =  (1.0 - ex) / np.abs(ky_vals) / d 
    #Ny = np.clip(1.0 - np.abs(ky_vals) * d / 2.0, 0.0, 1.0)
    omega_y_n = np.sqrt((omega_b + Nx * omega_M) * (omega_b + Ny * omega_M))
    gamma_y_n = alpha / 2.0 * (omega_b + Nx * omega_M + omega_b + Ny * omega_M)



    pl.figure()
    pl.plot(kx_vals, omega_x.real / GHz_2pi, label='k||x')
    pl.plot(kx_vals, omega_x1.real / GHz_2pi, label='k||x')
    pl.plot(kx_vals, omega_x2.real / GHz_2pi, label='k||x')
    pl.plot(kx_vals, omega_x3.real / GHz_2pi, label='k||x')
    pl.plot(ky_vals, omega_y.real / GHz_2pi, label='k||y')
    pl.plot(ky_vals, omega_y1.real / GHz_2pi, label='k||y')
    pl.plot(ky_vals, omega_y2.real / GHz_2pi, label='k||y')
    pl.plot(ky_vals, omega_y3.real / GHz_2pi, label='k||y')
    pl.plot(kx_vals, omega_x_naive / GHz_2pi, '--', label='semi-exact: k||x')
    pl.plot(ky_vals, omega_y_naive / GHz_2pi, '--', label='semi-exact: k||y')
    pl.plot(kx_vals, omega_x_n / GHz_2pi, '--', label='demag approx: k||x')
    pl.plot(ky_vals, omega_y_n / GHz_2pi, '--', label='demag-approx: k||y')
    pl.plot(kx_vals, omega_x_t.real / GHz_2pi, label='k||x')
    pl.plot(ky_vals, omega_y_t.real / GHz_2pi, label='k||y')
    pl.xlabel(r"Wavenumber $k_x$, $k_y$")
    pl.ylabel(r"Frequency $\omega/2\pi$, GHz")
    pl.legend()

    pl.figure()
    pl.plot(kx_vals, -omega_x.imag * ns, label='gamma: k||x')
    pl.plot(kx_vals, -omega_x1.imag * ns, label='gamma: k||x')
    pl.plot(ky_vals, -omega_y.imag * ns, label='gamma: k||y')
    pl.plot(ky_vals, -omega_y1.imag * ns, label='gamma: k||y')
    pl.plot(kx_vals, gamma_x_n * ns, '--', label='gamma_n: k||x')
    pl.plot(ky_vals, gamma_y_n * ns, '--', label='gamma_n: k||y')
    pl.plot(kx_vals, -omega_x_t.imag * ns, label='gamma: k||x')
    pl.plot(ky_vals, -omega_y_t.imag * ns, label='gamma: k||y')
    pl.xlabel(r"Wavenumber $k_x$, $k_y$")
    pl.ylabel(r"Decay rate $\Gamma$, $\rm{ns}^{-1}$")
    pl.legend()
    np.savez("exdip-dispersion.npz",
             kx_vals = kx_vals, ky_vals = ky_vals,
             omega_x = omega_x, omega_y = omega_y,
             omega_x_naive = omega_x_naive, omega_x_n = omega_x_n,
             omega_y_naive = omega_y_naive, omega_y_n = omega_y_n,
             gamma_x_n = gamma_x_n, gamma_y_n = gamma_y_n)
    pl.show()

show_dispersion()
    
def show_modes():
    kvals = [-50.0, -10.0, -3.0, -1.0, -0.5, -0.1, -0.01,
             0.01, 0.1, 0.5, 1.0, 3.0, 10.0, 50.0]

    pl.figure()
    ax_x_mx = pl.gca()
    pl.title("mx, k || x")
    pl.figure()
    ax_x_mz = pl.gca()
    pl.title("mz, k || x")
    pl.figure()
    ax_x_phi = pl.gca()
    pl.title("phi, k || x")

    pl.figure()
    ax_norm = pl.gca()
    pl.title("normalisation")

    omega_px_vals = []
    omega_mx_vals = []
    E_px_vals = []
    E_mx_vals = []

    for k in kvals:
        kx = k
        ky = 0.0
        mode_plus, mode_minus = slab.make_modes(kx, ky)
        mode_p, E_p = mode_plus
        mode_m, E_m = mode_minus
        omega_p = mode_p.omega
        mx_p = mode_p.mx
        mz_p = mode_p.mz
        phi_p = mode_p.phi
        z_p = mode_p.z
        omega_m = mode_m.omega
        mx_m = mode_m.mx
        mz_m = mode_m.mz
        phi_m = mode_m.phi
        z_m = mode_m.z
        k_lab = 'k = (%g, %g)' % (kx, ky)
        o_re = integrate.trapz((mx_p.conj() * mz_m - mz_p.conj() * mx_m).real, z_p)
        o_im = integrate.trapz((mx_p.conj() * mz_m - mz_p.conj() * mx_m).imag, z_p)
        print ("<+|-> = ", o_re + 1j * o_im)
        p = ax_x_mx.plot(z_p, mx_p.real,  label=k_lab)
        ax_x_mx.plot(z_m, mx_m.real, '--', color=p[0].get_color())
        p = ax_x_mz.plot(z_p, mz_p.imag,  label=k_lab)
        ax_x_mz.plot(z_m, mz_m.imag, '--', color=p[0].get_color())
        p = ax_x_phi.plot(z_p, phi_p.imag,  label=k_lab)
        ax_x_phi.plot(z_m, phi_m.imag, '--', color=p[0].get_color())
        omega_px_vals.append(omega_p)
        omega_mx_vals.append(omega_m)
        E_px_vals.append(E_p)
        E_mx_vals.append(E_m)


    omega_px_vals = np.array(omega_px_vals)
    omega_mx_vals = np.array(omega_mx_vals)
    E_px_vals = np.array(E_px_vals)
    E_mx_vals = np.array(E_mx_vals)

    ax_x_mx.legend()
    ax_x_mz.legend()
    ax_x_phi.legend()

    ax_norm.plot(kvals, E_px_vals / omega_px_vals, label='+, ||x')
    ax_norm.plot(kvals, -E_mx_vals / omega_mx_vals, label='-, ||x')


    #pl.figure()
    #pl.plot(kx_vals, omega_x)
    #pl.plot(kvals,  omega_px_vals, 'o', label='+')
    #pl.plot(kvals, -omega_mx_vals, 'o', label='-')
    #pl.title("k || x")
    #pl.legend()


    pl.figure()
    ax_y_mx = pl.gca()
    pl.title("mx, k || y")
    pl.figure()
    ax_y_mz = pl.gca()
    pl.title("mz, k || y")
    pl.figure()
    ax_y_phi = pl.gca()
    pl.title("phi, k || y")

    omega_py_vals = []
    omega_my_vals = []
    E_py_vals = []
    E_my_vals = []

    for k in kvals:
        kx = 0
        ky = k
        mode_plus, mode_minus = slab.make_modes(kx, ky)
        mode_p, E_p = mode_plus
        mode_m, E_m = mode_minus
        omega_p = mode_p.omega
        mx_p = mode_p.mx
        mz_p = mode_p.mz
        phi_p = mode_p.phi
        z_p = mode_p.z
        omega_m = mode_m.omega
        mx_m = mode_m.mx
        mz_m = mode_m.mz
        phi_m = mode_m.phi
        z_m = mode_m.z
        o_re = integrate.trapz((mx_p.conj() * mz_m - mz_p.conj() * mx_m).real, z_p)
        o_im = integrate.trapz((mx_p.conj() * mz_m - mz_p.conj() * mx_m).imag, z_p)
        print ("<+|-> = ", o_re + 1j * o_im)
        k_lab = 'k = (%g, %g)' % (kx, ky)
        p = ax_y_mx.plot(z_p, mx_p.real,  label=k_lab)
        ax_y_mx.plot(z_m, mx_m.real, '--', color=p[0].get_color())
        p = ax_y_mz.plot(z_p, mz_p.imag,  label=k_lab)
        ax_y_mz.plot(z_m, mz_m.imag, '--', color=p[0].get_color())
        p = ax_y_phi.plot(z_p, phi_p.imag,  label=k_lab)
        ax_y_phi.plot(z_m, phi_m.imag, '--', color=p[0].get_color())
        omega_py_vals.append(omega_p)
        omega_my_vals.append(omega_m)
        E_py_vals.append(E_p)
        E_my_vals.append(E_m)


    omega_py_vals = np.array(omega_py_vals)
    omega_my_vals = np.array(omega_my_vals)
    E_py_vals = np.array(E_py_vals)
    E_my_vals = np.array(E_my_vals)

    ax_norm.plot(kvals, E_py_vals / omega_py_vals, label='+, ||y')
    ax_norm.plot(kvals, -E_my_vals / omega_my_vals, label='-, ||y')
    ax_norm.legend()

    ax_y_mx.legend()
    ax_y_mz.legend()
    ax_y_phi.legend()

    #pl.figure()
    #pl.plot(ky_vals, omega_y)
    #pl.plot(kvals,  omega_py_vals, 'o', label='+')
    #pl.plot(kvals, -omega_my_vals, 'o', label='-')
    #pl.title("k || y")
    #pl.legend()
    
    pl.show()

show_modes()
