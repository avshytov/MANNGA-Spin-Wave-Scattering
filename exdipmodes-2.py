import numpy as np
import pylab as pl
from scipy import linalg
from scipy import integrate

from slab import Slab, Mode
from thinslab import ThinSlab, UniformMode
from resonator import Resonator, ResonatorAnti
from constants import um, nm, kA_m, mT, pJ_m, GHz, GHz_2pi, ns


#res_V = 0.025
res_W = 200 * nm
res_L = 100 * nm
#res_W = 200 * nm
#res_L = 100 * nm
res_H = 40  * nm
#res_H = 30  * nm
res_Ms = 4.5 * 140.0 * kA_m 
res_bias = 5 * mT
res_alpha = 0.01
res_Nx = 0.01
res_Ny = 1.0 - res_Nx
#res_Nx = 0.5
#res_Ny = 0.5
res_theta = 0.0
#res_theta = np.pi / 2.0
z_res = 5 * nm

#resonator = Resonator(res_L, res_W, res_H,
#                      res_Ms, res_bias, res_alpha, res_Nx, res_Ny,
#                      res_theta)

resonator = ResonatorAnti(res_L, res_W, res_H,
                      res_Ms, res_bias, res_alpha, res_Nx, res_Ny,
                      res_theta)

print ("resonant freq: ", resonator.omega_0() / GHz_2pi,
       "gamma_0 = ", resonator.gamma_0())
#Jex = 0.0005
#Jex = 0.0001
d = 50 * nm
Bext = 5 * mT
Ms = 140 * kA_m
N = 10
alpha = 0.001
Aex = 3.5 * pJ_m
Jex = Aex / Ms**2

slab = Slab(-d, 0.0, Bext, Ms, Jex, alpha, N)
slab_t = ThinSlab(-d, 0.0, Bext, Ms, Jex, alpha)

from dispersion import Dispersion
from scattering import ScatteringProblem
        
scattering_problem = ScatteringProblem(slab, resonator, z_res)
scattering_problem_t = ScatteringProblem(slab_t, resonator, z_res)

def show_scat_wave(omega, theta_inc, kx, ky):
    print ("solve for: o = ", omega, "theta = ", theta_inc)
    inc_data = scattering_problem.find_data_theta(omega, theta_inc)
    Delta_inc, k_inc, u_inc, v_inc, alpha_inc, alpha_p = inc_data
    print ("*** incident: k = ", k_inc)
    print ("Coupling: ", Delta_inc)
    KY, KX = np.meshgrid(ky, kx)
    G = 0.0 * KX + 0.0j
    Delta = 0.0 * KX + 0.0j
    C = 0.0 * G
    #K_O = 0.0 * KX
    #U_O = 0.0 * KX
    #A_O = 0.0 * KX
    #A_P = 0.0 * KX
    for i in range(len(kx)):
        for j in range(len(ky)):
            G[i, j]     = scattering_problem.G(omega, kx[i], ky[j])
            print ("get: k = ", kx[i], ky[j])
            Delta[i, j] = scattering_problem.Delta(kx[i], ky[j])
            #Delta_k, k, u_o, alpha_o, alpha_p = find_data_k(kx[i], ky[j])
            #Delta[i, j] = Delta_k
            #K_O[i, j] = k
            #U_O[i, j] = u_o
            #A_O[i, j] = alpha_o
            #A_P[i, j] = alpha_p
            
    f_res = scattering_problem.resonant_factor(omega)
    print ("resonant freq: ", resonator.omega_0() / GHz_2pi)
    psi_scat = G * Delta * Delta_inc.conj() * f_res
    np.savez("exdip-psi-f=%gGHz.npz" % (omega / GHz_2pi),
             kx=kx, ky=ky, psi_scat=psi_scat, G=G,
             theta_inc=theta_inc, omega=omega,
             f_res = f_res, 
             **scattering_problem.describe())
    if  False:
        from matplotlib.colors import LogNorm
        pl.figure()
        pl.pcolormesh(KX, KY, np.abs(G)**2, cmap='magma',
                      norm=LogNorm(vmin=1.0, vmax=1e+6))
        pl.title(r"Propagator for $f = %g$GHz" % omega / GHz_2pi)
        pl.gca().set_aspect('equal', 'box')
        pl.colorbar()
        pl.figure()
        pl.pcolormesh(KX, KY, np.abs(psi_scat)**2,
                      cmap='magma',  norm=LogNorm(vmin=1.0, vmax=1e+4))
        pl.title(r"Scattered wave for $f = %g$GHz, $\theta = %g^\circ$"
                 % (omega / GHz_2pi, theta_inc / np.pi * 180.0))
        pl.gca().set_aspect('equal', 'box')
        pl.colorbar()
        pl.show()
                  
#kx_vals = np.linspace(-2.0, 2.0, 200)
#ky_vals = np.linspace(-8.0, 8.0, 800)
#kx_vals = np.linspace(-25.0, 25.0, 250)
#ky_vals = np.linspace(-80, 80, 800)
kx_vals = np.linspace(-50.0, 50.0, 500)
ky_vals = np.linspace(-100, 100, 1000)
theta_inc = 0.0
#for f in [2.82, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.#2, 3.3, 3.4, 3.5]:
#for f in [2.8, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.85, 2.90, 2.95, 3.0]:
for f in [2.8, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.75, 2.8, 2.85,
          2.90, 3.0, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]:
#for f in [3.2, 3.5, 4.0]:
    omega = GHz_2pi * f
    show_scat_wave(omega, theta_inc, kx_vals, ky_vals)
#for omega in [0.16, 0.11, 0.12, 0.13, 0.14, 0.15, 0.17, 0.18, 0.19, 0.20]:
#for omega in [12.0, 8.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0, 16.0]:
#    show_scat_wave(omega, theta_inc, kx_vals, ky_vals)
#for omega in [0.13, 0.14]:

def show_scat_quantities(omega, theta_tab):

    k_theta = []
    u_theta = []
    v_theta = []
    Delta_theta = []
    alpha_theta = []
    alpha_prime = []
    Gamma_rad = scattering_problem.Gamma_rad(omega)
    print ("radiative linewidth: ", Gamma_rad)
    for theta in theta_tab:
        k_data = scattering_problem.find_data_theta(omega, theta)
        Delta, k, u_o, v_o, alpha_o, alpha_p = k_data
        k_theta.append(k)
        Delta_theta.append(Delta)
        u_theta.append(u_o)
        v_theta.append(v_o)
        alpha_prime.append(alpha_p)
        if (len(alpha_theta) > 0):
            while alpha_o > alpha_theta[-1] + np.pi:
                alpha_o -= 2.0 * np.pi
            while alpha_o < alpha_theta[-1] - np.pi:
                alpha_o += 2.0 * np.pi
        alpha_theta.append(alpha_o)
        print ("theta = ", theta, "alpha = ", alpha)
        
    u_theta = np.array(u_theta)
    v_theta = np.array(v_theta)
    Delta_theta = np.array(Delta_theta)
    k_theta = np.array(k_theta)
    alpha_theta = np.array(alpha_theta)
    alpha_prime = np.array(alpha_prime)

    print ("k_theta = ", k_theta)
    print ("u_theta = ", u_theta)
    print ("v_theta = ", v_theta)

    np.savez("exdip-scat-data-f=%gGHz" % (omega / GHz_2pi),
             omega=omega,
             k_theta = k_theta,
             u_theta = u_theta,
             v_theta = v_theta,
             Delta_theta = Delta_theta,
             theta=theta_tab,
             alpha_theta = alpha_theta,
             alpha_prime = alpha_prime,
             Gamma_rad = Gamma_rad, 
             **scattering_problem.describe())

    if False:
        pl.figure()
        pl.plot(theta_tab, alpha_theta)
        pl.xlabel(r"$\theta$")
        pl.ylabel(r"$\alpha(\theta)$")
        pl.figure()
        pl.polar(theta_tab, np.abs(u_theta))
        pl.title("Group velocity")
        pl.figure()
        pl.polar(theta_tab, k_theta)
        pl.title("wavenumber")
        pl.figure()
        pl.polar(theta_tab, np.abs(Delta_theta))
        pl.title("Coupling")
        pl.show()

#show_scat_quantities(2.8 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(1.5 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(2.0 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(2.5 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(3.0 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(3.5 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))

def test_scat_quantities(omega, theta_tab):

    k_theta = []
    k_theta_t = []
    u_theta = []
    u_theta_t = []
    v_theta = []
    v_theta_t = []
    Delta_theta = []
    Delta_theta_t = []
    alpha_theta = []
    alpha_theta_t = []
    alpha_prime = []
    alpha_prime_t = []
    Gamma_rad = scattering_problem.Gamma_rad(omega)
    Gamma_rad_t = scattering_problem_t.Gamma_rad(omega)
    print ("radiative linewidth: ", Gamma_rad, Gamma_rad_t)
    for theta in theta_tab:
        k_data = scattering_problem.find_data_theta(omega, theta)
        Delta, k, u_o, v_o, alpha_o, alpha_p = k_data
        k_theta.append(k)
        Delta_theta.append(Delta)
        u_theta.append(u_o)
        v_theta.append(v_o)
        alpha_prime.append(alpha_p)
        if (len(alpha_theta) > 0):
            while alpha_o > alpha_theta[-1] + np.pi:
                alpha_o -= 2.0 * np.pi
            while alpha_o < alpha_theta[-1] - np.pi:
                alpha_o += 2.0 * np.pi
        alpha_theta.append(alpha_o)
        print ("theta = ", theta, "alpha = ", alpha)
        k_data_t = scattering_problem_t.find_data_theta(omega, theta)
        Delta_t, k_t, u_o_t, v_o_t, alpha_o_t, alpha_p_t = k_data_t
        k_theta_t.append(k_t)
        Delta_theta_t.append(Delta_t)
        u_theta_t.append(u_o_t)
        v_theta_t.append(v_o_t)
        alpha_prime_t.append(alpha_p_t)
        if (len(alpha_theta_t) > 0):
            while alpha_o_t > alpha_theta_t[-1] + np.pi:
                alpha_o_t -= 2.0 * np.pi
            while alpha_o_t < alpha_theta_t[-1] - np.pi:
                alpha_o_t += 2.0 * np.pi
        alpha_theta_t.append(alpha_o_t)
        
    u_theta = np.array(u_theta)
    v_theta = np.array(v_theta)
    Delta_theta = np.array(Delta_theta)
    k_theta = np.array(k_theta)
    alpha_theta = np.array(alpha_theta)
    alpha_prime = np.array(alpha_prime)
    
    u_theta_t = np.array(u_theta_t)
    v_theta_t = np.array(v_theta_t)
    Delta_theta_t = np.array(Delta_theta_t)
    k_theta_t = np.array(k_theta_t)
    alpha_theta_t = np.array(alpha_theta_t)
    alpha_prime_t = np.array(alpha_prime_t)

    print ("k_theta = ", k_theta)
    print ("u_theta = ", u_theta)
    print ("v_theta = ", v_theta)

    np.savez("exdip-scat-data-test-f=%gGHz" % (omega / GHz_2pi),
             omega=omega,
             k_theta = k_theta,
             u_theta = u_theta,
             v_theta = v_theta,
             Delta_theta = Delta_theta,
             theta=theta_tab,
             alpha_theta = alpha_theta,
             alpha_prime = alpha_prime,
             Gamma_rad = Gamma_rad, 
             Gamma_rad_t = Gamma_rad_t, 
             k_theta_t = k_theta_t,
             u_theta_t = u_theta_t,
             v_theta_t = v_theta_t,
             Delta_theta_t = Delta_theta_t,
             alpha_theta_t = alpha_theta_t,
             alpha_prime_t = alpha_prime_t,
             **scattering_problem.describe())

    if False:
        pl.figure()
        pl.polar(theta_tab, k_theta)
        pl.polar(theta_tab, k_theta_t, '--')
        pl.title("k_theta")
        pl.figure()
        pl.plot(theta_tab, alpha_theta)
        pl.plot(theta_tab, alpha_theta_t, '--')
        pl.xlabel(r"$\theta$")
        pl.ylabel(r"$\alpha(\theta)$")
        pl.figure()
        pl.polar(theta_tab, np.abs(u_theta))
        pl.polar(theta_tab, np.abs(u_theta_t), '--')
        pl.title("Group velocity (radial)")
        pl.figure()
        pl.polar(theta_tab, np.abs(v_theta))
        pl.polar(theta_tab, np.abs(v_theta_t), '--')
        pl.title("Group velocity (full)")
        pl.figure()
        pl.polar(theta_tab, k_theta)
        pl.polar(theta_tab, k_theta_t, '--')
        pl.title("wavenumber")
        pl.figure()
        pl.polar(theta_tab, np.abs(Delta_theta))
        pl.polar(theta_tab, np.abs(Delta_theta_t), '--')
        pl.title("Coupling")
        pl.figure()
        pl.polar(theta_tab, np.abs(Delta_theta))
        pl.title("Coupling: full")
        pl.figure()
        pl.polar(theta_tab, np.abs(Delta_theta_t))
        pl.title("Coupling: approx")
        pl.figure()
        pl.plot(theta_tab, np.abs(Delta_theta))
        pl.plot(theta_tab, np.abs(Delta_theta_t), '--')
        pl.title("Coupling")
        pl.show()

#test_scat_quantities(1.5 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(2.0 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(2.5 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(2.8 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(2.82 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(3.0 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(3.5 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))

#show_scat_quantities(2.82 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(2.2 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(0.16, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(12.0, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(12.0, np.linspace(0.0, 2.0 * np.pi, 1001))
    
def show_scat_ampl(theta_inc, omega_tab, theta_tab):
    result  = []
    gamma_rad_o = []
    scat_tot_o  = []
    scat_abs_o  = []
    for theta in theta_tab:
        result.append([])
        
    for omega in omega_tab:
        print ("solve for: o = ", omega / GHz_2pi, "theta = ", theta_inc)
        inc_data = scattering_problem.find_data_theta(omega, theta_inc)
        Delta_inc, k_inc, u_inc, v_inc, alpha_inc, alpha_p_inc = inc_data
        kx_inc = k_inc * np.cos(theta_inc)
        ky_inc = k_inc * np.sin(theta_inc)
        print ("*** incident: k = ", kx_inc, ky_inc)
        Gamma_rad = scattering_problem.Gamma_rad (omega)
        gamma_rad_o.append(Gamma_rad)
        print ("resonant freq: ", resonator.omega_0())
        print ("exact Gamma_rad(omega) = ", Gamma_rad)
        f_res = scattering_problem.resonant_factor(omega)
        
        Deltas = []
        for i_theta, theta in enumerate(theta_tab):
            print("f = ", omega / GHz_2pi, "theta = ", theta / np.pi * 180.0)
            k_data = scattering_problem.find_data_theta(omega, theta)
            Delta_scat, k_scat, u_scat, v_scat, alpha_scat, alpha_p_sc = k_data
            kx_scat = k_scat * np.cos(theta)
            ky_scat = k_scat * np.sin(theta)
            f_scat = Delta_inc.conj() * Delta_scat * f_res
            f_scat *= np.sqrt(abs(k_scat / u_scat / alpha_p_sc))
            f_scat *= np.sqrt( 1.0 / (2.0 * np.pi))
            f_scat *= np.sqrt(1.0 / v_inc)
            result[i_theta].append(f_scat)
        sigma_tot = np.abs(Delta_inc)**2 * 2.0 / v_inc * np.abs(f_res)**2 * Gamma_rad
        sigma_abs = sigma_tot * resonator.gamma_0() / Gamma_rad
        scat_tot_o.append(sigma_tot)
        scat_abs_o.append(sigma_abs)
        np.savez("exdip-scat-tmp.npz",
            omega_tab   = np.array(omega_tab),
            theta_tab   = np.array(theta_tab),
            theta_inc   = np.array(theta_inc),
            result      = np.array(result),
            gamma_rad_o = np.array(gamma_rad_o),
            scat_tot_o = np.array(scat_tot_o),
            scat_abs_o = np.array(scat_abs_o),
            **scattering_problem.describe())
            
    np.savez("exdip-scat.npz",
             omega_tab   = np.array(omega_tab),
             theta_tab   = np.array(theta_tab),
             theta_inc   = np.array(theta_inc),
             result      = np.array(result),
             gamma_rad_o = np.array(gamma_rad_o),
            scat_tot_o   = np.array(scat_tot_o),
            scat_abs_o   = np.array(scat_abs_o),
             **scattering_problem.describe())
    
    pl.figure()
    for i_theta, theta in enumerate(theta_tab):
        pl.plot(omega_tab, np.abs(np.array(result[i_theta]))**2,
                label=r'$\theta=%g^\circ$' % (theta / np.pi * 180.0))
    pl.legend()
    pl.show()
    
theta_tab = [0.0, np.pi/6.0, np.pi/3.0, np.pi/2.0, 2.0 * np.pi/3.0,
             5.0 * np.pi/6.0, np.pi]

#omega_tab = np.array([0.16, 0.105, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20])
#omega_tab = np.array([12.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
#omega_tab = np.linspace(8.0, 13.0, 101)
#omega_tab = np.array([2.8, 2.3, 2.4, 2.5, 2.6, 2.7, 2.9, 3.0, 3.1]) * GHz_2pi
#omega_tab = np.array([2.82, 2.5, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
#                      2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45,
#                      2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95,
#                      3.0, 3.05, 3.1, 3.15, 3.2, 3.25, 3.3, 3.35, 3.4, 3.45,
#                      3.5]) * GHz_2pi
omega_tab = np.linspace(1.5, 3.2, 17*5 + 1) * GHz_2pi
#omega_tab = np.array([2.82, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75,
#                      2.8, 2.85, 2.9, 2.95, 3.0]) * GHz_2pi
#omega_tab = np.linspace(8.0, 13.0, 501)
#omega_tab = np.linspace(0.105, 0.2, 96)
#omega_tab = np.linspace(0.105, 0.2, 476)
#show_scat_ampl(theta_inc, omega_tab, theta_tab)

            
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
    omega_y = np.vectorize(lambda k: get_omega(0.0, k))(ky_vals)
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
    pl.plot(ky_vals, omega_y.real / GHz_2pi, label='k||y')
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
    pl.plot(ky_vals, -omega_y.imag * ns, label='gamma: k||y')
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

#show_modes()
