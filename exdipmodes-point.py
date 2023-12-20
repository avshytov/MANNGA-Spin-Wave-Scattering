import numpy as np
import pylab as pl
from scipy import linalg
from scipy import integrate

from slab import Slab, Mode
from thinslab import ThinSlab, UniformMode
from resonator import Resonator, ResonatorAnti
from constants import um, nm, kA_m, mT, pJ_m, GHz, GHz_2pi, ns, mJ_m2
from constants import gamma_s
#from modes2 import Area, Grid, Material, CellArrayModel
#from nearfield import solveNearField
#from resonator2d import make_resonant_mode_collection
#from resonator2d import make_mode_collection


#res_V = 0.025
res_W = 10 * nm
res_L = 10 * nm
#res_W = 200 * nm
#res_L = 100 * nm
res_H = 1  * nm
res_Ms = 140 * kA_m
alpha_res = 0.01
res_Nx = 0.01
res_Nz = 0.99

Bext = 5 * mT
res_bias = Bext
res_theta = np.pi
#res_theta = np.pi / 2.0
s = 5 * nm
res_alpha = 1e-3

d_slab = 50 * nm

Ms_YIG = 140 * kA_m
Aex_YIG = 2 * 3.5 * pJ_m
Jex_YIG = Aex_YIG / Ms_YIG**2
alpha_YIG = 1e-3


resonator = Resonator(res_L, res_W, res_H,
                      res_Ms, res_bias, res_alpha, res_Nx, res_Nz,
                      res_theta)
       
slab   = Slab(-s - d_slab, -s, Bext, Ms_YIG, Jex_YIG, alpha_YIG, 10)
slab_t = ThinSlab(-s - d_slab, -s, Bext, Ms_YIG, Jex_YIG, alpha_YIG)


     

#resonator = Resonator(res_L, res_W, res_H,
#                      res_Ms, res_bias, res_alpha, res_Nx, res_Ny,
#                      res_theta)

#resonator = ResonatorAnti(res_L, res_W, res_H,
#                      res_Ms, res_bias, res_alpha, res_Nx, res_Ny,
#                      res_theta)

print ("resonant freq: ", resonator.omega_0() / GHz_2pi,
       "gamma_0 = ", resonator.gamma_0())
#Jex = 0.0005
#Jex = 0.0001
#d = 60 * nm
#Bext = 3 * mT
#Ms = 140 * kA_m
#N = 50
#alpha = 0.001
#Aex = 2 * 3.5 * pJ_m * 2
#Jex = Aex / Ms**2

#d = 20 * nm

#slab = Slab(-d, 0.0, Bext, Ms, Jex, alpha, N)
#slab_t = ThinSlab(-d, 0.0, Bext, Ms, Jex, alpha)

from dispersion import Dispersion
from scattering import ScatteringProblem
        
scattering_problem = ScatteringProblem(slab, resonator, 0.0)
scattering_problem_t = ScatteringProblem(slab_t, resonator, 0.0)

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
    np.savez("exdip-psi-f=%gGHz-s=%gnm-d=%gnm-f0=%gGHz.npz" % (omega / GHz_2pi,
                                                        s / nm, d_slab / nm,
                                                resonator.omega_0() / GHz_2pi),
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
#for f in [2.8, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.75, 2.8, 2.85,
#          2.90, 3.0, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]:
#for f in [3.2, 3.5, 4.0]:
#for f in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]:
#for f in [1.0, 1.5, 2.0, 2.5, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 4.0, 4.5]:
#    omega = GHz_2pi * f
#    show_scat_wave(omega, theta_inc, kx_vals, ky_vals)
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
        print ("theta = ", theta, "alpha = ", alpha_o)
        
    u_theta = np.array(u_theta)
    v_theta = np.array(v_theta)
    Delta_theta = np.array(Delta_theta)
    k_theta = np.array(k_theta)
    alpha_theta = np.array(alpha_theta)
    alpha_prime = np.array(alpha_prime)

    print ("k_theta = ", k_theta)
    print ("u_theta = ", u_theta)
    print ("v_theta = ", v_theta)

    np.savez("exdip-scat-data-f=%gGHz-s=%gnm-d=%gnm-f0=%gGHz" % (omega / GHz_2pi,
                                                         s / nm, d_slab / nm, 
                                              resonator.omega_0() / GHz_2pi),
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

#show_scat_quantities(1.0 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(1.2 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(1.4 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(1.6 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(1.8 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(2.0 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(2.2 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(2.4 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(2.6 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(3.0 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(3.5 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(4.0 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))

#show_scat_quantities(1.0 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(1.5 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(2.0 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(2.5 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(2.8 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(3.0 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(3.2 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(3.6 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(3.8 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(4.0 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(4.5 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#show_scat_quantities(5.0 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))


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
        print ("theta = ", theta, "alpha = ", alpha_o)
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

    res_or = '%gdeg' % (resonator.theta_or / np.pi * 180.0)
    if np.abs(resonator.theta_or) < 0.1:
      res_or = '+x'
    elif np.abs(resonator.theta_or - np.pi/2.0) < 0.1:
      res_or='+y'
    elif np.abs(resonator.theta_or - np.pi) < 0.1:
      res_or='-x'
    elif np.abs(resonator.theta_or + np.pi/2.0) < 0.1:
      res_or='-y'
    np.savez("exdip-scat-data-test-f=%gGHz-s=%gnm-d=%gnm-f0=%gGHz-or=%s" % (
      omega / GHz_2pi, s / nm, d_slab / nm,
      resonator.omega_0() / GHz_2pi, res_or),
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

test_scat_quantities(1.0 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
test_scat_quantities(1.5 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
test_scat_quantities(2.0 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
test_scat_quantities(2.5 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
test_scat_quantities(3.0 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
test_scat_quantities(3.5 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
test_scat_quantities(4.0 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
test_scat_quantities(4.5 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))

#test_scat_quantities(1.0 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(1.5 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(1.8 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(2.0 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(2.2 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(2.5 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(2.6 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(2.7 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(2.8 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(2.9 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(3.0 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(3.1 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(3.2 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(3.3 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(3.4 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(3.5 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(3.6 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(3.8 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(4.0 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))
#test_scat_quantities(4.5 * GHz_2pi, np.linspace(0.0, 2.0 * np.pi, 1001))



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
        np.savez("exdip-scat-tmp-s=%gnm-d=%gnm-f0=%gGHz.npz" % (s / nm,
                                                                d_slab / nm,
                                        resonator.omega_0() / GHz_2pi),
            omega_tab   = np.array(omega_tab),
            theta_tab   = np.array(theta_tab),
            theta_inc   = np.array(theta_inc),
            result      = np.array(result),
            gamma_rad_o = np.array(gamma_rad_o),
            scat_tot_o = np.array(scat_tot_o),
            scat_abs_o = np.array(scat_abs_o),
            **scattering_problem.describe())
            
    np.savez("exdip-scat-s=%gnm-d=%gnm-f0=%gGHz.npz" % (s / nm, d_slab / nm,
                                      resonator.omega_0() / GHz_2pi),
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
#omega_tab = np.linspace(1.5, 3.2, 17*5 + 1) * GHz_2pi
omega_tab = np.linspace(1.0, 5.0, 40 * 10 + 1) * GHz_2pi
#omega_tab = np.array([2.82, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75,
#                      2.8, 2.85, 2.9, 2.95, 3.0]) * GHz_2pi
#omega_tab = np.linspace(8.0, 13.0, 501)
#omega_tab = np.linspace(0.105, 0.2, 96)
#omega_tab = np.linspace(0.105, 0.2, 476)
#show_scat_ampl(theta_inc, omega_tab, theta_tab)

            
