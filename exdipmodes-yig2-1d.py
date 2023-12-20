import numpy as np
import pylab as pl
from scipy import linalg
from scipy import integrate

from slab import Slab, Mode
from thinslab import ThinSlab, UniformMode
#from resonator import Resonator, ResonatorAnti
from constants import um, nm, kA_m, mT, pJ_m, GHz, GHz_2pi, ns
from resonator2d import Resonator2D


#res_V = 0.025
res_W = 1000 * nm
res_L = 100 * nm
#res_W = 200 * nm
#res_L = 100 * nm
res_H = 20  * nm
#res_H = 30  * nm
res_Ms = 140.0 * kA_m 
res_bias = 3 * mT
res_alpha = 0.001
#res_Nx = 0.1
#min(0.5, np.pi / 2.0 * res_H / res_L)
#res_Ny = 1.0 - res_Nx
#res_Nx = 0.5
#res_Ny = 0.5
res_Nx = 100
res_Nz = 5
res_theta = 0.0
#res_theta = np.pi / 2.0
z_res = 20 * nm
res_Aex = 3.5 * pJ_m
res_Jex = res_Aex / res_Ms**2
f_min = 1.0 * GHz
f_max = 5.0 * GHz

resonator = Resonator2D(res_L, res_W, res_H,
                        res_Ms, res_Jex, res_bias, res_alpha,
                        res_Nx, res_Nz,
                        f_min, f_max,
                        res_theta)

#resonator = ResonatorAnti(res_L, res_W, res_H,
#                      res_Ms, res_bias, res_alpha, res_Nx, res_Ny,
#                      res_theta)

print ("resonant freq: ", resonator.omega_0() / GHz_2pi,
       "gamma_0 = ", resonator.gamma_0())
#Jex = 0.0005
#Jex = 0.0001
d = 60 * nm
Bext = 3 * mT
Ms = 140 * kA_m
N = 10
alpha = 0.001
Aex = 3.5 * pJ_m
Jex = Aex / Ms**2

slab = Slab(-d, 0.0, Bext, Ms, Jex, alpha, N)
slab_t = ThinSlab(-d, 0.0, Bext, Ms, Jex, alpha)

from dispersion import Dispersion
from scattering import ScatteringProblem_1D, ScatteringProblem_1D_Multi
        
scattering_problem_1d_multi = ScatteringProblem_1D_Multi(slab,
                                       resonator.modes, z_res)

def solve_1d(omega_tab):
    T1_o = []
    R1_o = []
    T2_o = []
    R2_o = []
    A1_o = []
    A2_o = []
    Gamma_rad_o = []
    for omega in omega_tab:
        Gamma_rad = scattering_problem_1d_multi.Gamma_rad_ab(omega)
        Gamma_rad_o.append(Gamma_rad)
        T1, R1, T2, R2 = scattering_problem_1d_multi.T_and_R(omega)
        T1_o.append(T1)
        R1_o.append(R1)
        A1 = 1.0 - np.abs(T1)**2 - np.abs(R1)**2
        A1_o.append(A1)
        T2_o.append(T2)
        R2_o.append(R2)
        A2 = 1.0 - np.abs(T2)**2 - np.abs(R2)**2
        A2_o.append(A2)
        print ("f = ", omega / GHz_2pi,
               "T, R, A = ", np.abs(T1), np.abs(R1), A1)
        print ("   Gammas: ", np.diag(Gamma_rad))
    np.savez("exdip-scat-1d.npz", omega = omega_tab,
             T1 = np.array(T1_o), R1 = np.array(R1_o), A1 = np.array(A1_o),
             T2 = np.array(T2_o), R2 = np.array(R2_o), A2 = np.array(A2_o),
             Gamma_rad = np.array(Gamma_rad_o), 
             **scattering_problem_1d_multi.describe())    

#omega_tab = np.linspace(2.0, 3.0, 50 + 1) * GHz_2pi
#omega_tab = np.linspace(1.0, 4.5, 350 + 1) * GHz_2pi
omega_tab = np.linspace(1.0, 4.5, 100 + 1) * GHz_2pi
solve_1d(omega_tab)
