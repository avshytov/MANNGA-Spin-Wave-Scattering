import numpy as np
import pylab as pl
from scipy import linalg
from scipy import integrate

from slab import Slab, Mode
from thinslab import ThinSlab, UniformMode
#from resonator import Resonator, ResonatorAnti
from constants import um, nm, kA_m, mT, pJ_m, GHz, GHz_2pi, ns, rad_um
from resonator2d import Resonator2D
from resonator2d import Resonator2DNearField, GeometrySpec, MaterialSpec


#res_V = 0.025
res_W = 10000 * nm
#res_L = 150 * nm
res_L = 250 * nm
#res_L = 50 * nm
#res_W = 200 * nm
#res_L = 100 * nm
#res_H = 30  * nm
res_H = 30  * nm
#res_H = 30  * nm
#res_Ms = -140 * kA_m 
res_Ms = -140 * kA_m 
res_bias = 3 * mT
res_alpha = 0.001
#res_Nx = 0.1
#min(0.5, np.pi / 2.0 * res_H / res_L)
#res_Ny = 1.0 - res_Nx
#res_Nx = 0.5
#res_Ny = 0.5
#res_Nx = 75
#res_Nx = 50
#res_Nx = 40
res_Nx = 85
#res_Nx = 25
#res_Nz = 15
#res_Nz = 10
res_Nz = 10


res_theta = 0.0
#res_theta = np.pi / 2.0
z_res = 20 * nm
res_Aex = 3.5 * pJ_m * 2
res_Jex = res_Aex / res_Ms**2
f_min = 0.001 * GHz
#f_max = 5.0 * GHz
f_max = 15.0 * GHz
#f_max = 500.0 * GHz

resonator = Resonator2D(res_L, res_W, res_H,
                        res_Ms, res_Jex, res_bias, res_alpha,
                        res_Nx, res_Nz,
                        f_min, f_max,
                        res_theta)
res_modes_mx   = []
res_modes_mz   = []
res_modes_freq = []
res_modes_mx_nrf   = []
res_modes_mz_nrf   = []
res_modes_freq_nrf = []
res_mode_X_nrf = 0.0
res_mode_Z_nrf = 0.0
for mode in resonator.modes:
    res_mode_X = mode.X
    res_mode_Z = mode.Z
    res_modes_mx.append(mode.mx)
    res_modes_mz.append(mode.mz)
    res_modes_freq.append(mode.omega)


    
#resonator = ResonatorAnti(res_L, res_W, res_H,
#                      res_Ms, res_bias, res_alpha, res_Nx, res_Ny,
#                      res_theta)

print ("resonant freq: ", resonator.omega_0() / GHz_2pi,
       "gamma_0 = ", resonator.gamma_0())
#Jex = 0.0005
#Jex = 0.0001
d = 60 * nm
Bext = res_bias #5 * mT
Ms = 140 * kA_m
#N = 30
N = 20
alpha = 0.001
Aex = 3.5 * pJ_m * 2
Jex = Aex / Ms**2
#res_Nsx = 100
#res_Nsz = 30
res_Nsx = 150
res_Nsz = 20

slab = Slab(-d, 0.0, Bext, Ms, Jex, alpha, N)
k0 = 1e-3
mode0_E_plus, mode0_E_minus = slab.make_modes(k0, 0.0, 1)
mode0_plus, E_plus = mode0_E_plus
mode0_minus, E_minus = mode0_E_minus
print ("min freq: ", mode0_plus.omega / GHz_2pi)

slab_t = ThinSlab(-d, 0.0, Bext, Ms, Jex, alpha)


res_geometry = GeometrySpec(res_L, res_W, res_H,
                            res_Nx, res_Nz)
slab_geometry = GeometrySpec(res_L + 3 * res_H + 3 * z_res, res_W, d,
                             res_Nsx, res_Nsz)
slab_material = MaterialSpec("YIG-slab", Ms, Jex, alpha)
res_material = MaterialSpec("YIG-res", res_Ms, res_Jex, res_alpha)
resonator_nrf = Resonator2DNearField( res_geometry, slab_geometry,
                                      res_material, slab_material, 
                                      Bext, z_res,  
                                      f_min, f_max, res_theta)
#resonator_nrf = Resonator2DNearField(res_L, res_W, res_H,
#                        res_Ms, res_Jex, res_bias, res_alpha,
#                        res_Nx, res_Nz,
#                        z_res, d, res_Nsx, res_Nsz, 
#                        f_min, f_max,
#                        res_theta)

for mode in resonator_nrf.modes:
    res_mode_X_nrf = mode.X
    res_mode_Z_nrf = mode.Z
    res_modes_mx_nrf.append(mode.mx)
    res_modes_mz_nrf.append(mode.mz)
    res_modes_freq_nrf.append(mode.omega)


from dispersion import Dispersion
from scattering import ScatteringProblem_1D, ScatteringProblem_1D_Multi
        
scattering_problem_1d_multi = ScatteringProblem_1D_Multi(slab,
                                                resonator.modes, z_res, 2)
scattering_problem_1d_nrf = ScatteringProblem_1D_Multi(slab,
                                            resonator_nrf.modes, z_res, 2)

def solve_1d(omega_tab):
    T1_o = []; T1_o_nrf = []
    R1_o = []; R1_o_nrf = []
    T2_o = []; T2_o_nrf = []
    R2_o = []; R2_o_nrf = []
    A1_o = []; A1_o_nrf = []
    A2_o = []; A2_o_nrf = []
    Gamma_rad_o = []; Gamma_rad_o_nrf = []
    Gamma_R_o = []; Gamma_R_o_nrf = []
    Gamma_L_o = []; Gamma_L_o_nrf = []
    for omega in omega_tab:
        Gamma_rad = scattering_problem_1d_multi.Gamma_rad_ab(omega)
        Gamma_rad_nrf = scattering_problem_1d_nrf.Gamma_rad_ab(omega)
        Gamma_rad_o.append(Gamma_rad)
        Gamma_rad_o_nrf.append(Gamma_rad_nrf)
        Gamma_R = scattering_problem_1d_multi.Gamma_R_ab(omega)
        Gamma_R_nrf = scattering_problem_1d_nrf.Gamma_R_ab(omega)
        Gamma_R_o.append(Gamma_R)
        Gamma_R_o_nrf.append(Gamma_R_nrf)
        Gamma_L = scattering_problem_1d_multi.Gamma_L_ab(omega)
        Gamma_L_o.append(Gamma_L)
        Gamma_L_nrf = scattering_problem_1d_nrf.Gamma_L_ab(omega)
        Gamma_L_o_nrf.append(Gamma_L_nrf)
        T1, R1, T2, R2 = scattering_problem_1d_multi.T_and_R(omega)
        T1_o.append(T1)
        R1_o.append(R1)
        A1 = 1.0 - np.abs(T1)**2 - np.abs(R1)**2
        A1_o.append(A1)
        T2_o.append(T2)
        R2_o.append(R2)
        A2 = 1.0 - np.abs(T2)**2 - np.abs(R2)**2
        A2_o.append(A2)
        T1_nrf, R1_nrf, T2_nrf, R2_nrf = scattering_problem_1d_nrf.T_and_R(omega)
        T1_o_nrf.append(T1_nrf)
        R1_o_nrf.append(R1_nrf)
        A1_nrf = 1.0 - np.abs(T1_nrf)**2 - np.abs(R1_nrf)**2
        A1_o_nrf.append(A1_nrf)
        T2_o_nrf.append(T2_nrf)
        R2_o_nrf.append(R2_nrf)
        A2_nrf = 1.0 - np.abs(T2_nrf)**2 - np.abs(R2_nrf)**2
        A2_o_nrf.append(A2_nrf)
        print ("f = ", omega / GHz_2pi,
               "T, R, A = ", np.abs(T1), np.abs(R1), A1)
        print ("   Gammas: ", np.diag(Gamma_rad))
        print ("   nrf : ",
               "T, R, A = ", np.abs(T1_nrf), np.abs(R1_nrf), A1_nrf)
        print ("   Gammas: ", np.diag(Gamma_rad_nrf))
        np.savez("exdip-scat-1d-s=%gnm-nrf2.npz" % (z_res / nm),
             omega = omega_tab,
             T1 = np.array(T1_o), R1 = np.array(R1_o), A1 = np.array(A1_o),
             T2 = np.array(T2_o), R2 = np.array(R2_o), A2 = np.array(A2_o),
             T1_nrf = np.array(T1_o_nrf), R1_nrf = np.array(R1_o_nrf),
             A1_nrf = np.array(A1_o_nrf),
             T2_nrf = np.array(T2_o_nrf), R2_nrf = np.array(R2_o_nrf),
             A2_nrf = np.array(A2_o_nrf),
             Gamma_rad = np.array(Gamma_rad_o), 
             Gamma_R = np.array(Gamma_R_o), 
             Gamma_L = np.array(Gamma_L_o),
             Gamma_rad_nrf = np.array(Gamma_rad_o_nrf), 
             Gamma_R_nrf = np.array(Gamma_R_o_nrf), 
             Gamma_L_nrf = np.array(Gamma_L_o_nrf),
             omega_0 = np.diag(resonator.omega_0()),
             Gamma_0 = np.diag(resonator.gamma_0()),
             omega_0_nrf = np.diag(resonator_nrf.omega_0()),
             Gamma_0_nrf = np.diag(resonator_nrf.gamma_0()),
             res_modes_freq = np.array(res_modes_freq),
             res_mode_X = res_mode_X,
             res_mode_Z = res_mode_Z,
             res_modes_mx = np.array(res_modes_mx),
             res_modes_mz = np.array(res_modes_mz),
             res_modes_freq_nrf = np.array(res_modes_freq_nrf),
             res_mode_X_nrf = res_mode_X_nrf,
             res_mode_Z_nrf = res_mode_Z_nrf,
             res_modes_mx_nrf = np.array(res_modes_mx_nrf),
             res_modes_mz_nrf = np.array(res_modes_mz_nrf),
             **scattering_problem_1d_multi.describe())    

#omega_tab = np.linspace(2.0, 3.0, 50 + 1) * GHz_2pi
#omega_tab = np.linspace(1.0, 4.5, 350 + 1) * GHz_2pi
#omega_tab = np.linspace(1.0, 4.5, 3500 + 1) * GHz_2pi
#omega_tab = np.linspace(1.0, 4.5, 3500 + 1) * GHz_2pi
#omega_tab = np.linspace(1.0, 4.5, 700 + 1) * GHz_2pi
omega_tab = np.linspace(0.7, 3.0, 1150 + 1) * GHz_2pi
#omega_tab = np.linspace(1.0, 2.5, 1500 + 1) * GHz_2pi
#omega_tab = np.linspace(1.0, 1.5, 500 + 1) * GHz_2pi
#omega_tab = np.linspace(1.4, 4.5, 3100 + 1) * GHz_2pi
#omega_tab = np.linspace(1.3, 4.5, 3200 + 1) * GHz_2pi
#solve_1d(omega_tab)

#k_max = 200 * rad_um
# ORIG: 150
k_max = 150 * rad_um
Nk = 500
T1, R1, T2, R2 = scattering_problem_1d_nrf.T_and_R_really_all(omega_tab,
                                                              k_max, Nk)
A1 = 1.0 - np.abs(T1)**2 - np.abs(R1)**2
A2 = 1.0 - np.abs(T2)**2 - np.abs(R2)**2

np.savez("exdip-scat-1d-L=%gnm-s=%gnm-B=%gmT-nrf.npz" % (res_L / nm,
                                                         z_res / nm,
                                                         Bext  / mT),
             omega = omega_tab,
             T1_nrf = np.array(T1), R1_nrf = np.array(R1),
             A1_nrf = np.array(A1),
             T2_nrf = np.array(T2), R2_nrf = np.array(R2),
             A2_nrf = np.array(A2),
             #T1_nrf = np.array(T1_o_nrf), R1_nrf = np.array(R1_o_nrf),
             #A1_nrf = np.array(A1_o_nrf),
             #T2_nrf = np.array(T2_o_nrf), R2_nrf = np.array(R2_o_nrf),
             #A2_nrf = np.array(A2_o_nrf),
             #Gamma_rad = np.array(Gamma_rad_o), 
             #Gamma_R = np.array(Gamma_R_o), 
             #Gamma_L = np.array(Gamma_L_o),
             #Gamma_rad_nrf = np.array(Gamma_rad_o_nrf), 
             #Gamma_R_nrf = np.array(Gamma_R_o_nrf), 
             #Gamma_L_nrf = np.array(Gamma_L_o_nrf),
             #omega_0 = np.diag(resonator.omega_0()),
             #Gamma_0 = np.diag(resonator.gamma_0()),
             #omega_0_nrf = np.diag(resonator_nrf.omega_0()),
             #Gamma_0_nrf = np.diag(resonator_nrf.gamma_0()),
             res_modes_freq = np.array(res_modes_freq),
             res_mode_X = res_mode_X,
             res_mode_Z = res_mode_Z,
             res_modes_mx = np.array(res_modes_mx),
             res_modes_mz = np.array(res_modes_mz),
             res_modes_freq_nrf = np.array(res_modes_freq_nrf),
             res_mode_X_nrf = res_mode_X_nrf,
             res_mode_Z_nrf = res_mode_Z_nrf,
             res_modes_mx_nrf = np.array(res_modes_mx_nrf),
             res_modes_mz_nrf = np.array(res_modes_mz_nrf),
             **scattering_problem_1d_multi.describe())
#omega_tab = np.array([1.8]) * GHz_2pi
#omega_tab = np.linspace(3.0, 3.3, 3 + 1) * GHz_2pi 

