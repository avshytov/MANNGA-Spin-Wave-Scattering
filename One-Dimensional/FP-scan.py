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


def solve_M(Ms):

    #res_V = 0.025
    res_W = 10000 * nm
    #res_L = 150 * nm
    res_L = 850 * nm
    #res_L = 50 * nm
    #res_W = 200 * nm
    #res_L = 100 * nm
    #res_H = 30  * nm
    res_H = 30  * nm
    #res_H = 30  * nm
    #res_Ms = -140 * kA_m
    # 1080?
    #res_Ms = -1080 * kA_m #-1080 * kA_m
    res_Ms = Ms
    #75 * kA_m #-1080 * kA_m
    #res_Ms = 1150 * kA_m #-1080 * kA_m 
    res_bias = 3 * mT
    res_alpha = 0.008 # 0.001
    #res_Nx = 0.1
    #min(0.5, np.pi / 2.0 * res_H / res_L)
    #res_Ny = 1.0 - res_Nx
    #res_Nx = 0.5
    #res_Ny = 0.5
    #res_Nx = 75
    #res_Nx = 50
    #res_Nx = 40
    res_Nx = 101 # 171
    #res_Nx = 25
    #res_Nz = 15
    #res_Nz = 10
    res_Nz = 4 # 7


    res_theta = 0.0
    #res_theta = np.pi / 2.0
    z_res = 5 * nm
    # 2 * 13 ?
    res_Aex = 16 * pJ_m * 2 #13 * pJ_m * 2
    #res_Aex = 13 * pJ_m * 2 #13 * pJ_m * 2
    res_Jex = res_Aex / res_Ms**2
    f_min = 0.001 * GHz
    #f_max = 5.0 * GHz
    #f_max = 25.0 * GHz
    f_max = 40.0 * GHz
    #f_max = 500.0 * GHz

    resonator = Resonator2D(res_L, res_W, res_H,
                            res_Ms, res_Jex, res_bias, res_alpha,
                            res_Nx, res_Nz,
                            f_min, f_max,
                            res_theta)
    #Jex = 0.0005
    #Jex = 0.0001
    d = 85 * nm
    Bext = res_bias #5 * mT
    slab_Ms = 120 * kA_m
    #N = 30
    N = 10 #17
    slab_alpha = 0.0005
    slab_Aex = 3.5 * pJ_m * 2
    slab_Jex = slab_Aex / slab_Ms**2
    #res_Nsx = 100
    #res_Nsz = 30
    #res_Nsx = 250
    #res_Nsz = 30

    res_geometry = GeometrySpec(res_L, res_W, res_H,
                                res_Nx, res_Nz)
    #slab_geometry = GeometrySpec(res_L + 3 * res_H + 3 * z_res, res_W, d,
    #                             res_Nsx, res_Nsz)
    slab_material = MaterialSpec("YIG-slab", slab_Ms, slab_Jex, slab_alpha)
    res_material = MaterialSpec("CoFeB-res", res_Ms, res_Jex, res_alpha)

    slab = Slab(-d, 0.0, Bext, slab_Ms, slab_Jex, slab_alpha, N)
    #slab_t = ThinSlab(-d, 0.0, Bext, Ms, Jex, alpha)

    from dispersion import Dispersion
    from scattering import ScatteringProblem_1D, ScatteringProblem_1D_Multi

    scattering_problem_1d_multi = ScatteringProblem_1D_Multi(slab,
                                                    resonator.modes, z_res, 3)

    #omega_tab = np.linspace(2.0, 3.0, 50 + 1) * GHz_2pi
    #omega_tab = np.linspace(1.0, 4.5, 350 + 1) * GHz_2pi
    #omega_tab = np.linspace(1.0, 4.5, 3500 + 1) * GHz_2pi
    #omega_tab = np.linspace(1.0, 4.5, 3500 + 1) * GHz_2pi
    #omega_tab = np.linspace(0.9, 2.0, 550 + 1) * GHz_2pi
    #omega_tab = np.linspace(0.7,  5.0, 2150 + 1) * GHz_2pi
    omega_tab = np.linspace(0.65,  3.7, 1525 + 1) * GHz_2pi
    #omega_tab = np.linspace(0.7,  10.0, 4650 + 1) * GHz_2pi
    #omega_tab = np.linspace(1.0, 2.5, 1500 + 1) * GHz_2pi
    #omega_tab = np.linspace(1.0, 1.5, 500 + 1) * GHz_2pi
    #omega_tab = np.linspace(1.4, 4.5, 3100 + 1) * GHz_2pi
    #omega_tab = np.linspace(1.3, 4.5, 3200 + 1) * GHz_2pi
    #solve_1d(omega_tab)

    #k_max = 200 * rad_um
    # ORIG: 150
    k_max = 100 * rad_um
    Nk = 750
    result_bare = scattering_problem_1d_multi.T_and_R_really_all_bare(omega_tab,
                                                                  k_max, Nk,
                                                                  True)
    T1, R1, T2, R2, phi_R, phibar_R, phi_L, phibar_L = result_bare
    A1 = 1.0 - np.abs(T1)**2 - np.abs(R1)**2
    A2 = 1.0 - np.abs(T2)**2 - np.abs(R2)**2

    np.savez("exdip-scat-1d-Ms=%g-L=%gnm-s=%gnm-B=%gmT-nrf.npz" % (
                                                             res_Ms / kA_m,
                                                             res_L / nm,
                                                             z_res / nm,
                                                             Bext  / mT),
                 omega = omega_tab,
                 T1_bare = np.array(T1), R1_bare = np.array(R1),
                 A1_bare = np.array(A1),
                 T2_bare = np.array(T2), R2_bare = np.array(R2),
                 A2_bare = np.array(A2),
                 phi_R_bare  = np.array(phi_R),
                 phibar_R_bare = np.array(phibar_R), 
                 phi_L_bare  = np.array(phi_L),
                 res_Ms = res_Ms,
                 res_Jex = res_Jex,
                 res_alpha = res_alpha,
                 res_H = res_H,
                 res_L = res_L, 
                 phibar_L_bare = np.array(phibar_L),
                 **scattering_problem_1d_multi.describe())
    #omega_tab = np.array([1.8]) * GHz_2pi
    #omega_tab = np.linspace(3.0, 3.3, 3 + 1) * GHz_2pi 

#Ms_vals = np.linspace(20 * kA_m, 1100 * kA_m, 217)[1::5][41:]
#Ms_vals = np.linspace(-20 * kA_m, -1100 * kA_m, 217)[-5::-5]
#Ms_vals = np.linspace(200 * kA_m, 500 * kA_m, 301)[4::5]
Ms_vals = np.linspace(-1 * kA_m, -10 * kA_m, 451)[4::5]
#Ms_vals =  np.array([-700, -750]) * kA_m
#Ms_vals =  np.array([-700, -750]) * kA_m
for Ms in Ms_vals:
    solve_M(Ms)
