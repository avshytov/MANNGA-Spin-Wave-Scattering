import numpy as np
import pylab as pl
from scipy import linalg
from slab import Slab, Mode
from thinslab import ThinSlab, UniformMode
#from resonator import Resonator, ResonatorAnti                                
from constants import um, nm, kA_m, mT, pJ_m, GHz, GHz_2pi, ns
from resonator2d import Resonator2D
from resonator2d import Resonator2DNearField
from dispersion import Dispersion                                              
from scattering import ScatteringProblem_1D, ScatteringProblem_1D_Multi
import constants

#res_V = 0.025
res_W = 1000 * nm
res_L = 200 * nm
#res_W = 200 * nm
#res_L = 100 * nm
res_H = 30  * nm
#res_H = 30  * nm
res_Ms = 140.0 * kA_m 
res_bias = 5 * mT
res_alpha = 0.001
#res_Nx = 0.1
#min(0.5, np.pi / 2.0 * res_H / res_L)
#res_Ny = 1.0 - res_Nx
#res_Nx = 0.5
#res_Ny = 0.5
res_Nx = 50
res_Nz = 5


res_theta = 0.0
#res_theta = np.pi / 2.0
z_res = 10 * nm
res_Aex = 3.5 * pJ_m * 2
res_Jex = res_Aex / res_Ms**2
f_min = 1.0 * GHz
f_max = 10.0 * GHz

d = 20 * nm
Bext = 5 * mT
Ms = 140 * kA_m
N = 10
alpha = 0.001
Aex = 3.5 * pJ_m * 2
Jex = Aex / Ms**2
res_Nsx = 100
res_Nsz = 5

resonator_nrf = Resonator2DNearField(res_L, res_W, res_H,
                        res_Ms, res_Jex, res_bias, res_alpha,
                        res_Nx, res_Nz,
                        z_res, d, res_Nsx, res_Nsz, 
                        f_min, f_max,
                        res_theta)

slab = Slab(-d, 0.0, Bext, Ms, Jex, alpha, N)

scattering_problem_1d_nrf = ScatteringProblem_1D_Multi(slab,
                                       resonator_nrf.modes, z_res)

o_vals = np.linspace(1.0, 4.2, 1281) * GHz_2pi
Nmodes = len(resonator_nrf.modes)
Gamma_ab = np.zeros((Nmodes, Nmodes, len(o_vals)), dtype=complex)
for i_o, o in enumerate(o_vals):
    Gamma_ab[:, :, i_o] = scattering_problem_1d_nrf.Gamma_rad_ab(o)
#k_vals = np.linspace(-100 * constants.rad_um, 100 * constants.rad_um, 10)
#Sigma_ab = scattering_problem_1d_nrf.Sigma_1D(o_vals, 100 * constants.rad_um,
#                                              100)

f_vals = o_vals / GHz_2pi

if False:
    for a in range(Nmodes):
        for b in range(Nmodes):
            pl.figure()
            pl.title("a, b = %d %d" % (a, b))
            pl.plot(f_vals, Sigma_ab[a, b, :].real, label='Re Re Sigma')
            pl.plot(f_vals, Gamma_ab[a, b, :].real, label='Re Gamma_rad')
            pl.plot(f_vals, Sigma_ab[a, b, :].imag, label='Im Re Sigma')
            pl.plot(f_vals, Gamma_ab[a, b, :].imag, label='Im Gamma_rad')
            pl.legend()

if False:
   M = np.zeros((Nmodes, Nmodes, len(o_vals)), dtype=complex)
   for i in range(len(o_vals)):
       M[:, :, i] += o_vals[i] * np.eye(Nmodes)
       M[:, :, i] += - scattering_problem_1d_nrf.omegas_0()
       M[:, :, i] += 1j * scattering_problem_1d_nrf.gammas_0() 
       M[:, :, i] += 1j * Gamma_ab[:, :, i]
       M[:, :, i] += -Sigma_ab[:, :, i] 

   Det = np.zeros((len(o_vals)), dtype=complex)
   for i in range(len(o_vals)):
       Det[i] = linalg.det(M[:, :, i])
   pl.figure()
   pl.title("det")
   pl.plot(f_vals, np.abs(Det), label='|Det|')
   pl.plot(f_vals, Det.real, label='Re det')
   pl.plot(f_vals, Det.imag, label='Im det')
   pl.legend()

k_max = 150 * constants.rad_um
Nk = 500
T1, R1, T2, R2 = scattering_problem_1d_nrf.T_and_R_full(o_vals, k_max, Nk)
pl.figure()
T1_old = np.zeros((len(o_vals)), dtype=complex)
T2_old = np.zeros((len(o_vals)), dtype=complex)
R1_old = np.zeros((len(o_vals)), dtype=complex)
R2_old = np.zeros((len(o_vals)), dtype=complex)
for i, o in enumerate(o_vals):
  T1_old[i], R1_old[i], T2_old[i], R2_old[i] = scattering_problem_1d_nrf.T_and_R(o)
  
pl.figure()
p1 = pl.plot(f_vals, np.abs(T1), label='T1')
p2 = pl.plot(f_vals, np.abs(T2), label='T2')
p3 = pl.plot(f_vals, np.abs(R1), label='R1')
p4 = pl.plot(f_vals, np.abs(R2), label='R2')
pl.plot (f_vals, np.abs(T1_old), '--', color=p1[0].get_color())
pl.plot (f_vals, np.abs(T2_old), '--', color=p2[0].get_color())
pl.plot (f_vals, np.abs(R1_old), '--', color=p3[0].get_color())
pl.plot (f_vals, np.abs(R2_old), '--', color=p4[0].get_color())
pl.legend()
pl.show()
