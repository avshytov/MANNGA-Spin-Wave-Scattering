import numpy as np
import pylab as pl
import constants
from modes2 import Material, CellArrayModel, Grid, Area
from nearfield import solveNearField

W = 120 * constants.nm
t1 = 20 * constants.nm
t2 = 15 * constants.nm

Bext =   5 * constants.mT

Ms1 = -800 * constants.kA_m
Aex1 = 13  * constants.pJ_m * 2
Jex1 = Aex1 / Ms1**2
alpha1 = 5e-3

Ms2 = 1150 * constants.kA_m
       #0.55 * 1e-6 * 1e-7 / 1e-2  # 0.55 x 10^-6 erg / cm 
       # https://ieeexplore.ieee.org/document/1065345
Aex2 = 16 * constants.pJ_m * 2
Jex2 = Aex2 / Ms2**2
alpha2 = 5e-3

Ms_YIG = 140 * constants.kA_m
Aex_YIG = 7.0 * constants.pJ_m
Jex_YIG = Aex_YIG / Ms_YIG**2
alpha_YIG = 1e-3

RKKY_CoFeB_Py = -1.5 * constants.mJ_m2 
J_RKKY = RKKY_CoFeB_Py / Ms1 / Ms2 

d_slab = 50 * constants.nm
s = 30 * constants.nm

# https://www.sciencedirect.com/science/article/abs/pii/S0304885320323027
Py = Material("Py", Ms1, Jex1, alpha1, constants.gamma_s)
CoFeB = Material("CoFeB", Ms2, Jex2, alpha2, constants.gamma_s)
YIG = Material("YIG", Ms_YIG, Jex_YIG, alpha_YIG, constants.gamma_s)
py_area = Area("Py-area", Grid(-W/2, W/2, 0, t1, 60, 5), Py)
cofeb_area = Area("CoFeB-area", Grid(-W/2, W/2, t1, t1 + t2, 60, 4), CoFeB)
L = W + 5 * s
slab_area = Area("YIG-area", Grid(-L/2, L/2, -s - d_slab, -s, 200, 15), YIG)

bare_model = CellArrayModel()
bare_model.add_area(py_area, Bext)
bare_model.add_area(cofeb_area, Bext)
bare_model.connect(py_area.north(), cofeb_area.south(), J_RKKY,
                   Ms1, Ms2)
result = bare_model.solve()
X, Z = result['coords_all']

f_min = 0.5 * constants.GHz
f_max = 10  * constants.GHz
for mode in result['modes']:
    print ("freq: ", mode.f )
    if mode.f < f_min or mode.f > f_max: continue
    mx_all, mz_all = mode.m_all()
    i_max = np.argmax(np.abs(mx_all))
    C_phase = mx_all[i_max] / np.abs(mx_all[i_max])
    mx_all /= C_phase
    mz_all /= C_phase
    pl.figure()
    pl.tripcolor(X, Z, np.abs(mx_all), cmap='magma')
    cb = pl.colorbar()
    pl.gca().set_aspect('equal', 'box')
    pl.tripcolor(X, Z + 1.5 * (t1 + t2), np.abs(mz_all), cmap='magma')
    mx_max = np.max(np.abs(mx_all))
    mz_max = np.max(np.abs(mz_all))
    pl.xlabel(r"Position $x$, $\mu$m")
    pl.ylabel(r"Position $y$, $\mu$m")
    cb.set_label(r"$|m_{x, z}|(x, y)$")
    pl.text(-W/4, t1/2.0, r"$|m_x|$")
    pl.text(-W/4,  1.5 * (t1 + t2) + 0.5 * t2, r"$|m_z|$")
    
    pl.figure()
    pl.tripcolor(X, Z, mx_all.real, cmap='bwr', vmin = -mx_max, vmax=mx_max)
    pl.tripcolor(X, Z + 1.5 * (t1 + t2), mz_all.real, cmap='bwr',
                 vmin=-mx_max, vmax=mx_max)
    pl.gca().set_aspect('equal', 'box')
    cb = pl.colorbar()
    pl.text(-W/4, t1/2.0, r"$m_x$")
    pl.text(-W/4,  1.5 * (t1 + t2) + 0.5 * t2, r"$m_z$")
    pl.xlabel(r"Position $x$, $\mu$m")
    pl.ylabel(r"Position $y$, $\mu$m")
    cb.set_label(r"$m_{x, z}(x, y)$")

    pl.figure()
    pl.tripcolor(X, Z, mx_all.imag, cmap='bwr', vmin = -mx_max, vmax=mx_max)
    pl.tripcolor(X, Z + 1.5 * (t1 + t2), mz_all.imag, cmap='bwr',
                 vmin=-mx_max, vmax=mx_max)
    pl.gca().set_aspect('equal', 'box')
    cb = pl.colorbar()
    pl.text(-W/4, t1/2.0, r"$m_x$")
    pl.text(-W/4,  1.5 * (t1 + t2) + 0.5 * t2, r"$m_z$")
    pl.xlabel(r"Position $x$, $\mu$m")
    pl.ylabel(r"Position $y$, $\mu$m")
    cb.set_label(r"$m_{x, z}(x, y)$")

    pl.figure()
    pl.tripcolor(X, Z, np.angle(mx_all) * 180.0/np.pi,
                 cmap='hsv', vmin=-180.0, vmax=180.0)
    pl.tripcolor(X, Z + 1.5 * (t1 + t2),
                 np.angle(mz_all) * 180.0/np.pi,
                 cmap='hsv', vmin=-180.0, vmax=180.0)
    pl.gca().set_aspect('equal', 'box')
    cb = pl.colorbar()
    pl.text(-W/4, t1/2.0, r"arg $m_x$")
    pl.text(-W/4,  1.5 * (t1 + t2) + 0.5 * t2, r"arg $m_z$")
    pl.xlabel(r"Position $x$, $\mu$m")
    pl.ylabel(r"Position $y$, $\mu$m")
    cb.set_label(r"arg $m_{x, z}(x, y)$")
    
    pl.figure()
    pl.quiver(X[::5], Z[::5], mx_all[::5].real, mz_all[::5].real, color='red')
    pl.quiver(X[::5], Z[::5], mx_all[::5].imag, mz_all[::5].imag, color='blue')
    pl.gca().set_aspect('equal', 'box')
    pl.xlabel(r"Position $x$, $\mu$m")
    pl.ylabel(r"Position $y$, $\mu$m")
    
    pl.show()


    
nrf_model = CellArrayModel()
nrf_model.add_area(py_area, Bext)
nrf_model.add_area(cofeb_area, Bext)
nrf_model.add_area(slab_area, Bext)
nrf_model.connect(py_area.north(), cofeb_area.south(), J_RKKY)

Wr = 200 * constants.nm
Hr = t1 + t2
Lr = W
nrf_modes = solveNearField(nrf_model, f_min, f_max, Lr, Wr, Hr, 0.0,  
                        "CoFeB-area", "Py-area")

for mode in nrf_modes:
    print ("NRF : ", mode.omega / constants.GHz_2pi)
    pl.figure()
    f = mode.omega / constants.GHz_2pi
    pl.title("f = %g + i %g" % ( f.real, f.imag))
    pl.tripcolor(mode.X, mode.Z, np.abs(mode.mx), cmap='magma')
    pl.tripcolor(mode.X, mode.Z + 1.5 * (t1 + t2),
                np.abs(mode.mz), cmap='magma')
    pl.gca().set_aspect('equal', 'box')
    cb = pl.colorbar()

    mx_max = np.max(np.abs(mode.mx))
    mz_max = np.max(np.abs(mode.mz))
    pl.figure()
    pl.tripcolor(mode.X, mode.Z, mode.mx.real,
                 cmap='bwr', vmin = -mx_max, vmax=mx_max)
    pl.tripcolor(mode.X, mode.Z + 1.5 * (t1 + t2), mode.mz.real, cmap='bwr',
                 vmin=-mz_max, vmax=mz_max)
    pl.gca().set_aspect('equal', 'box')
    cb = pl.colorbar()
    pl.text(-W/4, t1/2.0, r"$m_x$")
    pl.text(-W/4,  1.5 * (t1 + t2) + 0.5 * t2, r"$m_z$")
    pl.xlabel(r"Position $x$, $\mu$m")
    pl.ylabel(r"Position $y$, $\mu$m")
    cb.set_label(r"$m_{x, z}(x, y)$")

    pl.figure()
    pl.tripcolor(mode.X, mode.Z, mode.mx.imag,
                 cmap='bwr', vmin = -mx_max, vmax=mx_max)
    pl.tripcolor(mode.X, mode.Z + 1.5 * (t1 + t2), mode.mz.imag, cmap='bwr',
                 vmin=-mz_max, vmax=mz_max)
    pl.gca().set_aspect('equal', 'box')
    cb = pl.colorbar()
    pl.text(-W/4, t1/2.0, r"$m_x$")
    pl.text(-W/4,  1.5 * (t1 + t2) + 0.5 * t2, r"$m_z$")
    pl.xlabel(r"Position $x$, $\mu$m")
    pl.ylabel(r"Position $y$, $\mu$m")
    cb.set_label(r"$m_{x, z}(x, y)$")

    pl.figure()
    pl.tripcolor(mode.X, mode.Z, np.angle(mode.mx) * 180.0/np.pi,
                 cmap='hsv', vmin=-180.0, vmax=180.0)
    pl.tripcolor(mode.X, mode.Z + 1.5 * (t1 + t2),
                 np.angle(mode.mz) * 180.0/np.pi,
                 cmap='hsv', vmin=-180.0, vmax=180.0)
    pl.gca().set_aspect('equal', 'box')
    cb = pl.colorbar()
    pl.text(-W/4, t1/2.0, r"arg $m_x$")
    pl.text(-W/4,  1.5 * (t1 + t2) + 0.5 * t2, r"arg $m_z$")
    pl.xlabel(r"Position $x$, $\mu$m")
    pl.ylabel(r"Position $y$, $\mu$m")
    cb.set_label(r"arg $m_{x, z}(x, y)$")
    
    pl.figure()
    pl.quiver(mode.X[::5], mode.Z[::5],
              mode.mx[::5].real, mode.mz[::5].real, color='red')
    pl.quiver(mode.X[::5], mode.Z[::5], mode.mx[::5].imag, mode.mz[::5].imag,
              color='blue')
    pl.gca().set_aspect('equal', 'box')
    pl.xlabel(r"Position $x$, $\mu$m")
    pl.ylabel(r"Position $y$, $\mu$m")

    pl.show()
