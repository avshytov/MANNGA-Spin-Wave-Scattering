import numpy as np
import pylab as pl
import resonator2d
import constants


from resonator2d import Resonator2D, Resonator2DNearField

from constants import GHz_2pi, nm

res_L = 200 * nm
res_H = 30  * nm
res_W = 1000 * nm
Nx = 50
Nz = 5
res_Ms  = 140 * constants.kA_m
Aex = 2 * 3.5 * constants.pJ_m
res_Jex = Aex / res_Ms**2
Bbias = 5 * constants.mT
res_alpha = 0.001

f_min = 1.0
f_max = 5.0

res_bare = Resonator2D (res_L, res_W, res_H, res_Ms, res_Jex, Bbias, res_alpha,
                        Nx, Nz, f_min, f_max)

slab_d = 20 * nm
Nsx = 200
Nsz = 5

#for s_nm in [1000, 200, 100, 50, 20, 10]:
for s_nm in [10]:
    s = s_nm * nm
    res_nrf = Resonator2DNearField(res_L, res_W, res_H, res_Ms, res_Jex,
                                   Bbias, res_alpha, Nx, Nz,
                                   s, slab_d, Nsx, Nsz,
                                   f_min, f_max)
    for i in range(len(res_nrf.modes)):
        mode = res_nrf.modes[i]
        bare_mode = res_bare.modes[i]
        mx, mz = mode.mx, mode.mz
        mx_bare, mz_bare = bare_mode.mx, bare_mode.mz
        print ("Mode %g @ %g: freq %f vs %g, width %g vs %g" % (i, s_nm,
                                     mode.omega.real / GHz_2pi,
                                     bare_mode.omega.real / GHz_2pi,
                                    -mode.omega.imag,
                                    -bare_mode.omega.imag))
        i_o = np.argmin(np.abs(mode.X + 1j * mode.Z
                               - ( - 0.0 * res_L/2.0  + 1j * res_H/2.0)))
        i_e = np.argmin(np.abs(mode.X + 1j * mode.Z
                               - (  1.0 * res_L/2.0  + 1j * res_H/2.0)))
        i_w = np.argmin(np.abs(mode.X + 1j * mode.Z
                               - ( -1.0 * res_L/2.0  + 1j * res_H/2.0)))
        i_n = np.argmin(np.abs(mode.X + 1j * mode.Z
                               - (  0.0 * res_L/2.0  + 1j * res_H)))
        i_s = np.argmin(np.abs(mode.X + 1j * mode.Z
                               - (  0.0 * res_L/2.0  + 1j * 0 * res_H)))
        i_ne = np.argmin(np.abs(mode.X + 1j * mode.Z
                               - (  1.0 * res_L/2.0  + 1j * res_H)))
        i_nw = np.argmin(np.abs(mode.X + 1j * mode.Z
                               - ( -1.0 * res_L/2.0  + 1j * res_H)))
        i_se = np.argmin(np.abs(mode.X + 1j * mode.Z
                               - (  1.0 * res_L/2.0  + 1j * 0 * res_H)))
        i_sw = np.argmin(np.abs(mode.X + 1j * mode.Z
                               - ( -1.0 * res_L/2.0  + 1j * 0 * res_H)))

        i_bot = [t for t in range(len(mode.Z))
                 if abs(mode.Z[t]) < 0.005]
        i_top = [t for t in range(len(mode.Z))
                 if abs(mode.Z[t] - res_H) < 0.005]
        i_mid = [t for t in range(len(mode.Z))
                 if abs(mode.Z[t] - res_H/2.0) < 0.001]
        x_top = np.array([mode.X[t] for t in i_top])
        x_bot = np.array([mode.X[t] for t in i_bot])
        x_mid = np.array([mode.X[t] for t in i_mid])

        mx_top = np.array([mx[t] for t in i_top])
        mx_bot = np.array([mx[t] for t in i_bot])
        mx_mid = np.array([mx[t] for t in i_mid])
        mz_top = np.array([mz[t] for t in i_top])
        mz_bot = np.array([mz[t] for t in i_bot])
        mz_mid = np.array([mz[t] for t in i_mid])

        mx_bare_top = np.array([mx_bare[t] for t in i_top])
        mx_bare_bot = np.array([mx_bare[t] for t in i_bot])
        mx_bare_mid = np.array([mx_bare[t] for t in i_mid])
        mz_bare_top = np.array([mz_bare[t] for t in i_top])
        mz_bare_bot = np.array([mz_bare[t] for t in i_bot])
        mz_bare_mid = np.array([mz_bare[t] for t in i_mid])

        #print (x_bot, mx_bot, mx_bare_bot)
        pl.figure()
        p = pl.plot(x_bot, np.abs(mx_bot), label='|m_x| @ bot')
        pl.plot(x_bot, np.abs(mx_bare_bot), '--', color = p[0].get_color())
        p = pl.plot(x_bot, np.abs(mz_bot), label='|m_z| @ bot')
        pl.plot(x_bot, np.abs(mz_bare_bot), '--', color = p[0].get_color())
        p = pl.plot(x_top, np.abs(mx_top), label='|m_x| @ top')
        pl.plot(x_top, np.abs(mx_bare_top), '--', color = p[0].get_color())
        p = pl.plot(x_top, np.abs(mz_top), label='|m_z| @ top')
        pl.plot(x_top, np.abs(mz_bare_top), '--', color = p[0].get_color())
        pl.legend()
        
        pl.figure()
        p = pl.plot(x_bot, mx_bot.real, label='Re m_x @ bot')
        pl.plot(x_bot, mx_bare_bot.real, '--', color = p[0].get_color())
        p = pl.plot(x_bot, mz_bot.real, label='Re m_z @ bot')
        pl.plot(x_bot, mz_bot.real, '--', color = p[0].get_color())
        p = pl.plot(x_top, mx_top.real, label='Re m_x @ top')
        pl.plot(x_top, mx_bare_top.real, '--', color = p[0].get_color())
        p = pl.plot(x_top, mz_top.real, label='Re m_z @ top')
        pl.plot(x_top, mz_bare_top.real, '--', color = p[0].get_color())
        pl.legend()

        pl.figure()
        p = pl.plot(x_bot, mx_bot.imag, label='Im m_x @ bot')
        pl.plot(x_bot, mx_bare_bot.imag, '--', color = p[0].get_color())
        p = pl.plot(x_bot, mz_bot.imag, label='Im m_z @ bot')
        pl.plot(x_bot, mz_bot.imag, '--', color = p[0].get_color())
        p = pl.plot(x_top, mx_top.imag, label='Im m_x @ top')
        pl.plot(x_top, mx_bare_top.imag, '--', color = p[0].get_color())
        p = pl.plot(x_top, mz_top.imag, label='Im m_z @ top')
        pl.plot(x_top, mz_bare_top.imag, '--', color = p[0].get_color())
        pl.legend()

        pl.figure()
        pl.tripcolor(mode.X, mode.Z, np.angle(mx), cmap='hsv')
                     #vmin=-np.pi, vmax=np.pi)
        pl.colorbar()
        pl.tripcolor(mode.X, mode.Z + 1.2 * res_H, np.angle(mz), cmap='hsv')
                     #vmin=-np.pi, vmax=np.pi)
        pl.colorbar() 
        

        pl.figure()
        m_max = max(np.max(np.abs(mx)), np.max(np.abs(mz)))
        dx_min = mode.Z[1] - mode.Z[0]
        q_scale = 1.2 * m_max / dx_min 
        pl.gca().set_aspect('equal', 'box')
        pl.tripcolor(mode.X, mode.Z, np.abs(mx), cmap='magma',
                     vmin = 0.0, vmax = m_max)
        pl.quiver(mode.X, mode.Z + res_H * 1.1, mx.real, mz.real,
                  color='red', pivot='middle', scale_units='x',
                  scale = q_scale)
        pl.quiver(mode.X, mode.Z + res_H * 1.1, mx.imag, mz.imag,
                  color='blue', pivot='middle', scale_units='x',
                  scale = q_scale)
        pl.tripcolor(mode.X, mode.Z + res_H * 2.2, np.abs(mz), cmap='magma',
                     vmin = 0.0, vmax = m_max)
        pl.tripcolor(bare_mode.X, bare_mode.Z + res_H * 4.0,
                     np.abs(mx_bare), cmap='magma', vmin = 0.0, vmax = m_max)
        pl.quiver(bare_mode.X, bare_mode.Z + res_H * 5.1,
                  mx_bare.real, mz_bare.real, color='red',
                  pivot='middle', scale_units='x', scale=q_scale)
        pl.quiver(bare_mode.X, bare_mode.Z + res_H * 5.1,
                  mx_bare.imag, mz_bare.imag, color='blue', pivot='middle',
                  scale_units='x', scale=q_scale)
        pl.tripcolor(bare_mode.X, bare_mode.Z + res_H * 6.2,
                     np.abs(mz_bare), cmap='magma', vmin = 0.0, vmax = m_max)

        i_max = np.argmax(np.abs(mx))
        mx_max = mx[i_max]
        mz_max = mz[i_max]
        i_bare_max = np.argmax(np.abs(mx_bare))
        mx_bare_max = mx_bare[i_bare_max]
        mz_bare_max = mz_bare[i_bare_max]

        theta = np.linspace(-np.pi, np.pi, 1001)
        exp_th = np.exp(1j * theta)

        pl.figure()
        pl.plot((mx_max * exp_th).real, (mz_max * exp_th).real)
        pl.plot((mx_bare_max * exp_th).real, (mz_bare_max * exp_th).real, '--')
        pl.gca().set_aspect('equal', 'box')

        pl.show()

        
