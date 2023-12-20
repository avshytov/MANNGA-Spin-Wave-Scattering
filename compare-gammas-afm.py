import numpy as np
import pylab as pl
import sys

from printconfig import print_config

from constants import nm, GHz_2pi, ns

nsec = ns

def plot_sorted(x, y, *args, **kwargs):
    xy = zip(list(x), list(y))
    xy = list(xy)
    xy.sort(key = lambda t: t[0])
    x_new = np.array([t[0] for t in xy])
    y_new = np.array([t[1] for t in xy])
    return pl.plot(x_new, y_new, *args, **kwargs)

def readData(fname):
    d = np.load(fname)
    print ("***** ", fname, "*******")
    print_config(d)
    for k in d.keys(): print (k)
    omega = d['omega_tab']
    print (np.shape(omega))
    sigma_abs_o = d['scat_abs_o']
    print (np.shape(sigma_abs_o))
    sigma_tot_o = d['scat_tot_o']
    gamma_rad_o = d['gamma_rad_o']
    print (np.shape(gamma_rad_o))
    theta_tab = d['theta_tab']
    i_th0 = np.argmin(np.abs(theta_tab - 0))
    sigma_th0 = np.abs(d['result'][i_th0])**2
    W = 0.0
    L = 0.0
    H = 0.0
    z_res = d['z_res']
    res_type = 'UNI'
    #if 'res_anti_L' in d.keys():
    res_type = 'AFM'
    slab_a = d['slab_a']
    slab_b = d['slab_b']
    slab_d = slab_b - slab_a
    for k in d.keys():
        if k[-2:] == '_W':
            W = d[k]
        if k[-2:] == '_L':
            L = d[k]
        if k[-2:] == '_H':
            H = d[k]

    z_res = z_res - slab_b

    f_0 = d['res_omega_0'] / GHz_2pi

    return dict(fname=fname, W=W / nm, L=L / nm, H=H / nm,
                z_res=z_res / nm, slab_d = slab_d / nm,
                omega = omega, gamma_rad_o = gamma_rad_o,
                sigma_tot_o=sigma_tot_o, sigma_abs_o = sigma_abs_o,
                res_type=res_type, f_0 = f_0, sigma_th0 = sigma_th0)

    


dataset = []
for fname in sys.argv[1:]:
    data = readData(fname)
    dataset.append(data)

pl.figure()
for data in dataset:
    #lab = r'%s $%gx%gx%g$ @ $%g$nm' % (data['res_type'],
    #                                   data['L'], data['W'], data['H'],
    #                                   data['z_res'])
    lab = r'%s, $f_\mathrm{res} = %.3g$GHz @ $%g$nm' % (data['res_type'],
                                       data['f_0'],
                                       data['z_res'])
    print (np.shape(data['omega']), np.shape(data['gamma_rad_o']))
    plot_sorted(data['omega'] / GHz_2pi, data['gamma_rad_o'] * nsec,
                label=lab)
pl.xlabel(r'Frequency $\omega/ 2\pi$, GHz')
pl.ylabel(r'Radiative linewidth $\Gamma_{rad})(\omega)$, $\rm{ns}^{-1}$')
pl.legend()

pl.figure()
for data in dataset:
    #lab = r'%s $%gx%gx%g$ @ $%g$nm' % (data['res_type'],
    #                                   data['L'], data['W'], data['H'],
    #                                   data['z_res'])
    lab = r'%s, $f_\mathrm{res} = %.3g$GHz @ $%g$nm' % (data['res_type'],
                                       data['f_0'],
                                       data['z_res'])
    print (np.shape(data['omega']), np.shape(data['sigma_tot_o']))
    p = plot_sorted(data['omega'] / GHz_2pi, data['sigma_tot_o'] / nm,
                label=lab)
    p = plot_sorted(data['omega'] / GHz_2pi, data['sigma_abs_o'] / nm,
                    '--', color=p[0].get_color())

pl.xlabel(r'Frequency $\omega/ 2\pi$, GHz')
pl.ylabel(r'Scattering cross-secton $\sigma_{tot})(\omega)$, $nm$')
pl.legend()

pl.figure()
for data in dataset:
    #lab = r'%s $%gx%gx%g$ @ $%g$nm' % (data['res_type'],
    #                                   data['L'], data['W'], data['H'],
    #                                   data['z_res'])
    lab = r'%s, $f_\mathrm{res} = %.3g$GHz @ $%g$nm' % (data['res_type'],
                                       data['f_0'],
                                       data['z_res'])
    p = plot_sorted(data['omega'] / GHz_2pi, data['sigma_th0'] / nm,
                label=lab)

pl.xlabel(r'Frequency $\omega/ 2\pi$, GHz')
pl.ylabel(r'Forward scattering cross-section $\sigma(\omega, \theta=0)$, $nm$')
pl.legend()

pl.figure()
for data in dataset:
    #lab = r'%s $%gx%gx%g$ @ $%g$nm' % (data['res_type'],
    #                                   data['L'], data['W'], data['H'],
    #                                   data['z_res'])
    lab = r'%s, $f_\mathrm{res} = %.3g$GHz @ $%g$nm' % (data['res_type'],
                                       data['f_0'],
                                       data['z_res'])
    print (np.shape(data['omega']), np.shape(data['sigma_abs_o']))
    plot_sorted(data['omega'] / GHz_2pi, data['sigma_abs_o'] / nm,
                label=lab)

pl.xlabel(r'Frequency $\omega/ 2\pi$, GHz')
pl.ylabel(r'Absorption cross-secton $\sigma_{abs})(\omega)$, $nm$')
pl.legend()
pl.figure()
for data in dataset:
    #lab = r'%s $%gx%gx%g$ @ $%g$nm' % (data['res_type'],
    #                                   data['L'], data['W'], data['H'],
    #                                   data['z_res'])
    lab = r'%s, $f_\mathrm{res} = %.3g$GHz @ $%g$nm' % (data['res_type'],
                                       data['f_0'],
                                       data['z_res'])
    print (np.shape(data['omega']), np.shape(data['sigma_tot_o']))
    plot_sorted(data['omega'] / GHz_2pi, data['sigma_tot_o'] / nm,
                label=lab)

pl.xlabel(r'Frequency $\omega/ 2\pi$, GHz')
pl.ylabel(r'Scattering cross-secton $\sigma_{tot})(\omega)$, $nm$')
pl.legend()
pl.show()
