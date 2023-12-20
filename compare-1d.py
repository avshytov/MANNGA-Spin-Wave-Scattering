import numpy as np
import pylab as pl
import sys

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
    for k in d.keys(): print (k)
    omega = d['omega']
    print (np.shape(omega))
    T1 = d['T1']
    T2 = d['T2']
    R1 = d['R1']
    R2 = d['R2']
    A1 = d['A1']
    A2 = d['A2']
    Gamma_rad = d['Gamma_rad']
    W = 0.0
    L = 0.0
    H = 0.0
    #film_d = 0.0
    #film_a = 0.0
    #film_b = 0.0
    z_res = d['z_res']
    res_type = 'UNI'
    if 'res_anti_L' in d.keys():
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
        #if k[-2:] == '_a':
        #    film_a = d[k]
        #if k[-2:] == '_b':
        #    film_b = d[k]

    #film_d = np.abs(film_b - film_a)

    return dict(fname=fname, W=W / nm, L=L / nm, H=H / nm,
                z_res=z_res / nm, slab_d = slab_d / nm,
                omega = omega, Gamma_rad = Gamma_rad,
                T1 = T1, R1 = R1, T2 = T2, R2 = R2, A1 = A1, A2 = A2,
                res_type=res_type)

    


dataset = []
for fname in sys.argv[1:]:
    data = readData(fname)
    dataset.append(data)

pl.figure()
for data in dataset:
    lab = r'%s $%gx%gx%g$ @ $%g$nm, $d = %g$nm' % (data['res_type'],
                                       data['L'], data['W'], data['H'],
                                       data['z_res'], data['slab_d'])
    plot_sorted(data['omega'] / GHz_2pi, data['Gamma_rad'] * nsec,
                label=lab)
    
pl.xlabel(r'Frequency $\omega/ 2\pi$, GHz')
pl.ylabel(r'Radiative linewidth $\Gamma_{rad})(\omega)$, $\rm{ns}^{-1}$')
pl.legend()

pl.figure()
for data in dataset:
    lab = r'%s $%gx%gx%g$ @ $%g$nm, $d = %g$nm' % (data['res_type'],
                                       data['L'], data['W'], data['H'],
                                       data['z_res'], data['slab_d'])
    p = plot_sorted(data['omega'] / GHz_2pi, np.abs(data['T1']),
                label=lab)
    p = plot_sorted(data['omega'] / GHz_2pi, np.abs(data['T2']),
                    '--', color=p[0].get_color())
    
pl.xlabel(r'Frequency $\omega/ 2\pi$, GHz')
pl.ylabel(r'Transmittance $T(\omega)$')
pl.legend()

pl.figure()
for data in dataset:
    lab = r'%s $%gx%gx%g$ @ $%g$nm, $d = %g$nm' % (data['res_type'],
                                       data['L'], data['W'], data['H'],
                                       data['z_res'], data['slab_d'])
    p = plot_sorted(data['omega'] / GHz_2pi, np.abs(data['R1']),
                label=lab)
    p = plot_sorted(data['omega'] / GHz_2pi, np.abs(data['R2']),
                    '--', color=p[0].get_color())
    
pl.xlabel(r'Frequency $\omega/ 2\pi$, GHz')
pl.ylabel(r'Reflectivity $R(\omega)$')
pl.legend()

pl.figure()
for data in dataset:
    lab = r'%s $%gx%gx%g$ @ $%g$nm, $d = %g$nm' % (data['res_type'],
                                       data['L'], data['W'], data['H'],
                                       data['z_res'], data['slab_d'])
    p = plot_sorted(data['omega'] / GHz_2pi, data['A1'],
                label=lab)
    p = plot_sorted(data['omega'] / GHz_2pi, data['A2'],
                    '--', color=p[0].get_color())
    
pl.xlabel(r'Frequency $\omega/ 2\pi$, GHz')
pl.ylabel(r'Absorbance $A(\omega)$')
pl.legend()

pl.show()
