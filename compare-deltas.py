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
    pl.plot(x_new, y_new, *args, **kwargs)

def readData(fname):
    d = np.load(fname)
    for k in d.keys(): print (k)
    omega = d['omega']
    theta = d['theta']
    Delta = d['Delta_theta']
    k = d['k_theta']
    alpha = d['alpha_theta']
    Gamma_rad = d['Gamma_rad']
    W = 0.0
    L = 0.0
    H = 0.0
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

    return dict(fname=fname, W=W / nm, L=L / nm, H=H / nm,
                z_res=z_res / nm, slab_d = slab_d / nm,
                omega = omega,
                theta=theta, Delta = Delta, alpha = alpha, k = k,
                Gamma_rad = Gamma_rad, 
                res_type=res_type)

    


dataset = []
for fname in sys.argv[1:]:
    data = readData(fname)
    dataset.append(data)

pl.figure()
for data in dataset:
    lab = r'%s $%gx%gx%g$ @ $%g$nm' % (data['res_type'],
                                       data['L'], data['W'], data['H'],
                                       data['z_res'])
    print (np.shape(data['theta']), np.shape(data['Delta']))
    pl.polar(data['theta'], np.abs(data['Delta']), label=lab)
pl.legend()
pl.title(r"$\Delta_\theta(\theta)$, $\omega = %g$ GHz" % (data['omega'] / GHz_2pi))

pl.figure()
for data in dataset:
    lab = r'%s $%gx%gx%g$ @ $%g$nm' % (data['res_type'],
                                       data['L'], data['W'], data['H'],
                                       data['z_res'])
    print (np.shape(data['theta']), np.shape(data['Delta']))
    pl.polar(data['alpha'], np.abs(data['Delta']), label=lab)
pl.legend()
pl.title(r"$\Delta_\theta(\alpha)$, $\omega = %g$ GHz" % (data['omega'] / GHz_2pi))

pl.show()
