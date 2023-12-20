import numpy as np
import pylab as pl
import sys

def readData(fname, o_plot):
    d = np.load(fname)
    for k in d.keys():
        print(k)
    o = d['omega']
    p = d['part_ratio']
    print ("max part ratio: ", np.max(p))
    #o_plot = np.linspace(0.0, 5.0, 2001)
    gamma = 0.03
    d_plot = 0.0 * o_plot
    for omega, p_ratio in zip(o, p):
        d_plot += p_ratio * gamma / np.pi / ((o_plot - omega.imag)**2  + gamma**2) 

    #pl.plot(o_plot, d_plot, label=r'$s = %g$nm' % d['s'])
    #pl.legend()
    return d_plot, d['s']
    #pl.show()


fnames = sys.argv[1:]
o_plot = np.linspace(0.0, 5.0, 2001)
d_data = []
for fname in fnames:
    d_plot, s = readData(fname, o_plot)
    d_data.append((d_plot, s))

d_data.sort(key = lambda i: i[1])
s_vals = np.array([t[1] for t in d_data])
S, O = np.meshgrid(s_vals, o_plot)
D = 0.0 * O
pl.figure()
for i_data, data in enumerate(d_data):
    d_plot, s = data
    D[:, i_data] = d_plot
pl.pcolor(S, O, D, cmap='magma', vmin=0.0, vmax = np.max(D))
pl.colorbar()
pl.figure()
pl.pcolor(S, O, np.log(D), cmap='bwr', vmin=-5.0, vmax = 5.0)
pl.colorbar()
pl.show()
