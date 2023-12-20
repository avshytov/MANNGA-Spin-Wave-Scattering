import numpy as np
import pylab as pl
import sys

from constants import GHz_2pi, nm

def readData(fname):
    result = {}
    d = np.load(fname)
    print ("==========", fname)
    for k in d.keys(): print (k)
    result['f']  = d['omega'] / GHz_2pi
    result['T1'] = d['T1_bare']
    result['T2'] = d['T2_bare']
    result['R1'] = d['R1_bare']
    result['R2'] = d['R2_bare']
    result['s'] = d['z_res']
    return result

results = []
for fname in sys.argv[1:]:
    results.append(readData(fname))
    
results.sort(key = lambda r: r['s'])

pl.figure()
for d in results:
    pl.plot(d['f'], np.abs(d['T1']), label=r'$s = %g$nm' % (d['s'] / nm))

pl.xlabel(r'Frequendy $\omega/2\pi$, GHz')
pl.ylabel(r'Forw transmissity $|T_{\rm fw}(\omega)|$')
pl.legend()

pl.figure()
for d in results:
    pl.plot(d['f'], np.abs(d['T2']), label=r'$s = %g$nm' % (d['s'] / nm))

pl.xlabel(r'Frequendy $\omega/2\pi$, GHz')
pl.ylabel(r'Backw transmissity $|T_{\rm bk}(\omega)|$')
pl.legend()

pl.figure()
for d in results:
    pl.plot(d['f'], np.abs(d['R1']), label=r'$s = %g$nm' % (d['s'] / nm))

pl.xlabel(r'Frequendy $\omega/2\pi$, GHz')
pl.ylabel(r'Reflectivity $|R(\omega)|$')
pl.legend()

pl.show()
