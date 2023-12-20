import numpy as np
import pylab as pl
import sys
import constants

def get_params(fname):
    d = np.load(fname)
    for k in d.keys():
        print (k)
    Omega_res = d['Omega_res']
    Omega_0   = d['omega'].real
    Z_res     = d['Z_res']
    Gamma_res = d['Gamma_res']
    Gamma_0   = -d['omega'].imag
    result = {}
    result['Omega_res'] = Omega_res
    result['fname'] = fname
    result['Omega_0'] = Omega_0
    result['Z_res'] = Z_res
    result['Gamma_res'] = Gamma_res
    result['s'] = d['s']
    result['d_film'] = d['d_film']
    return result

data = []
for fname in sys.argv[1:]:
    data.append(get_params(fname))

d_values = set()
for result in data:
    print (result['d_film'])
    d_values.add(float(result['d_film']))

print ("d_values = ", d_values)

pl.figure()
for d in d_values:
    s_vals = []
    Z_vals = []
    Gamma_res_vals = []
    for result in data:
        if np.abs(result['d_film'] - d) > 1e-6: continue
        s_vals.append(result['s'])
        Z_vals.append(result['Z_res'])
        Gamma_res_vals.append(result['Gamma_res'])
    res_vals = list(zip(s_vals, Gamma_res_vals, Z_vals))
    res_vals.sort( key = lambda x: x[0])
    s_vals = np.array([t[0] for t in res_vals])
    #Omega_0_vals   = np.array([t[1] for t in res_vals])
    Gamma_res_vals = np.array([t[1] for t in res_vals])
    Z_vals = np.array([t[2] for t in res_vals])

    p = pl.plot(s_vals / constants.nm,
                Z_vals * Gamma_res_vals * constants.ns,
                'o-',    label='$d = %g$nm' % (d/constants.nm))
    #pl.plot(s_vals / constants.nm, Omega_0_vals / constants.GHz_2pi,
    #        'd--', color=p[0].get_color())

pl.legend()
pl.xlabel(r"Spacing $s$, nm")
pl.ylabel(r"Mode width $\Gamma_{\rm res}$, ${\rm ns}^{-1}$")
pl.figure()
for d in d_values:
    s_vals = []
    Omega_0_vals = []
    Omega_res_vals = []
    for result in data:
        if np.abs(result['d_film'] - d) > 1e-6: continue
        s_vals.append(result['s'])
        Omega_0_vals.append(result['Omega_0'])
        Omega_res_vals.append(result['Omega_res'])
    res_vals = list(zip(s_vals, Omega_0_vals, Omega_res_vals))
    res_vals.sort( key = lambda x: x[0])
    s_vals = np.array([t[0] for t in res_vals])
    Omega_0_vals   = np.array([t[1] for t in res_vals])
    Omega_res_vals = np.array([t[2] for t in res_vals])

    p = pl.plot(s_vals / constants.nm, Omega_res_vals / constants.GHz_2pi,
                'o-',    label='$d = %g$nm' % (d/constants.nm))
    pl.plot(s_vals / constants.nm, Omega_0_vals / constants.GHz_2pi,
            'd--', color=p[0].get_color())

pl.legend()
pl.xlabel(r"Spacing $s$, nm")
pl.ylabel(r"Mode frequency $\omega/2\pi$, GHz")
pl.show()
    

