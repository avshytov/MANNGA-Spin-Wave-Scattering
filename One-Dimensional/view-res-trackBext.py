import numpy as np
import pylab as pl
import constants
import sys
from scipy import linalg

def readData(fname):
    print ("read: ", fname)
    result = {}
    result['fname'] = fname
    d = np.load(fname)
    #for k in d.keys():
    #    print(k)
    result['freq'] = d['omega'] / constants.GHz_2pi
    result['Tfw'] = d['T1_bare']
    result['Tbk'] = d['T2_bare']
    result['Ms'] = d['res_Ms'] / constants.kA_m
    result['Bext'] = d['Bext'] / constants.mT
    result['f_min'] = d['omega_min'] / constants.GHz_2pi
    result['f_max'] = d['omega_max'] / constants.GHz_2pi
    
    return result



results = []
for fname in sys.argv[1:]:
    result = readData(fname)
    results.append(result)

Bext_res = []
for i_res, result in enumerate(results):
    Bext_res.append((result['Bext'], i_res))

Bext_res.sort(key = lambda i: np.abs(i[0]))

Bext_vals = []
f_vals = []
T_fw_vals = []
T_bk_vals = []
f_min_vals = []
f_max_vals = []

for Bext, i_res in Bext_res:
    result = results[i_res]
    Bext_vals.append(Bext)
    if len(f_vals) < 1:
        f_vals = np.array(result['freq'])
    else:
        if len(f_vals) != len(result['freq']):
            raise Exception("different lengths for freq axis")
        if linalg.norm(f_vals - result['freq']) > 1e-6:
            raise Exception("different freq vals")
    T_fw_vals.append(result['Tfw'])
    T_bk_vals.append(result['Tbk'])
    f_min_vals.append(result['f_min'])
    f_max_vals.append(result['f_max'])

Bext_vals = np.array(Bext_vals)
T_fw_vals = np.array(T_fw_vals)
T_bk_vals = np.array(T_bk_vals)
f_min_vals = np.array(f_min_vals)
f_max_vals = np.array(f_max_vals)

F, B = np.meshgrid(f_vals, Bext_vals)

print ("M: ", np.shape(B), "T:", np.shape(T_fw_vals))

#T_cmap = 'spring'
T_cmap = 'autumn'
#T_cmap = 'hot'
#T_cmap = 'afmhot'
#T_cmap="gist_heat"

pl.figure()
pl.pcolormesh(F, B, np.abs(T_fw_vals), vmin=0.0, vmax=1.01, cmap=T_cmap)
cb = pl.colorbar()
cb.set_label(r"Transmissivity $|T(f)|$")
pl.xlabel(r"Frequency, GHz")
pl.ylabel(r"Bias field $B_{\rm ext}$, mT")
orientation = ""
#if    min(Ms_vals) < 0 and max(Ms_vals) < 0:
#    orientation = "(antiparallel)"
#elif  min(Ms_vals) > 0 and max(Ms_vals) > 0:
#    orientation = "(parallel)"
pl.title('Forward transmissivity %s' % orientation)
pl.plot(f_min_vals, Bext_vals, 'k--')
pl.plot(f_max_vals, Bext_vals, 'k--')
pl.fill_betweenx(Bext_vals, 0.0 * Bext_vals + f_vals[0],
                f_min_vals, color='white')
pl.fill_betweenx(Bext_vals, 0.0 * Bext_vals + max(f_max_vals[0], f_vals[-1]),
                f_max_vals, color='white')
pl.xlim(f_vals[0], f_vals[-1])
pl.ylim(Bext_vals[0], Bext_vals[-1])

pl.figure()
pl.pcolormesh(F, B, np.angle(T_fw_vals)/np.pi * 180.0, vmin=-180,
              vmax=180, cmap='hsv')
cb = pl.colorbar()
cb.set_label(r"Phase arg $T(f)$, deg")
cb.set_ticks([-180.0, -90.0, 0.0, 90.0, 180.0])
pl.xlabel(r"Frequency, GHz")
pl.ylabel(r"Bias field $B_{\rm ext}$, mT")
pl.title('Forward transmission %s' % orientation)
pl.plot(f_min_vals, Bext_vals, 'k--')
pl.plot(f_max_vals, Bext_vals, 'k--')
pl.fill_betweenx(Bext_vals, 0.0 * Bext_vals + f_vals[0],
                f_min_vals, color='white')
pl.fill_betweenx(Bext_vals, 0.0 * Bext_vals + max(f_max_vals[0], f_vals[-1]),
                f_max_vals, color='white')
pl.xlim(f_vals[0], f_vals[-1])
pl.ylim(Bext_vals[0], Bext_vals[-1])



pl.figure()
pl.pcolormesh(F, B, np.abs(T_bk_vals), vmin=0.0, vmax=1.01, cmap=T_cmap)
cb = pl.colorbar()
cb.set_label(r"Transmissivity $|T(f)|$")
pl.xlabel(r"Frequency, GHz")
pl.ylabel(r"Bias field $B_{\rm ext}$, mT")
pl.title('Backward transmission %s' % orientation)
pl.plot(f_min_vals, Bext_vals, 'k--')
pl.plot(f_max_vals, Bext_vals, 'k--')
pl.fill_betweenx(Bext_vals, 0.0 * Bext_vals + f_vals[0],
                f_min_vals, color='white')
pl.fill_betweenx(Bext_vals, 0.0 * Bext_vals + max(f_max_vals[0], f_vals[-1]),
                f_max_vals, color='white')
pl.xlim(f_vals[0], f_vals[-1])
pl.ylim(Bext_vals[0], Bext_vals[-1])


pl.figure()
pl.pcolormesh(F, B, np.angle(T_bk_vals)/np.pi * 180.0,
              vmin=-180.0, vmax=180.0, cmap='hsv')
cb = pl.colorbar()
cb.set_label(r"Phase arg $ T(f)$")
cb.set_ticks([-180.0, -90.0, 0.0, 90.0, 180.0])
pl.xlabel(r"Frequency, GHz")
pl.ylabel(r"Bias field $B_{\rm ext}$, mT")
pl.title('Backward transmission %s' % orientation)
pl.plot(f_min_vals, Bext_vals, 'k--')
pl.plot(f_max_vals, Bext_vals, 'k--')
pl.fill_betweenx(Bext_vals, 0.0 * Bext_vals + f_vals[0],
                f_min_vals, color='white')
pl.fill_betweenx(Bext_vals, 0.0 * Bext_vals + max(f_max_vals[0], f_vals[-1]),
                f_max_vals, color='white')
pl.xlim(f_vals[0], f_vals[-1])
pl.ylim(Bext_vals[0], Bext_vals[-1])


pl.show()


