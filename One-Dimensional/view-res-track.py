import numpy as np
import pylab as pl
import constants
import sys
from scipy import linalg

def readData(fname):
    result = {}
    result['fname'] = fname
    d = np.load(fname)
    #for k in d.keys():
    #    print(k)
    result['freq'] = d['omega'] / constants.GHz_2pi
    result['Tfw'] = d['T1_bare']
    result['Tbk'] = d['T2_bare']
    result['Ms'] = d['res_Ms'] / constants.kA_m
    return result



results = []
for fname in sys.argv[1:]:
    result = readData(fname)
    results.append(result)

Ms_res = []
for i_res, result in enumerate(results):
    Ms_res.append((result['Ms'], i_res))

Ms_res.sort(key = lambda i: np.abs(i[0]))

Ms_vals = []
f_vals = []
T_fw_vals = []
T_bk_vals = []

for Ms, i_res in Ms_res:
    result = results[i_res]
    Ms_vals.append(Ms)
    if len(f_vals) < 1:
        f_vals = np.array(result['freq'])
    else:
        if len(f_vals) != len(result['freq']):
            raise Exception("different lengths for freq axis")
        if linalg.norm(f_vals - result['freq']) > 1e-6:
            raise Exception("different freq vals")
    T_fw_vals.append(result['Tfw'])
    T_bk_vals.append(result['Tbk'])

Ms_vals = np.array(Ms_vals)
T_fw_vals = np.array(T_fw_vals)
T_bk_vals = np.array(T_bk_vals)

F, M = np.meshgrid(f_vals, Ms_vals)

print ("M: ", np.shape(M), "T:", np.shape(T_fw_vals))

#T_cmap = 'autumn'
#T_cmap = 'hot'
#T_cmap = 'afmhot'
T_cmap="gist_heat"

pl.figure()
pl.pcolormesh(F, M, np.abs(T_fw_vals), vmin=0.0, vmax=1.01, cmap=T_cmap)
cb = pl.colorbar()
cb.set_label(r"Transmissivity $|T(f)|$")
pl.xlabel(r"Frequency, GHz")
pl.ylabel(r"Magnetisation in the resonator, kA/m")
orientation = ""
if    min(Ms_vals) < 0 and max(Ms_vals) < 0:
    orientation = "(antiparallel)"
elif  min(Ms_vals) > 0 and max(Ms_vals) > 0:
    orientation = "(parallel)"
pl.title('Forward transmissivity %s' % orientation)

pl.figure()
pl.pcolormesh(F, M, np.angle(T_fw_vals)/np.pi * 180.0, vmin=-180,
              vmax=180, cmap='hsv')
cb = pl.colorbar()
cb.set_label(r"Phase arg $T(f)$, deg")
cb.set_ticks([-180.0, -90.0, 0.0, 90.0, 180.0])
pl.xlabel(r"Frequency, GHz")
pl.ylabel(r"Magnetisation in the resonator, kA/m")
pl.title('Forward transmission %s' % orientation)

pl.figure()
pl.pcolormesh(F, M, np.abs(T_bk_vals), vmin=0.0, vmax=1.01, cmap=T_cmap)
cb = pl.colorbar()
cb.set_label(r"Transmissivity $|T(f)|$")
pl.xlabel(r"Frequency, GHz")
pl.ylabel(r"Magnetisation in the resonator, kA/m")
pl.title('Backward transmission %s' % orientation)

pl.figure()
pl.pcolormesh(F, M, np.angle(T_bk_vals)/np.pi * 180.0,
              vmin=-180.0, vmax=180.0, cmap='hsv')
cb = pl.colorbar()
cb.set_label(r"Phase arg $ T(f)$")
cb.set_ticks([-180.0, -90.0, 0.0, 90.0, 180.0])
pl.xlabel(r"Frequency, GHz")
pl.ylabel(r"Magnetisation in the resonator, kA/m")
pl.title('Backward transmission %s' % orientation)


pl.show()


