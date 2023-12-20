import numpy as np
import pylab as pl
import constants
import sys

def readData(fname):
    result = {}
    result['fname'] = fname
    d = np.load(fname)
    for k in d.keys():
        print(k)
    result['freq'] = d['omega'] / constants.GHz_2pi
    if 'T1_bare' in d.keys():
       result['Tfw'] = d['T1_bare']
       result['Tbk'] = d['T2_bare']
    elif 'T1_nrf' in d.keys():
       result['Tfw'] = d['T1_nrf']
       result['Tbk'] = d['T2_nrf']        
    return result



results = []
for fname in sys.argv[1:]:
    result = readData(fname)
    results.append(result)

pl.figure()
for result in results:
    p = pl.plot(result['freq'], np.abs(result['Tfw']), label=result['fname'])
    pl.plot(result['freq'], np.abs(result['Tbk']),
            '--', color=p[0].get_color())
pl.xlabel(r'Frequency $\omega/2\pi$, GHz')
pl.ylabel(r'Transmissivity $|T(\omega)|$')
pl.legend()

pl.figure()
for result in results:
    p = pl.plot(result['freq'], np.angle(result['Tfw']), label=result['fname'])
    pl.plot(result['freq'], np.angle(result['Tbk']),
            '--', color=p[0].get_color())
pl.xlabel(r'Frequency $\omega/2\pi$, GHz')
pl.ylabel(r'Phase $arg T(\omega)$')
pl.legend()
pl.show()



