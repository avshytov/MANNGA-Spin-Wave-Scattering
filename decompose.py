import numpy as np
import constants

class Mode:
    def __init__ (self, X, Y, Z, dV, ma, mb, omega):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.dV = dV
        self.ma = ma
        self.mb = mb
        self.omega = omega
        

def readMode(fname):
    d = np.load(fname)
    for k in d.keys(): print(k)
    mode = Mode(d['X'], d['Y'], d['Z'], d['dV'], d['ma'], d['mb'], d['omega'])
    return mode

nrf_mode = readMode('NRF3d-mode-0-60x60x10nm-10mT-s=20nm-d=20nm-32x32x3.npz')
bare_mode_files = [
    "bare/BARE3d-mode-0-60x60x10nm-10mT-32x32x3.npz",
    "bare/BARE3d-mode-1-60x60x10nm-10mT-32x32x3.npz",
    "bare/BARE3d-mode-2-60x60x10nm-10mT-32x32x3.npz",
    "bare/BARE3d-mode-3-60x60x10nm-10mT-32x32x3.npz"
]

nrf_mode = readMode('NRF3d-mode-0-60x60x10nm-20mT-s=20nm-d=20nm-32x32x3.npz')
bare_mode_files = [
    "bare/BARE3d-mode-0-60x60x10nm-20mT-32x32x3.npz",
    "bare/BARE3d-mode-1-60x60x10nm-20mT-32x32x3.npz",
    "bare/BARE3d-mode-2-60x60x10nm-20mT-32x32x3.npz",
    "bare/BARE3d-mode-3-60x60x10nm-20mT-32x32x3.npz"
]

bare_modes = [readMode(t) for t in bare_mode_files]

def dotprod(mode1, mode2):
    prod  = mode1.ma.conj() * mode2.mb
    prod -= mode1.mb.conj() * mode2.ma
    prod *= mode1.dV
    return np.sum(prod)

def overlap(mode1, mode2):
    C11 = dotprod(mode1, mode1)
    C22 = dotprod(mode2, mode2)
    C12 = dotprod(mode1, mode2)
    return np.abs(C12 / np.sqrt(np.abs(C11) * np.abs(C22)))

for bare_mode in bare_modes:
    print ("bare mode @ ", bare_mode.omega / constants.GHz_2pi, " overlap: ",
           overlap(nrf_mode, bare_mode))

for bare_mode1 in bare_modes:
    for bare_mode2 in bare_modes:
        print ("bare overlaps:", bare_mode1.omega / constants.GHz_2pi,
               bare_mode2.omega / constants.GHz_2pi,
               overlap(bare_mode1, bare_mode2))
    
