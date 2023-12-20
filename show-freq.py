import numpy as np
import sys
import constants

def show_freq(fname):
    d = np.load(fname)
    print (fname, d['omega'].real / constants.GHz_2pi)

for fname in sys.argv[1:]:
    show_freq(fname)
