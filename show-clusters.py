import numpy as np
import pylab as pl
import sys
from clusterize import Clusterizer

class mode_norm_quasi:
    def __init__ (self, f, p):
        self.p = p
        self.f = f

    def participation(self, *args):
        return self.p

    

def show_data(fname):
    d = np.load(fname)
    for k in d.keys():
        print (k)
    clusterizer = Clusterizer(None, ['resonator'], ['film'])
    C_ij = d['C_ij']
    N, M = np.shape(C_ij)
    modes = []
    f_vals = d['f_vals']
    p_vals = d['p_vals']
    volume_ratio = d['volume_ratio']
    for i in range(N):
        modes.append(mode_norm_quasi(f_vals[i], p_vals[i]))
    clusterizer.mode_norm = modes
    clusterizer.C_ij = d['C_ij']
    clusterizer.S_ij = d['S_ij']
    clusterizer.clusterize_modes()

    cluster_results = clusterizer.analyze_clusters()

    def get_p_max_no(cluster_data):
        return np.argmax(np.abs(cluster_data[1]))
    def get_f_max(cluster_data):
        return cluster_data[0][get_p_max_no(cluster_data)].real

    clusters_sorted = list(enumerate(cluster_results))
    clusters_sorted.sort(key = lambda x: get_f_max (x[1]))

    pl.figure()
    for cluster, c_result in clusters_sorted: #enumerate(cluster_results):
        f_vals, p_vals, C_cluster, S_cluster = c_result
        max_p_in_cluster = np.argmax(np.abs(p_vals))
        f_max = f_vals[max_p_in_cluster]
        mode_max_no = clusterizer.clusters[cluster][max_p_in_cluster]
        p_max = p_vals[max_p_in_cluster]
        #modes_to_show.append((mode_max_no, f_max, p_max / volume_ratio))
        f_axis = np.linspace(np.min(f_vals.real), np.max(f_vals.real),
                                 2000)
        p_axis = 0.0 * f_axis
        gamma = 1e-3
        for f_val, p_val in zip(f_vals, p_vals):
            p_cur = p_val / volume_ratio
            f_cur = f_val.real
            column = (gamma**2 / (gamma**2 + (f_axis - f_cur)**2))**2
            p_axis += column * p_cur
        p = pl.plot(f_vals.real, p_vals  / volume_ratio, '-o', ms=5.0,
                    label='cluster @ %g' % f_max.real)
        #p = pl.plot(f_vals.real, p_vals  / volume_ratio, '-')
        #pl.plot(f_axis, p_axis, '-o', ms=1.0,
        #        label='cluster @ %g' % f_max.real, 
        #        color=p[0].get_color())
    #for nrf_mode in nrf_mode_collection:
    #    nrf_f = nrf_mode.omega / constants.GHz_2pi
    #    pl.plot([nrf_f, nrf_f], [0.0, 10.0], 'k--')
    pl.legend()
    pl.xlabel(r"Frequency $f$, GHz")
    pl.ylabel(r"Mode participation")
    pl.show()
    



for fname in sys.argv[1:]:
    show_data(fname)
