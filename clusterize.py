import numpy as np

class Clusterizer:
    
    def __init__ (self, model, overlap_areas, rest_areas):
        self.model = model
        self.mode_norm = []
        self.overlap_areas = overlap_areas
        self.rest_areas    = rest_areas
        self.C_ij = None
        self.S_ij = None
        self.clusters = []

    def compute_mode_overlap(self, mode_set):
        mode_norm = []
        model = self.model
        for mode in mode_set:
            #ma_i, mb_i = mode.m_ab_all()
            C_res = model.dot_product(mode, mode, *self.overlap_areas)
            #print ("C_res = ", C_res)

            mode_copy = mode.copy()
            mode_copy.scale(1.0 / np.sqrt(np.abs(C_res)))
            mode_norm.append(mode_copy)

        C_ij = np.zeros ((len(mode_norm), len(mode_norm)), dtype=complex)
        S_ij = np.zeros ((len(mode_norm), len(mode_norm)), dtype=complex)
        for i in range(len(mode_norm)):
            for j in range(len(mode_norm)):
                C_ij[i, j] = model.dot_product(mode_norm[i], mode_norm[j],
                                               *self.overlap_areas)
                S_ij[i, j] = model.dot_product(mode_norm[i], mode_norm[j],
                                               *self.rest_areas)
        print ("mode matrix: ", np.abs(C_ij))
        print ("mode matrix for slab: ", np.abs(S_ij))

        self.mode_norm = mode_norm
        self.C_ij = C_ij
        self.S_ij = S_ij
        #return mode_norm, C_ij, S_ij

    def clusterize_modes(self):
        mode_norm  = self.mode_norm
        C_ij = self.C_ij
        modes_left = list(range(len(mode_norm)))
        clusters = []
        while len(modes_left):
            cluster = []
            first = modes_left.pop(0)
            cluster.append(first)
            modes_left.sort(key = lambda i: -np.abs(C_ij[first, i]))
            while len(modes_left) > 0 and np.abs(C_ij[modes_left[0], first]) > 0.8:
                  cluster.append(modes_left[0])
                  modes_left.pop(0)
            print ("cluster: ", cluster)
            cluster.sort()
            clusters.append(cluster)

        self.clusters = clusters

    def analyze_cluster(self, cluster):
        mode_norm = self.mode_norm
        C_ij = self.C_ij
        S_ij = self.S_ij
        print ("--- Cluster ", cluster)
        for c in cluster:
            print ("   o = ", mode_norm[c].f)
        print ("matrix in cluster:")
        C_c = np.zeros((len(cluster), len(cluster)), dtype=complex)
        S_c = np.zeros((len(cluster), len(cluster)), dtype=complex)
        for i1, c1 in enumerate(cluster):
            for i2, c2 in enumerate(cluster):
                C_c[i1, i2] = C_ij[c1, c2]
                S_c[i1, i2] = S_ij[c1, c2]
        print (" C = ")
        print (C_c)
        print (" S = ")
        print (S_c)
        f_av = 0.0
        part_tot = 0.0
        f_vals = []
        p_vals = []
        for c in cluster:
            mode_c = mode_norm[c]
            part_c = mode_c.participation(*self.overlap_areas)
            f_av += part_c * mode_c.f
            part_tot += part_c
            f_vals.append(mode_c.f)
            p_vals.append(part_c)
        f_av /= part_tot
        p_vals = np.array(p_vals)
        f_vals = np.array(f_vals)
        print ("average f: ", f_av)
        return f_vals, p_vals, C_c, S_c

    def analyze_clusters(self):
        results = []
        for cluster in self.clusters:
            result = self.analyze_cluster(cluster)
            results.append(result)
        return results
