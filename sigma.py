import numpy as np

def Sigma_Gamma(o_vals, o_tab, Gamma_tab, nrf = True):
    print ("compute Sigma_Gamma")
    Sigma_vals = np.zeros((len(o_vals)), dtype=complex)
    for i_o, o in enumerate(o_vals):
        S_o = 0.0  + 0.0j
        for j_o, o_1 in enumerate(o_tab[:-1]):
            o_2 = o_tab[j_o + 1]
            if nrf: 
               F_1 = Gamma_tab[j_o    ] / o_1**2
               F_2 = Gamma_tab[j_o + 1] / o_2**2
            else:
               F_1 = Gamma_tab[j_o]
               F_2 = Gamma_tab[j_o + 1]
            I_log   = np.log(np.abs((o - o_1)/(o - o_2)))
            if (abs(o - o_2) < 1e-6):
                print ("o close to o_2 at: o = ", o,
                       "i_o = ", i_o, "j_o = ", j_o)
            if (abs(o - o_1) < 1e-6):
                print ("o close to o_1 at: o = ", o,
                       "i_o = ", i_o, "j_o = ", j_o)
            I_const = - (o_2 - o_1)
            F_prime = (F_2 - F_1) / (o_2 - o_1)
            F_o = F_1 + (o - o_1) * F_prime
            S_o += F_o * I_log + F_prime * I_const
            #print ("o, o1, S", o / GHz_2pi, o_1 / GHz_2pi, S_o)
        S_o *= 1.0 / np.pi
        if nrf:
            S_o *= o**2
        Sigma_vals[i_o] += S_o
    return Sigma_vals

def Sigma_prime(o_vals, o_tab, Gamma_prime_tab, nrf=True):
    print ("compute Sigma_prime")
    Sigma_prime_vals = np.zeros((len(o_vals)), dtype=complex)
    for i_o, o in enumerate(o_vals):
        S_o = 0.0  + 0.0j
        for j_o, o_1 in enumerate(o_tab[:-1]):
            o_2 = o_tab[j_o + 1]
            if nrf:
               F_1 = Gamma_prime_tab[j_o    ] / o_1**2
               F_2 = Gamma_prime_tab[j_o + 1] / o_2**2
            else:
               F_1 = Gamma_prime_tab[j_o]
               F_2 = Gamma_prime_tab[j_o + 1]
            I_log   = np.log(np.abs((o + o_2)/(o + o_1)))
            if (abs(o + o_2) < 1e-6):
                print ("o close to o_2 at: o = ", o,
                       "i_o = ", i_o, "j_o = ", j_o)
            if (abs(o + o_1) < 1e-6):
                print ("o close to o_1 at: o = ", o,
                       "i_o = ", i_o, "j_o = ", j_o)
            I_const =  (o_2 - o_1)
            F_prime = (F_2 - F_1) / (o_2 - o_1)
            F_o = F_1 - (o + o_1) * F_prime
            S_o += F_o * I_log + F_prime * I_const
            #print ("o, o1, S", o / GHz_2pi, o_1 / GHz_2pi, S_o)
        S_o *= - 1.0 / np.pi
        if nrf:
            S_o *= o**2
        Sigma_prime_vals[i_o] += S_o
    return Sigma_prime_vals

def Sigma_prime_k(o_vals, scattering_problem, kmax, Nk):
    print ("compute Sigma_prime")
    kx = np.linspace(-kmax, kmax, 2 * Nk)
    ky = np.linspace(-kmax, kmax, 2 * Nk)
    KY, KX = np.meshgrid(ky, kx)
    O_K = 0.0 * KX + 0.0j
    D_K = 0.0 * KX + 0.0j
    print ("tabulate on k grid")
    for i in range(len(kx)):
        for j in range(len(ky)):
            O_K[i, j] = scattering_problem.omega_k(kx[i], ky[j])
            D_K[i, j] = scattering_problem.Delta_prime(kx[i], ky[j])
    Sigma_vals = 0.0 * o_vals + 0.0j
    dkx = kx[1] - kx[0]
    dky = ky[1] - ky[0]
    print ("done")
    for i_o, o in enumerate(o_vals):
        I_K = np.abs(D_K)**2 / O_K**2 / (o_vals[i_o] + O_K)
        S_o = np.sum(I_K, axis=(0, 1)) * dkx * dky / (2.0 * np.pi)**2
        Sigma_vals[i_o] += - o**2 * S_o
    return Sigma_vals

def Sigma(o_vals, o_tab, Gamma_tab, Gamma_prime_tab, nrf = True):
    Sigma_tot  =  Sigma_Gamma(o_vals, o_tab, Gamma_tab      , nrf)
    Sigma_tot +=  Sigma_prime(o_vals, o_tab, Gamma_prime_tab, nrf)
    return Sigma_tot

def Sigma_k(o_vals, o_tab, Gamma_tab, scattering_problem, kmax, Nk):
    Sigma_tot  =  Sigma_Gamma(o_vals, o_tab, Gamma_tab) + 0.0j
    Sigma_tot +=  Sigma_prime_k(o_vals, scattering_problem, kmax, Nk)
    return Sigma_tot
    
    
