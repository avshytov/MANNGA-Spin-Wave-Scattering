import numpy as np
import pylab as pl
from scipy import integrate, linalg

from constants import mu_0
import constants

from dispersion import Dispersion

class ScatteringProblem_1D:
    def __init__ (self, slab, resonator, z_res):
        self.resonator = resonator
        self.slab = slab
        self.z_res = z_res
        self.dispersion = Dispersion(self.omega_k)
        self.theta_or = resonator.theta_or
        
    def describe(self):
        result = dict()
        result.update(self.slab.describe())
        result.update(self.resonator.describe())
        result.update(z_res = self.z_res, theta_or = self.theta_or)
        return result
        
    def omega_k(self, kx, ky):
        return self.mode(kx, ky).omega

    def mode(self, kx, ky):
        #kx = k * np.cos(self.theta_or)
        #ky = k * np.sin(self.theta_or)
        mode_plus_E, mode_minus_E = self.slab.make_modes(kx, ky)
        return mode_plus_E[0]
    
    def Delta(self, k):
        kx = k * np.cos(self.theta_or)
        ky = k * np.sin(self.theta_or)
        mode_k = self.mode(kx, ky)
        phi_slab = mode_k.potential(self.z_res)
        K_res = self.resonator.K_1D(k)
        #print ("phi = ", phi_slab, "K_1D = ", K_res)
        return K_res * phi_slab().conj() * mu_0 * 0.25

    def Gamma_rad(self, omega):
        k_R = self.dispersion.k_omega(omega, self.theta_or)
        k_L = self.dispersion.k_omega(omega, self.theta_or + np.pi)
        v_Rx, v_Ry = self.dispersion.v_omega(omega, self.theta_or)
        v_Lx, v_Ly = self.dispersion.v_omega(omega, self.theta_or + np.pi)
        v_R = np.abs(v_Rx + 1j * v_Ry)
        v_L = np.abs(v_Lx + 1j * v_Ly)
        Delta_R = self.Delta( k_R)
        Delta_L = self.Delta(-k_L)
        Gamma_R =  0.5 * np.abs(Delta_R)**2 / v_R
        Gamma_L =  0.5 * np.abs(Delta_L)**2 / v_L
        return Gamma_R + Gamma_L

    def Gamma_tot(self, omega):
        return self.resonator.gamma_0() + self.Gamma_rad(omega)

    def T_and_R(self, omega):
        k_R = self.dispersion.k_omega(omega, self.theta_or)
        k_L = self.dispersion.k_omega(omega, self.theta_or + np.pi)
        v_Rx, v_Ry = self.dispersion.v_omega(omega, self.theta_or)
        v_Lx, v_Ly = self.dispersion.v_omega(omega, self.theta_or + np.pi)
        v_R = np.abs(v_Rx + 1j * v_Ry)
        v_L = np.abs(v_Lx + 1j * v_Ly)
        Delta_R = self.Deltas(k_R)
        Delta_L = self.Deltas(-k_L)
        #Gamma_R = 0.5 / abs(v_R) * np.outer(Delta_R.conjugate(), Delta_R)
        #Gamma_L = 0.5 / abs(v_R) * np.outer(Delta_L.conjugate(), Delta_L)
        Gamma_R =  0.5 * np.abs(Delta_R)**2 / abs(v_R)
        Gamma_L =  0.5 * np.abs(Delta_L)**2 / abs(v_L)
        Gamma_0 =  self.gamma_0()
        Gamma_tot = Gamma_0 + Gamma_L + Gamma_R
        #print ("o = ", omega, "Delta = ", Delta_R, Delta_L)
        omega_denom = omega - self.resonator.omega_0() + 1j * Gamma_tot
        T1 = 1.0 - 2.0j * Gamma_R / omega_denom
        R1 = -1j * Delta_R * Delta_L.conjugate() / np.sqrt(np.abs(v_R * v_L))
        R1 *= 1.0 / omega_denom
        R2 = 1j * Delta_R.conjugate() * Delta_L / np.sqrt(np.abs(v_R * v_L))
        R2 *= 1.0 / omega_denom
        T2 = 1.0 - 2.0j  * Gamma_L / omega_denom 
        return T1, R1, T2, R2

class ScatteringProblem_1D_Multi:
    def __init__ (self, slab, res_modes, z_res, slab_modes=1):
        self.res_modes = res_modes
        self.slab = slab
        self.z_res = z_res
        self.dispersion = Dispersion(self.omega_k)
        self.theta_or = res_modes[0].theta_or
        self.slab_modes = slab_modes
        
    def describe(self):
        result = dict()
        result.update(self.slab.describe())
        #result.update(self.resonator.describe())
        result.update(z_res = self.z_res, theta_or = self.theta_or)
        return result
    
    def omega_k(self, kx, ky):
        return self.mode(kx, ky).omega
    
    def omega_k_n(self, kx, ky, n):
        return self.mode_n(kx, ky, n).omega

    def omega_k1d(self, k):
        return self.omega_k(k * np.cos(self.theta_or),
                            k * np.sin(self.theta_or))
    
    def omega_k1d_n(self, k, n):
        return self.omega_k_n(k * np.cos(self.theta_or),
                              k * np.sin(self.theta_or), n)

    def mode(self, kx, ky):
        #kx = k * np.cos(self.theta_or)
        #ky = k * np.sin(self.theta_or)
        mode_plus_E, mode_minus_E = self.slab.make_modes(kx, ky)
        return mode_plus_E[0]

    def mode_n(self, kx, ky, n):
        mode_plus_E, mode_minus_E = self.slab.make_modes(kx, ky, max(2, n + 1))
        #print ("mode_n: [n]", mode_plus_E[n])
        #print ("mode_n: [n][0]", mode_plus_E[n][0])
        return mode_plus_E[n][0]
    
    def Deltas(self, k, n = 0):
        kx = k * np.cos(self.theta_or)
        ky = k * np.sin(self.theta_or)
        mode_k = self.mode_n(kx, ky, n)
        phi_slab = mode_k.potential(self.z_res)
        K_res = []
        for res_mode in self.res_modes:
            K_res.append(res_mode.K_1D(k))
        K_res = np.array(K_res)
        #print ("phi = ", phi_slab, "K_1D = ", K_res)
        return K_res * phi_slab.conj() * mu_0 * 0.25
    
    def Deltas_dual(self, k, n = 0):
        kx = k * np.cos(self.theta_or)
        ky = k * np.sin(self.theta_or)
        mode_k = self.mode_n(kx, ky, n)
        phi_slab = mode_k.potential(self.z_res)
        K_res = []
        for res_mode in self.res_modes:
            K_res.append(res_mode.K_1D_dual(k))
        K_res = np.array(K_res)
        #print ("phi = ", phi_slab, "K_1D = ", K_res)
        return K_res * phi_slab * mu_0 * 0.25

    def Deltas1(self, k, n = 0):
        kx = k * np.cos(self.theta_or)
        ky = k * np.sin(self.theta_or)
        mode_k = self.mode_n(kx, ky, n)
        phi_slab = mode_k.potential(self.z_res)
        K_res = []
        for res_mode in self.res_modes:
            K_res.append(res_mode.K_1D(-k))
        K_res = np.array(K_res)
        #print ("phi = ", phi_slab, "K_1D = ", K_res)
        return K_res * phi_slab * mu_0 * 0.25
    
    def Deltas1_dual(self, k, n = 0):
        kx = k * np.cos(self.theta_or)
        ky = k * np.sin(self.theta_or)
        mode_k = self.mode_n(kx, ky, n)
        phi_slab = mode_k.potential(self.z_res)
        K_res = []
        for res_mode in self.res_modes:
            K_res.append(res_mode.K_1D_dual(-k))
        K_res = np.array(K_res)
        #print ("phi = ", phi_slab, "K_1D = ", K_res)
        return K_res * phi_slab.conj() * mu_0 * 0.25 

    def Gamma_R_ab(self, omega):
        k_R = self.dispersion.k_omega(omega, self.theta_or)
        v_Rx, v_Ry = self.dispersion.v_omega(omega, self.theta_or)
        v_R = np.abs(v_Rx + 1j * v_Ry)
        Delta_R = self.Deltas( k_R)
        Gamma_R = 0.5 / np.abs(v_R) * np.outer(Delta_R.conjugate(), Delta_R)
        return Gamma_R
    
    def Gamma_L_ab(self, omega):
        k_L = self.dispersion.k_omega(omega, self.theta_or + np.pi)
        v_Lx, v_Ly = self.dispersion.v_omega(omega, self.theta_or + np.pi)
        v_L = np.abs(v_Lx + 1j * v_Ly)
        Delta_L = self.Deltas( -k_L)
        Gamma_L = 0.5 / np.abs(v_L) * np.outer(Delta_L.conjugate(), Delta_L)
        return Gamma_L
    
    def Gamma_rad_ab(self, omega):
        k_R = self.dispersion.k_omega(omega, self.theta_or)
        k_L = self.dispersion.k_omega(omega, self.theta_or + np.pi)
        v_Rx, v_Ry = self.dispersion.v_omega(omega, self.theta_or)
        v_Lx, v_Ly = self.dispersion.v_omega(omega, self.theta_or + np.pi)
        v_R = np.abs(v_Rx + 1j * v_Ry)
        v_L = np.abs(v_Lx + 1j * v_Ly)
        Delta_R = self.Deltas( k_R)
        Delta_L = self.Deltas(-k_L)
        Gamma_R = 0.5 / np.abs(v_R) * np.outer(Delta_R.conjugate(), Delta_R)
        Gamma_L = 0.5 / np.abs(v_L) * np.outer(Delta_L.conjugate(), Delta_L)
        #Gamma_R =  0.5 * np.abs(Delta_R)**2 / v_R
        #Gamma_L =  0.5 * np.abs(Delta_L)**2 / v_L
        return Gamma_R + Gamma_L
    
    def Gamma1_rad_ab(self, omega):
        k_R = self.dispersion.k_omega(omega, self.theta_or)
        k_L = self.dispersion.k_omega(omega, self.theta_or + np.pi)
        v_Rx, v_Ry = self.dispersion.v_omega(omega, self.theta_or)
        v_Lx, v_Ly = self.dispersion.v_omega(omega, self.theta_or + np.pi)
        v_R = np.abs(v_Rx + 1j * v_Ry)
        v_L = np.abs(v_Lx + 1j * v_Ly)
        Delta1_R = self.Deltas1( k_R)
        Delta1_L = self.Deltas1(-k_L)
        Gamma_R = 0.5 / np.abs(v_R) * np.outer(Delta1_R, Delta1_R.conj())
        Gamma_L = 0.5 / np.abs(v_L) * np.outer(Delta1_L, Delta1_L.conj())
        #Gamma_R =  0.5 * np.abs(Delta_R)**2 / v_R
        #Gamma_L =  0.5 * np.abs(Delta_L)**2 / v_L
        return Gamma_R + Gamma_L
    
    def Gamma2_rad_ab(self, omega):
        k_R = self.dispersion.k_omega(omega, self.theta_or)
        k_L = self.dispersion.k_omega(omega, self.theta_or + np.pi)
        v_Rx, v_Ry = self.dispersion.v_omega(omega, self.theta_or)
        v_Lx, v_Ly = self.dispersion.v_omega(omega, self.theta_or + np.pi)
        v_R = np.abs(v_Rx + 1j * v_Ry)
        v_L = np.abs(v_Lx + 1j * v_Ly)
        Delta_R = self.Deltas( k_R)
        Delta_L = self.Deltas(-k_L)
        Delta1_R = self.Deltas1( k_R)
        Delta1_L = self.Deltas1(-k_L)
        Gamma_R = 0.5 / np.abs(v_R) * np.outer(Delta1_R, Delta_R)
        Gamma_L = 0.5 / np.abs(v_L) * np.outer(Delta1_L, Delta_L)
        #Gamma_R =  0.5 * np.abs(Delta_R)**2 / v_R
        #Gamma_L =  0.5 * np.abs(Delta_L)**2 / v_L
        return Gamma_R + Gamma_L

    def eta_0 (self):
        res = []
        for res_mode in self.res_modes:
            omega_0 = res_mode.omega_0()
            gamma_0 = res_mode.gamma_0()
            res.append(omega_0 / (omega_0 + 1j * gamma_0))
        return np.array(res)
    
    def gammas_0(self):
        res = []
        for res_mode in self.res_modes:
            res.append(res_mode.gamma_0())
        return np.diag(np.array(res))
    
    def omegas_0(self):
        res = []
        for res_mode in self.res_modes:
            res.append(res_mode.omega_0())
        return np.diag(np.array(res))
    
    def Gamma_tot(self, omega):
        return self.gammas_0() + self.Gamma_rad(omega)

    def T_and_R(self, omega):
        k_R = self.dispersion.k_omega(omega, self.theta_or)
        k_L = self.dispersion.k_omega(omega, self.theta_or + np.pi)
        v_Rx, v_Ry = self.dispersion.v_omega(omega, self.theta_or)
        v_Lx, v_Ly = self.dispersion.v_omega(omega, self.theta_or + np.pi)
        v_R = np.abs(v_Rx + 1j * v_Ry)
        v_L = np.abs(v_Lx + 1j * v_Ly)
        Delta_R = self.Deltas(k_R)
        Delta_L = self.Deltas(-k_L)
        print ("*** old T, R; o = ", omega / constants.GHz_2pi)
        print ("Delta_R: ", Delta_R)
        print ("Delta_L: ", Delta_L)
        print ("velocities: ", v_R, v_L)
        Gamma_R = 0.5 / abs(v_R) * np.outer(Delta_R.conjugate(), Delta_R)
        Gamma_L = 0.5 / abs(v_L) * np.outer(Delta_L.conjugate(), Delta_L)
        print ("Gammas_R: ", np.diag(Gamma_R).real)
        print ("Gammas_L: ", np.diag(Gamma_L).real)
        #Gamma_R =  0.5 * np.abs(Delta_R)**2 / abs(v_R)
        #Gamma_L =  0.5 * np.abs(Delta_L)**2 / abs(v_L)
        Gamma_0 =  self.gammas_0() 
        print ("Gammas_0: ", np.diag(Gamma_0))
        Gammas_tot = Gamma_0 + Gamma_L + Gamma_R
        #print ("o = ", omega, "Delta = ", Delta_R, Delta_L)
        I = np.eye(len(self.res_modes), dtype=complex)
        omega_denom = omega * I - self.omegas_0() + 1j * Gammas_tot
        omega_d_inv = linalg.inv(omega_denom)
        T1 = 1.0 - 1.0j * np.dot(Delta_R,
                            np.dot(omega_d_inv, Delta_R.conjugate()))/abs(v_R)
        T2 = 1.0 - 1.0j * np.dot(Delta_L,
                            np.dot(omega_d_inv, Delta_L.conjugate()))/abs(v_L)
        #Gamma_R / omega_denom
        #R1 = -1j * Delta_R * Delta_L.conjugate() / np.sqrt(np.abs(v_R * v_L))
        R1 = -1j * np.dot(Delta_L, np.dot(omega_d_inv, Delta_R.conjugate()))
        R1 /=  np.sqrt(np.abs(v_R * v_L))
        R2 = -1j * np.dot(Delta_R, np.dot(omega_d_inv, Delta_L.conjugate()))
        R2 /=  np.sqrt(np.abs(v_R * v_L))
        #R1 *= 1.0 / omega_denom
        #R2 = 1j * Delta_R.conjugate() * Delta_L / np.sqrt(np.abs(v_R * v_L))
        #R2 *= 1.0 / omega_denom
        #T2 = 1.0 - 2.0j  * Gamma_L / omega_denom 
        return T1, R1, T2, R2
    
    def T_and_R_full(self, omegas, k_max, N_k):
        Sigma     = self.Sigma_1D(omegas, k_max, N_k)
        T1 = np.zeros((len(omegas)), dtype=complex)
        T2 = np.zeros((len(omegas)), dtype=complex)
        R1 = np.zeros((len(omegas)), dtype=complex)
        R2 = np.zeros((len(omegas)), dtype=complex)
        for i_o, omega in enumerate(omegas):
            k_R = self.dispersion.k_omega(omega, self.theta_or)
            k_L = self.dispersion.k_omega(omega, self.theta_or + np.pi)
            v_Rx, v_Ry = self.dispersion.v_omega(omega, self.theta_or)
            v_Lx, v_Ly = self.dispersion.v_omega(omega, self.theta_or + np.pi)
            v_R = np.abs(v_Rx + 1j * v_Ry)
            v_L = np.abs(v_Lx + 1j * v_Ly)
            Delta_R = self.Deltas(k_R)
            Delta_L = self.Deltas(-k_L)
            Delta_R1 = self.Deltas1(k_R)
            Delta_L1 = self.Deltas1(-k_L)
            print ("*** new T, R; o = ", omega / constants.GHz_2pi)
            print ("Delta_R: ", Delta_R)
            print ("Delta_L: ", Delta_L)
            print ("velocities: ", v_R, v_L)
            Gamma_R = 0.5 / abs(v_R) * np.outer(Delta_R.conjugate(), Delta_R)
            Gamma_L = 0.5 / abs(v_L) * np.outer(Delta_L.conjugate(), Delta_L)
            print ("Gammas_R: ", np.diag(Gamma_R).real)
            print ("Gammas_L: ", np.diag(Gamma_L).real)
            #Gamma_R =  0.5 * np.abs(Delta_R)**2 / abs(v_R)
            #Gamma_L =  0.5 * np.abs(Delta_L)**2 / abs(v_L)
            Gamma_0 =  self.gammas_0() 
            print ("Gammas_0: ", np.diag(Gamma_0))
            Gammas_tot = Gamma_0 + Gamma_L + Gamma_R
            #print ("o = ", omega, "Delta = ", Delta_R, Delta_L)
            I = np.eye(len(self.res_modes), dtype=complex)
            omega_denom  =   omega * I - self.omegas_0() + 1j * Gammas_tot
            omega_denom += - Sigma[:, :, i_o] 
            omega_d_inv = linalg.inv(omega_denom)
            omega_d_R = np.dot(omega_d_inv, Delta_R.conjugate())
            omega_d_L = np.dot(omega_d_inv, Delta_L.conjugate())
            T1[i_o] = 1.0 - 1.0j * np.dot(Delta_R, omega_d_R) / abs(v_R)
            T2[i_o] = 1.0 - 1.0j * np.dot(Delta_L, omega_d_L) / abs(v_L)
            v_RL = np.sqrt(np.abs(v_R) * np.abs(v_L))
            R1[i_o] = -1j * np.dot(Delta_L, omega_d_R) / v_RL
            R2[i_o] = -1j * np.dot(Delta_R, omega_d_L) / v_RL
            omega_denom1  = omega * I + self.omegas_0() #+ 1j * Gammas_tot
            #omega_denom1 += Sigma[:, :, i_o] 
            omega_d1_inv = linalg.inv(omega_denom1)
            omega_d1_R = np.dot(omega_d1_inv, Delta_R1)
            omega_d1_L = np.dot(omega_d1_inv, Delta_L1)
            T1[i_o] += 1.0j * np.dot(Delta_R1.conj(), omega_d1_R) / abs(v_R)
            T2[i_o] += 1.0j * np.dot(Delta_L1.conj(), omega_d1_L) / abs(v_L)
            R1[i_o] += 1.0j * np.dot(Delta_L1.conj(), omega_d1_R) / v_RL
            R2[i_o] += 1.0j * np.dot(Delta_R1.conj(), omega_d1_L) / v_RL
        return T1, R1, T2, R2
    
    def T_and_R_really_all(self, omegas, k_max, N_k, phi_wanted = False):
        Sigma_all = self.Sigma_1D_all(omegas, k_max, N_k)
        Sigma_11, Sigma_22, Sigma_12, Sigma_21 = Sigma_all
        if False:
           np.savez("sigma-nrf.npz", omegas = omegas,
                 Sigma_11 = Sigma_11, Sigma_22 = Sigma_22,
                 Sigma_12 = Sigma_12, Sigma_21 = Sigma_21)
        #Gamma_11, Gamma_22, Gamma_12, Gamma_22 = Gamma_all
        T1 = np.zeros((len(omegas)), dtype=complex)
        T2 = np.zeros((len(omegas)), dtype=complex)
        R1 = np.zeros((len(omegas)), dtype=complex)
        R2 = np.zeros((len(omegas)), dtype=complex)
        Nmodes = len(self.gammas_0())
        phi_R_o    = np.zeros((len(omegas), Nmodes), dtype=complex)
        phibar_R_o = np.zeros((len(omegas), Nmodes), dtype=complex)
        phi_L_o    = np.zeros((len(omegas), Nmodes), dtype=complex)
        phibar_L_o = np.zeros((len(omegas), Nmodes), dtype=complex)
        eta = self.eta_0()
        eta[:] = 1.0
        for i_o, omega in enumerate(omegas):
            k_R = self.dispersion.k_omega(omega, self.theta_or)
            k_L = self.dispersion.k_omega(omega, self.theta_or + np.pi)
            v_Rx, v_Ry = self.dispersion.v_omega(omega, self.theta_or)
            v_Lx, v_Ly = self.dispersion.v_omega(omega, self.theta_or + np.pi)
            v_R = np.abs(v_Rx + 1j * v_Ry)
            v_L = np.abs(v_Lx + 1j * v_Ly)
            Delta_R = self.Deltas(k_R)
            Delta_L = self.Deltas(-k_L)
            Delta_R_dual = self.Deltas_dual(k_R)
            Delta_L_dual = self.Deltas_dual(-k_L)
            Delta_R1 = self.Deltas1(k_R)
            Delta_L1 = self.Deltas1(-k_L)
            Delta_R1_dual = self.Deltas1_dual(k_R)
            Delta_L1_dual = self.Deltas1_dual(-k_L)
            print ("*** new T, R; o = ", omega / constants.GHz_2pi)
            print ("Delta_R: ", Delta_R)
            print ("Delta_L: ", Delta_L)
            print ("velocities: ", v_R, v_L)
            Gamma_R = 0.5 / abs(v_R) * np.outer(eta * Delta_R_dual, Delta_R)
            Gamma_L = 0.5 / abs(v_L) * np.outer(eta * Delta_L_dual, Delta_L)
            print ("Gammas_R: ", np.diag(Gamma_R).real)
            print ("Gammas_L: ", np.diag(Gamma_L).real)
            #Gamma_R =  0.5 * np.abs(Delta_R)**2 / abs(v_R)
            #Gamma_L =  0.5 * np.abs(Delta_L)**2 / abs(v_L)
            Gamma_0 =  self.gammas_0() 
            print ("Gammas_0: ", np.diag(Gamma_0))
            Gammas_tot = Gamma_0 + Gamma_L + Gamma_R
            Gammas_22_R = 0.5 / abs(v_R) * np.outer(eta.conj() * Delta_R1_dual.conj(),
                                                    Delta_R1.conj())
            Gammas_22_L = 0.5 / abs(v_L) * np.outer(eta.conj() * Delta_L1_dual.conj(),
                                                    Delta_L1.conj())
            Gammas_22 = - Gamma_0 + Gammas_22_R + Gammas_22_L
            Gammas_12_R = 0.5 / abs(v_R) * np.outer(eta * Delta_R_dual,
                                                    Delta_R1.conj())
            Gammas_12_L = 0.5 / abs(v_L) * np.outer(eta * Delta_L_dual,
                                                    Delta_L1.conj())
            Gammas_12 = Gammas_12_R + Gammas_12_L
            Gammas_21_R = 0.5 / abs(v_R) * np.outer(eta.conj() * Delta_R1_dual.conj(),
                                                    Delta_R)
            Gammas_21_L = 0.5 / abs(v_L) * np.outer(eta.conj() * Delta_L1_dual.conj(),
                                                    Delta_L)
            Gammas_21 = Gammas_21_R + Gammas_21_L
            
            #print ("o = ", omega, "Delta = ", Delta_R, Delta_L)
            Nmodes = len(self.res_modes)
            I = np.eye(Nmodes, dtype=complex)
            M = np.zeros ((2 * Nmodes, 2 * Nmodes), dtype=complex)
            M[0::2, 0::2] += omega * I
            M[1::2, 1::2] += omega * I
            M[0::2, 0::2] += - self.omegas_0()
            M[1::2, 1::2] +=   self.omegas_0()
            M[0::2, 0::2] +=  1j * Gammas_tot
            M[1::2, 1::2] += -1j * Gammas_22
            M[0::2, 1::2] +=  1j * Gammas_12
            M[1::2, 0::2] += -1j * Gammas_21
            
            M[0::2, 0::2] += - eta[:, None]        * Sigma_11[:, :, i_o]
            M[1::2, 1::2] +=   eta[:, None].conj() * Sigma_22[:, :, i_o]
            M[0::2, 1::2] += - eta[:, None]        * Sigma_12[:, :, i_o]
            M[1::2, 0::2] +=   eta[:, None].conj() * Sigma_21[:, :, i_o]

            b_R = np.zeros((2 * Nmodes), dtype=complex)
            b_L = np.zeros((2 * Nmodes), dtype=complex)
            b_R[0::2] =   eta        * Delta_R_dual
            b_R[1::2] = - eta.conj() * Delta_R1_dual.conj()
            b_L[0::2] =   eta        * Delta_L_dual
            b_L[1::2] = - eta.conj() * Delta_L1_dual.conj()

            phi_both_R = linalg.solve(M, b_R)
            phi_both_L = linalg.solve(M, b_L)
            phi_R = phi_both_R[0::2]; phibar_R = phi_both_R[1::2]
            phi_L = phi_both_L[0::2]; phibar_L = phi_both_L[1::2]

            psi_RR = np.dot(Delta_R, phi_R) + np.dot(Delta_R1.conj(), phibar_R)
            psi_LR = np.dot(Delta_L, phi_R) + np.dot(Delta_L1.conj(), phibar_R)
            psi_RL = np.dot(Delta_R, phi_L) + np.dot(Delta_R1.conj(), phibar_L)
            psi_LL = np.dot(Delta_L, phi_L) + np.dot(Delta_L1.conj(), phibar_L)
            
            v_RL = np.sqrt(np.abs(v_R) * np.abs(v_L))
            #T1[i_o] = 1.0 - 1.0j * psi_RR / np.abs(v_R)
            T2[i_o] = 1.0 - 1.0j * psi_LL / np.abs(v_L)
            T1[i_o] = 1.0 - 1.0j * psi_RR / np.abs(v_R)

            R1[i_o] = -1j * psi_LR / v_RL
            R2[i_o] = -1j * psi_RL / v_RL
            phi_R_o   [i_o, :] = phi_R
            phibar_R_o[i_o, :] = phibar_R
            phi_L_o   [i_o, :] = phi_L
            phibar_L_o[i_o, :] = phibar_L
            
            print ("really all: ", omega / constants.GHz_2pi,
                   "T=", np.abs(T1[i_o]), np.abs(T2[i_o]),
                   "R=", np.abs(R1[i_o]), np.abs(R2[i_o]))

        if phi_wanted:
            return T1, R1, T2, R2, phi_R_o, phibar_R_o, phi_L_o, phibar_L_o
        return T1, R1, T2, R2
             

    def T_and_R_really_all_bare(self, omegas, k_max, N_k, phi_wanted = False):
        Sigma_all = self.Sigma_1D_all_bare(omegas, k_max, N_k)
        Sigma_11, Sigma_22, Sigma_12, Sigma_21 = Sigma_all
        if False:
           np.savez("sigma-bare.npz", omegas = omegas,
                 Sigma_11 = Sigma_11, Sigma_22 = Sigma_22,
                 Sigma_12 = Sigma_12, Sigma_21 = Sigma_21)
        #Gamma_11, Gamma_22, Gamma_12, Gamma_22 = Gamma_all
        T1 = np.zeros((len(omegas)), dtype=complex)
        T2 = np.zeros((len(omegas)), dtype=complex)
        R1 = np.zeros((len(omegas)), dtype=complex)
        R2 = np.zeros((len(omegas)), dtype=complex)
        Nmodes = len(self.gammas_0())
        phi_R_o    = np.zeros((len(omegas), Nmodes), dtype=complex)
        phibar_R_o = np.zeros((len(omegas), Nmodes), dtype=complex)
        phi_L_o    = np.zeros((len(omegas), Nmodes), dtype=complex)
        phibar_L_o = np.zeros((len(omegas), Nmodes), dtype=complex)
        
        eta = self.eta_0()
        eta[:] = 1.0
        Gamma_0 =  self.gammas_0()
        for i_o, omega in enumerate(omegas):
            k_R = self.dispersion.k_omega(omega, self.theta_or)
            k_L = self.dispersion.k_omega(omega, self.theta_or + np.pi)
            v_Rx, v_Ry = self.dispersion.v_omega(omega, self.theta_or)
            v_Lx, v_Ly = self.dispersion.v_omega(omega, self.theta_or + np.pi)
            v_R = np.abs(v_Rx + 1j * v_Ry)
            v_L = np.abs(v_Lx + 1j * v_Ly)
            Delta_R = self.Deltas(k_R)
            Delta_L = self.Deltas(-k_L)
            Delta_R_dual = self.Deltas_dual(k_R)
            Delta_L_dual = self.Deltas_dual(-k_L)
            Delta_R1 = self.Deltas1(k_R)
            Delta_L1 = self.Deltas1(-k_L)
            Delta_R1_dual = self.Deltas1_dual(k_R)
            Delta_L1_dual = self.Deltas1_dual(-k_L)
            print ("*** new T, R; o = ", omega / constants.GHz_2pi)
            print ("Delta_R: ", Delta_R)
            print ("Delta_L: ", Delta_L)
            print ("velocities: ", v_R, v_L)
            Gamma_R = 0.5 / abs(v_R) * np.outer(eta * Delta_R_dual, Delta_R)
            Gamma_L = 0.5 / abs(v_L) * np.outer(eta * Delta_L_dual, Delta_L)
            print ("Gammas_R: ", np.diag(Gamma_R).real)
            print ("Gammas_L: ", np.diag(Gamma_L).real)
            #Gamma_R =  0.5 * np.abs(Delta_R)**2 / abs(v_R)
            #Gamma_L =  0.5 * np.abs(Delta_L)**2 / abs(v_L)
            print ("Gammas_0: ", np.diag(Gamma_0))
            Gammas_tot = Gamma_0 + Gamma_L + Gamma_R
            Gammas_22_R = 0.5 / abs(v_R) * np.outer(eta.conj () * Delta_R1_dual.conj(),
                                                    Delta_R1.conj())
            Gammas_22_L = 0.5 / abs(v_L) * np.outer(eta.conj() * Delta_L1_dual.conj(),
                                                    Delta_L1.conj())
            Gammas_22 = - Gamma_0 + Gammas_22_R + Gammas_22_L
            Gammas_12_R = 0.5 / abs(v_R) * np.outer(eta * Delta_R_dual,
                                                    Delta_R1.conj())
            Gammas_12_L = 0.5 / abs(v_L) * np.outer(eta * Delta_L_dual,
                                                    Delta_L1.conj())
            Gammas_12 = Gammas_12_R + Gammas_12_L
            Gammas_21_R = 0.5 / abs(v_R) * np.outer(eta.conj() * Delta_R1_dual.conj(),
                                                    Delta_R)
            Gammas_21_L = 0.5 / abs(v_L) * np.outer(eta.conj() * Delta_L1_dual.conj(),
                                                    Delta_L)
            Gammas_21 = Gammas_21_R + Gammas_21_L
            
            #print ("o = ", omega, "Delta = ", Delta_R, Delta_L)
            Nmodes = len(self.res_modes)
            I = np.eye(Nmodes, dtype=complex)
            M = np.zeros ((2 * Nmodes, 2 * Nmodes), dtype=complex)
            M[0::2, 0::2] += omega * I
            M[1::2, 1::2] += omega * I
            M[0::2, 0::2] += - self.omegas_0()
            M[1::2, 1::2] +=   self.omegas_0()
            M[0::2, 0::2] +=  1j * Gammas_tot
            M[1::2, 1::2] += -1j * Gammas_22
            M[0::2, 1::2] +=  1j * Gammas_12
            M[1::2, 0::2] += -1j * Gammas_21
            
            M[0::2, 0::2] += - eta[:, None]        * Sigma_11[:, :, i_o]
            M[1::2, 1::2] +=   eta[:, None].conj() * Sigma_22[:, :, i_o]
            M[0::2, 1::2] += - eta[:, None]        * Sigma_12[:, :, i_o]
            M[1::2, 0::2] +=   eta[:, None].conj() * Sigma_21[:, :, i_o]

            b_R = np.zeros((2 * Nmodes), dtype=complex)
            b_L = np.zeros((2 * Nmodes), dtype=complex)
            b_R[0::2] =   eta        * Delta_R_dual
            b_R[1::2] = - eta.conj() * Delta_R1_dual.conj()
            b_L[0::2] =   eta        * Delta_L_dual
            b_L[1::2] = - eta.conj() * Delta_L1_dual.conj()

            phi_both_R = linalg.solve(M, b_R)
            phi_both_L = linalg.solve(M, b_L)
            phi_R = phi_both_R[0::2]; phibar_R = phi_both_R[1::2]
            phi_L = phi_both_L[0::2]; phibar_L = phi_both_L[1::2]

            psi_RR = np.dot(Delta_R, phi_R) + np.dot(Delta_R1.conj(), phibar_R)
            psi_LR = np.dot(Delta_L, phi_R) + np.dot(Delta_L1.conj(), phibar_R)
            psi_RL = np.dot(Delta_R, phi_L) + np.dot(Delta_R1.conj(), phibar_L)
            psi_LL = np.dot(Delta_L, phi_L) + np.dot(Delta_L1.conj(), phibar_L)
            
            v_RL = np.sqrt(np.abs(v_R) * np.abs(v_L))
            #T1[i_o] = 1.0 - 1.0j * psi_RR / np.abs(v_R)
            T2[i_o] = 1.0 - 1.0j * psi_LL / np.abs(v_L)
            T1[i_o] = 1.0 - 1.0j * psi_RR / np.abs(v_R)

            R1[i_o] = -1j * psi_LR / v_RL
            R2[i_o] = -1j * psi_RL / v_RL
            phi_R_o   [i_o, :] = phi_R
            phibar_R_o[i_o, :] = phibar_R
            phi_L_o   [i_o, :] = phi_L
            phibar_L_o[i_o, :] = phibar_L
            
            print ("really all: ", omega / constants.GHz_2pi,
                   "T=", np.abs(T1[i_o]), np.abs(T2[i_o]),
                   "R=", np.abs(R1[i_o]), np.abs(R2[i_o]))

        if phi_wanted:
            return T1, R1, T2, R2, phi_R_o, phibar_R_o, phi_L_o, phibar_L_o
        return T1, R1, T2, R2
             

    def Sigma_1D(self, o_vals, k_max, Nk):
        k_min = 1e-3
        omega_min = self.omega_k(np.cos(self.theta_or) * k_min,
                                 np.sin(self.theta_or) * k_min).real
        omega_max = self.omega_k(np.cos(self.theta_or) * k_max,
                                 np.sin(self.theta_or) * k_max).real
        o_tab = np.linspace(omega_min, omega_max, Nk)
        Nmodes = len(self.res_modes)

        Gamma_ab_tab = np.zeros ((Nmodes, Nmodes, len(o_tab)), dtype=complex)
        from constants import GHz_2pi
        print ("o_min = ", omega_min / GHz_2pi, "max: ", omega_max/GHz_2pi)
        for i_o, o in enumerate(o_tab):
            print ("o = ", o / GHz_2pi)
            Gamma_ab_tab[:, :, i_o] = self.Gamma_rad_ab(o)
            

        Sigma_vals = np.zeros ((Nmodes, Nmodes, len(o_vals)), dtype=complex)
        for i_o, o in enumerate(o_vals):
            S_o = np.zeros ((Nmodes, Nmodes), dtype=complex)
            for j_o, o_1 in enumerate(o_tab[:-1]):
                o_2 = o_tab[j_o + 1]
                F_1 = Gamma_ab_tab[:, :, j_o    ] / o_1**2
                F_2 = Gamma_ab_tab[:, :, j_o + 1] / o_2**2
                I_log   = np.log(np.abs((o - o_1)/(o - o_2)))
                I_const = - (o_2 - o_1)
                F_prime = (F_2 - F_1) / (o_2 - o_1)
                F_o = F_1 + (o - o_1) * F_prime
                S_o += F_o * I_log + F_prime * I_const
                #print ("o, o1, S", o / GHz_2pi, o_1 / GHz_2pi, S_o)
            S_o *= o**2 / np.pi
            Sigma_vals[:, :, i_o] += S_o
        #continue
        k_vals = np.linspace(-k_max, k_max, int(2 * int(Nk//2) + 1))
        k_mid = 0.5 * (k_vals[1:] + k_vals[:-1])
        dk = k_vals[1:] - k_vals[:-1]
        for i_k, k in enumerate(k_mid):
            Deltas1 = self.Deltas1(k)
            o_k = self.omega_k1d(k).real
            DD = np.outer(Deltas1.conj(), Deltas1)
            f_o = - o_vals**2 / (o_vals + o_k) / o_k**2
            dSigma = np.zeros ((Nmodes, Nmodes, len(o_vals)), dtype=complex)
            for i_m in range(Nmodes):
                for j_m in range(Nmodes):
                    dSigma[i_m, j_m] += DD[i_m, j_m] * f_o
            dSigma *= dk[i_k] / 2.0 / np.pi
            #dSigma = np.outer(DD,  f_o) * dk[i_k] / 2.0 / np.pi
            Sigma_vals += dSigma
            #Sigma_vals += np.abs(DD) * o_vals**2 /  (o + o_k) * dk / 2.0 / np.pi
        return Sigma_vals
    
    def Sigma_1D_all(self, o_vals, k_max, Nk):
        k_min = 1e-3
        omega_min = self.omega_k(np.cos(self.theta_or) * k_min,
                                 np.sin(self.theta_or) * k_min).real
        omega_max = self.omega_k(np.cos(self.theta_or) * k_max,
                                 np.sin(self.theta_or) * k_max).real
        o_tab = np.linspace(omega_min, omega_max, Nk)
        Nmodes = len(self.res_modes)

        Gamma_11_ab = np.zeros ((Nmodes, Nmodes, len(o_tab)), dtype=complex)
        Gamma_22_ab = np.zeros ((Nmodes, Nmodes, len(o_tab)), dtype=complex)
        Gamma_12_ab = np.zeros ((Nmodes, Nmodes, len(o_tab)), dtype=complex)
        Gamma_21_ab = np.zeros ((Nmodes, Nmodes, len(o_tab)), dtype=complex)
        from constants import GHz_2pi
        print ("o_min = ", omega_min / GHz_2pi, "max: ", omega_max/GHz_2pi)
        for i_o, o in enumerate(o_tab):
            print ("o = ", o / GHz_2pi)
            k_R = self.dispersion.k_omega(o, self.theta_or)
            k_L = self.dispersion.k_omega(o, self.theta_or + np.pi)
            v_Rx, v_Ry = self.dispersion.v_omega(o, self.theta_or)
            v_Lx, v_Ly = self.dispersion.v_omega(o, self.theta_or + np.pi)
            v_R = np.abs(v_Rx + 1j * v_Ry)
            v_L = np.abs(v_Lx + 1j * v_Ly)
            Delta_R  = self.Deltas( k_R)
            Delta_L  = self.Deltas(-k_L)
            Delta1_R = self.Deltas1( k_R)
            Delta1_L = self.Deltas1(-k_L)
            Delta_R_dual  = self.Deltas_dual( k_R)
            Delta_L_dual  = self.Deltas_dual(-k_L)
            Delta1_R_dual = self.Deltas1_dual( k_R)
            Delta1_L_dual = self.Deltas1_dual(-k_L)
            vR_inv = 0.5 / np.abs(v_R)
            vL_inv = 0.5 / np.abs(v_L)
            Gamma_11_ab[:, :, i_o]  = vR_inv * np.outer(Delta_R_dual ,
                                                        Delta_R)  
            Gamma_11_ab[:, :, i_o] += vL_inv * np.outer(Delta_L_dual ,
                                                        Delta_L)  
            Gamma_22_ab[:, :, i_o]  = vR_inv * np.outer(Delta1_R_dual.conj(),
                                                        Delta1_R.conj())  
            Gamma_22_ab[:, :, i_o] += vL_inv * np.outer(Delta1_L_dual.conj(),
                                                        Delta1_L.conj())
            Gamma_12_ab[:, :, i_o]  = vR_inv * np.outer(Delta_R_dual,
                                                        Delta1_R.conj()) 
            Gamma_12_ab[:, :, i_o] += vL_inv * np.outer(Delta_L_dual,
                                                        Delta1_L.conj())
            Gamma_21_ab[:, :, i_o]  = vR_inv * np.outer(Delta1_R_dual.conj(),
                                                        Delta_R) 
            Gamma_21_ab[:, :, i_o] += vL_inv * np.outer(Delta1_L_dual.conj(),
                                                        Delta_L)
            #self.Gamma_rad_ab(o)
            #Gamma_22_ab[:, :, i_o] = self.Gamma1_rad_ab(o)
            #Gamma_21_ab[:, :, i_o] = self.Gamma2_rad_ab(o)
            #G_21 = Gamma_21_ab[:, :, i_o]
            #Gamma_12_ab[:, :, i_o] = np.transpose(G_21.conj())

        if False:
           np.savez("Gamma_for_sigma-nrf.npz",
                 o_tab = o_tab,
                 Gamma_11_ab = Gamma_11_ab, Gamma_22_ab = Gamma_22_ab,
                 Gamma_21_ab = Gamma_21_ab, Gamma_12_ab = Gamma_12_ab)
        Sigma_11 = np.zeros ((Nmodes, Nmodes, len(o_vals)), dtype=complex)
        Sigma_22 = np.zeros ((Nmodes, Nmodes, len(o_vals)), dtype=complex)
        Sigma_12 = np.zeros ((Nmodes, Nmodes, len(o_vals)), dtype=complex)
        Sigma_21 = np.zeros ((Nmodes, Nmodes, len(o_vals)), dtype=complex)
        def int_vp(F_1, F_2, o_1, o_2, o):
            I_log   = np.log(np.abs((o - o_1)/(o - o_2)))
            I_const = - (o_2 - o_1)
            F_prime = (F_2 - F_1) / (o_2 - o_1)
            F_o = F_1 + (o - o_1) * F_prime
            return F_o * I_log + F_prime * I_const
        for i_o, o in enumerate(o_vals):
            S11_o = np.zeros ((Nmodes, Nmodes), dtype=complex)
            S22_o = np.zeros ((Nmodes, Nmodes), dtype=complex)
            S12_o = np.zeros ((Nmodes, Nmodes), dtype=complex)
            S21_o = np.zeros ((Nmodes, Nmodes), dtype=complex)
            for j_o, o_1 in enumerate(o_tab[:-1]):
                o_2 = o_tab[j_o + 1]
                F11_1 = Gamma_11_ab[:, :, j_o    ] / o_1**2
                F11_2 = Gamma_11_ab[:, :, j_o + 1] / o_2**2
                F22_1 = Gamma_22_ab[:, :, j_o    ] / o_1**2
                F22_2 = Gamma_22_ab[:, :, j_o + 1] / o_2**2
                F12_1 = Gamma_12_ab[:, :, j_o    ] / o_1**2
                F12_2 = Gamma_12_ab[:, :, j_o + 1] / o_2**2
                F21_1 = Gamma_21_ab[:, :, j_o    ] / o_1**2
                F21_2 = Gamma_21_ab[:, :, j_o + 1] / o_2**2
                S11_o += int_vp(F11_1, F11_2, o_1, o_2, o)
                S22_o += int_vp(F22_1, F22_2, o_1, o_2, o)
                S12_o += int_vp(F12_1, F12_2, o_1, o_2, o)
                S21_o += int_vp(F21_1, F21_2, o_1, o_2, o)            
            S11_o *= o**2 / np.pi
            S22_o *= o**2 / np.pi
            S12_o *= o**2 / np.pi
            S21_o *= o**2 / np.pi
            Sigma_11[:, :, i_o]  += S11_o
            Sigma_22[:, :, i_o]  += S22_o
            Sigma_12[:, :, i_o]  += S12_o
            Sigma_21[:, :, i_o]  += S21_o
            print ("tabulated+: o = ", o / GHz_2pi)
        #continue
        k_vals = np.linspace(-k_max, k_max, int(2 * int(Nk//2) + 1))
        k_mid = 0.5 * (k_vals[1:] + k_vals[:-1])
        dk = k_vals[1:] - k_vals[:-1]
        print ("k_vals: max = ", k_max, "dk = ", k_vals[1] - k_vals[0],
               "min: ", np.min(np.abs(k_vals)))
        print ("tabulate -")
        for n_slab in range(self.slab_modes):
         print ("include slab mode ", n_slab)
         for i_k, k in enumerate(k_mid):
            Deltas1 = self.Deltas1(k, n_slab)
            Deltas  = self.Deltas(k, n_slab)
            Deltas1_dual = self.Deltas1_dual(k, n_slab)
            Deltas_dual  = self.Deltas_dual(k, n_slab)
            o_k = self.omega_k1d_n(k, n_slab)
            DDm_11  = - np.outer(Deltas1_dual,        Deltas1)
            DDm_22  = - np.outer(Deltas_dual.conj(),  Deltas.conj())
            DDm_12  = - np.outer(Deltas1_dual,        Deltas.conj())
            DDm_21  = - np.outer(Deltas_dual.conj(),  Deltas1)
            #DD2 = np.outer(Deltas1, Deltas)
            fm_o = o_vals**2 / (o_vals + o_k) / o_k**2
            dSigma_11 = np.zeros ((Nmodes, Nmodes, len(o_vals)), dtype=complex)
            dSigma_22 = np.zeros ((Nmodes, Nmodes, len(o_vals)), dtype=complex)
            dSigma_12 = np.zeros ((Nmodes, Nmodes, len(o_vals)), dtype=complex)
            dSigma_21 = np.zeros ((Nmodes, Nmodes, len(o_vals)), dtype=complex)
            for i_m in range(Nmodes):
                for j_m in range(Nmodes):
                    dSigma_11[i_m, j_m, :]  += DDm_11 [i_m, j_m] * fm_o
                    dSigma_22[i_m, j_m, :]  += DDm_22 [i_m, j_m] * fm_o
                    dSigma_12[i_m, j_m, :]  += DDm_12 [i_m, j_m] * fm_o
                    dSigma_21[i_m, j_m, :]  += DDm_21 [i_m, j_m] * fm_o
            if n_slab > 0:
               # contribution of psi, only for higher-order film modes
               # the n = 0 contribution was handled before as principal value
               # integral
               DDp_11  = np.outer(Deltas_dual,             Deltas)
               DDp_22  = np.outer(Deltas1_dual.conj(),     Deltas1.conj())
               DDp_12  = np.outer(Deltas_dual,             Deltas1.conj())
               DDp_21  = np.outer(Deltas1_dual.conj(),     Deltas)
               fp_o = 1.0 / (o_vals - o_k) * o_vals**2 / o_k**2
               for i_m in range(Nmodes):
                  for j_m in range(Nmodes):
                      dSigma_11[i_m, j_m, :]  += DDp_11 [i_m, j_m] * fp_o
                      dSigma_22[i_m, j_m, :]  += DDp_22 [i_m, j_m] * fp_o
                      dSigma_12[i_m, j_m, :]  += DDp_12 [i_m, j_m] * fp_o
                      dSigma_21[i_m, j_m, :]  += DDp_21 [i_m, j_m] * fp_o
               
            dSigma_11  *= dk[i_k] / 2.0 / np.pi
            dSigma_22  *= dk[i_k] / 2.0 / np.pi
            dSigma_12  *= dk[i_k] / 2.0 / np.pi
            dSigma_21  *= dk[i_k] / 2.0 / np.pi
            Sigma_11   += dSigma_11
            Sigma_22   += dSigma_22
            Sigma_12   += dSigma_12
            Sigma_21   += dSigma_21
            
        Sigmas = Sigma_11, Sigma_22, Sigma_12, Sigma_21
        if  False:
            pl.figure()
            pl.plot(o_vals / constants.GHz_2pi, Sigma_11[0, 0, :],
                    label=r'Re $\Sigma_{11}(\omega)$')
            pl.plot(o_vals / constants.GHz_2pi, Sigma_12[0, 0, :],
                    label=r'Re $\Sigma_{12}(\omega)$')
            pl.plot(o_vals / constants.GHz_2pi, Sigma_21[0, 0, :],
                    label=r'Re $\Sigma_{21}(\omega)$')
            pl.plot(o_vals / constants.GHz_2pi, Sigma_22[0, 0, :],
                    label=r'Re $\Sigma_{22}(\omega)$')
            pl.xlabel(r"Frequency $\omega / 2\pi$, GHz")
            pl.legend()
            pl.show()
        return Sigmas

    def Sigma_1D_all_bare(self, o_vals, k_max, Nk):
        k_min = 1e-3
        omega_min = self.omega_k(np.cos(self.theta_or) * k_min,
                                 np.sin(self.theta_or) * k_min).real
        omega_max = self.omega_k(np.cos(self.theta_or) * k_max,
                                 np.sin(self.theta_or) * k_max).real
        o_tab = np.linspace(omega_min, omega_max, Nk)
        Nmodes = len(self.res_modes)

        Gamma_11_ab = np.zeros ((Nmodes, Nmodes, len(o_tab)), dtype=complex)
        Gamma_22_ab = np.zeros ((Nmodes, Nmodes, len(o_tab)), dtype=complex)
        Gamma_12_ab = np.zeros ((Nmodes, Nmodes, len(o_tab)), dtype=complex)
        Gamma_21_ab = np.zeros ((Nmodes, Nmodes, len(o_tab)), dtype=complex)
        from constants import GHz_2pi
        print ("o_min = ", omega_min / GHz_2pi, "max: ", omega_max/GHz_2pi)
        for i_o, o in enumerate(o_tab):
            print ("o = ", o / GHz_2pi)
            k_R = self.dispersion.k_omega(o, self.theta_or)
            k_L = self.dispersion.k_omega(o, self.theta_or + np.pi)
            v_Rx, v_Ry = self.dispersion.v_omega(o, self.theta_or)
            v_Lx, v_Ly = self.dispersion.v_omega(o, self.theta_or + np.pi)
            v_R = np.abs(v_Rx + 1j * v_Ry)
            v_L = np.abs(v_Lx + 1j * v_Ly)
            Delta_R  = self.Deltas( k_R)
            Delta_L  = self.Deltas(-k_L)
            Delta1_R = self.Deltas1( k_R)
            Delta1_L = self.Deltas1(-k_L)
            Delta_R_dual  = self.Deltas_dual( k_R)
            Delta_L_dual  = self.Deltas_dual(-k_L)
            Delta1_R_dual = self.Deltas1_dual( k_R)
            Delta1_L_dual = self.Deltas1_dual(-k_L)
            vR_inv = 0.5 / np.abs(v_R)
            vL_inv = 0.5 / np.abs(v_L)
            Gamma_11_ab[:, :, i_o]  = vR_inv * np.outer(Delta_R_dual,
                                                        Delta_R)  
            Gamma_11_ab[:, :, i_o] += vL_inv * np.outer(Delta_L_dual,
                                                        Delta_L)  
            Gamma_22_ab[:, :, i_o]  = vR_inv * np.outer(Delta1_R_dual.conj(),
                                                        Delta1_R.conj())  
            Gamma_22_ab[:, :, i_o] += vL_inv * np.outer(Delta1_L_dual.conj(),
                                                        Delta1_L.conj())
            Gamma_12_ab[:, :, i_o]  = vR_inv * np.outer(Delta_R_dual,
                                                        Delta1_R.conj()) 
            Gamma_12_ab[:, :, i_o] += vL_inv * np.outer(Delta_L_dual,
                                                        Delta1_L.conj())
            Gamma_21_ab[:, :, i_o]  = vR_inv * np.outer(Delta1_R_dual.conj(),
                                                        Delta_R) 
            Gamma_21_ab[:, :, i_o] += vL_inv * np.outer(Delta1_L,
                                                        Delta_L)
            #self.Gamma_rad_ab(o)
            #Gamma_22_ab[:, :, i_o] = self.Gamma1_rad_ab(o)
            #Gamma_21_ab[:, :, i_o] = self.Gamma2_rad_ab(o)
            #G_21 = Gamma_21_ab[:, :, i_o]
            #Gamma_12_ab[:, :, i_o] = np.transpose(G_21.conj())

        if False:
           np.savez("Gamma_for_sigma-bare.npz",
                 o_tab = o_tab,
                 Gamma_11_ab = Gamma_11_ab, Gamma_22_ab = Gamma_22_ab,
                 Gamma_21_ab = Gamma_21_ab, Gamma_12_ab = Gamma_12_ab)
        Sigma_11 = np.zeros ((Nmodes, Nmodes, len(o_vals)), dtype=complex)
        Sigma_22 = np.zeros ((Nmodes, Nmodes, len(o_vals)), dtype=complex)
        Sigma_12 = np.zeros ((Nmodes, Nmodes, len(o_vals)), dtype=complex)
        Sigma_21 = np.zeros ((Nmodes, Nmodes, len(o_vals)), dtype=complex)
        def int_vp(F_1, F_2, o_1, o_2, o):
            I_log   = np.log(np.abs((o - o_1)/(o - o_2)))
            I_const = - (o_2 - o_1)
            F_prime = (F_2 - F_1) / (o_2 - o_1)
            F_o = F_1 + (o - o_1) * F_prime
            return F_o * I_log + F_prime * I_const
        for i_o, o in enumerate(o_vals):
            S11_o = np.zeros ((Nmodes, Nmodes), dtype=complex)
            S22_o = np.zeros ((Nmodes, Nmodes), dtype=complex)
            S12_o = np.zeros ((Nmodes, Nmodes), dtype=complex)
            S21_o = np.zeros ((Nmodes, Nmodes), dtype=complex)
            for j_o, o_1 in enumerate(o_tab[:-1]):
                o_2 = o_tab[j_o + 1]
                F11_1 = Gamma_11_ab[:, :, j_o    ] #/ o_1**2
                F11_2 = Gamma_11_ab[:, :, j_o + 1] #/ o_2**2
                F22_1 = Gamma_22_ab[:, :, j_o    ] #/ o_1**2
                F22_2 = Gamma_22_ab[:, :, j_o + 1] #/ o_2**2
                F12_1 = Gamma_12_ab[:, :, j_o    ] #/ o_1**2
                F12_2 = Gamma_12_ab[:, :, j_o + 1] #/ o_2**2
                F21_1 = Gamma_21_ab[:, :, j_o    ] #/ o_1**2
                F21_2 = Gamma_21_ab[:, :, j_o + 1] #/ o_2**2
                S11_o += int_vp(F11_1, F11_2, o_1, o_2, o)
                S22_o += int_vp(F22_1, F22_2, o_1, o_2, o)
                S12_o += int_vp(F12_1, F12_2, o_1, o_2, o)
                S21_o += int_vp(F21_1, F21_2, o_1, o_2, o)            
            S11_o *= 1.0 / np.pi
            S22_o *= 1.0 / np.pi
            S12_o *= 1.0 / np.pi
            S21_o *= 1.0 / np.pi
            Sigma_11[:, :, i_o]  += S11_o
            Sigma_22[:, :, i_o]  += S22_o
            Sigma_12[:, :, i_o]  += S12_o
            Sigma_21[:, :, i_o]  += S21_o
            print ("tabulated+: o = ", o / GHz_2pi)
        #continue
        k_vals = np.linspace(-k_max, k_max, int(2 * int(Nk//2) + 1))
        k_mid = 0.5 * (k_vals[1:] + k_vals[:-1])
        dk = k_vals[1:] - k_vals[:-1]
        print ("k_vals: max = ", k_max, "dk = ", k_vals[1] - k_vals[0],
               "min: ", np.min(np.abs(k_vals)))
        print ("tabulate non-resonant terms")
        for n_slab in range(self.slab_modes):
          print ("include slab mode", n_slab)
          for i_k, k in enumerate(k_mid):
            Deltas1 = self.Deltas1(k, n_slab)
            Deltas  = self.Deltas(k, n_slab)
            Deltas1_dual = self.Deltas1_dual(k, n_slab)
            Deltas_dual  = self.Deltas_dual(k, n_slab)
            o_k = self.omega_k1d_n(k, n_slab)
            DDm_11  = - np.outer(Deltas1_dual,             Deltas1)
            DDm_22  = - np.outer(Deltas_dual.conj(),       Deltas.conj())
            DDm_12  = - np.outer(Deltas1_dual,             Deltas.conj())
            DDm_21  = - np.outer(Deltas_dual.conj(),       Deltas1)
            #DD2 = np.outer(Deltas1, Deltas)
            fm_o = 1.0 / (o_vals + o_k)
            dSigma_11 = np.zeros ((Nmodes, Nmodes, len(o_vals)), dtype=complex)
            dSigma_22 = np.zeros ((Nmodes, Nmodes, len(o_vals)), dtype=complex)
            dSigma_12 = np.zeros ((Nmodes, Nmodes, len(o_vals)), dtype=complex)
            dSigma_21 = np.zeros ((Nmodes, Nmodes, len(o_vals)), dtype=complex)
            # contribution of psi_bar
            for i_m in range(Nmodes):
                for j_m in range(Nmodes):
                    dSigma_11[i_m, j_m, :]  += DDm_11 [i_m, j_m] * fm_o
                    dSigma_22[i_m, j_m, :]  += DDm_22 [i_m, j_m] * fm_o
                    dSigma_12[i_m, j_m, :]  += DDm_12 [i_m, j_m] * fm_o
                    dSigma_21[i_m, j_m, :]  += DDm_21 [i_m, j_m] * fm_o
            if n_slab > 0:
               # contribution of psi, only for higher-order film modes
               # the n = 0 contribution was handled before as principal value
               # integral
               DDp_11  = np.outer(Deltas_dual,             Deltas)
               DDp_22  = np.outer(Deltas1_dual.conj(),     Deltas1.conj())
               DDp_12  = np.outer(Deltas_dual,             Deltas1.conj())
               DDp_21  = np.outer(Deltas1_dual.conj(),     Deltas)
               fp_o = 1.0 / (o_vals - o_k)
               for i_m in range(Nmodes):
                  for j_m in range(Nmodes):
                    dSigma_11[i_m, j_m, :]  += DDp_11 [i_m, j_m] * fp_o
                    dSigma_22[i_m, j_m, :]  += DDp_22 [i_m, j_m] * fp_o
                    dSigma_12[i_m, j_m, :]  += DDp_12 [i_m, j_m] * fp_o
                    dSigma_21[i_m, j_m, :]  += DDp_21 [i_m, j_m] * fp_o
                
            dSigma_11  *= dk[i_k] / 2.0 / np.pi
            dSigma_22  *= dk[i_k] / 2.0 / np.pi
            dSigma_12  *= dk[i_k] / 2.0 / np.pi
            dSigma_21  *= dk[i_k] / 2.0 / np.pi
            Sigma_11   += dSigma_11
            Sigma_22   += dSigma_22
            Sigma_12   += dSigma_12
            Sigma_21   += dSigma_21
            
        Sigmas = Sigma_11, Sigma_22, Sigma_12, Sigma_21
        if  False:
            pl.figure()
            pl.plot(o_vals / constants.GHz_2pi, Sigma_11[0, 0, :],
                    label=r'Re $\Sigma_{11}(\omega)$')
            pl.plot(o_vals / constants.GHz_2pi, Sigma_12[0, 0, :],
                    label=r'Re $\Sigma_{12}(\omega)$')
            pl.plot(o_vals / constants.GHz_2pi, Sigma_21[0, 0, :],
                    label=r'Re $\Sigma_{21}(\omega)$')
            pl.plot(o_vals / constants.GHz_2pi, Sigma_22[0, 0, :],
                    label=r'Re $\Sigma_{22}(\omega)$')
            pl.xlabel(r"Frequency $\omega / 2\pi$, GHz")
            pl.legend()
            pl.show()
        return Sigmas
    

        
class ScatteringProblem:
    def __init__ (self, slab, resonator, z_res):

        self.slab = slab
        self.resonator = resonator
        self.z_res = z_res
        self.dispersion = Dispersion(self.omega_k)


    def describe(self):
        result = dict()
        result.update(self.slab.describe())
        result.update(self.resonator.describe())
        result.update(z_res = self.z_res)
        return result
        
    def omega_k(self, kx, ky):
        return self.mode(kx, ky).omega

    def G(self, omega, kx, ky):
        return self.dispersion.G(omega, kx, ky)
    
    def mode(self, kx, ky):
        mode_plus_E, mode_minus_E = self.slab.make_modes(kx, ky)
        return mode_plus_E[0]
                
    def Delta(self, kx, ky):
        mode_k = self.mode(kx, ky)
        phi_slab = mode_k.potential(self.z_res)
        K_res = self.resonator.K(kx, ky)
        return K_res * phi_slab.conj() * mu_0 * 0.25
    
    def Delta_prime(self, kx, ky):
        mode_k = self.mode(kx, ky)
        phi_slab = mode_k.potential(self.z_res)
        K_res = self.resonator.K(-kx, -ky)
        return K_res * phi_slab * mu_0 * 0.25

    def resonant_factor(self, omega):
        omega_0   = self.resonator.omega_0().real
        Gamma_res = self.Gamma_tot(omega)
        return 1.0 / (omega - omega_0 + 1j * Gamma_res)
    
    def Gamma_tot(self, omega):
        return self.resonator.gamma_0() + self.Gamma_rad(omega)
        
    def Gamma_rad(self, omega):
        def dGamma(theta):
            k_o = self.dispersion.k_omega(omega, theta)
            u_o = self.dispersion.u_omega(omega, theta)
            Delta_theta = self.Delta(k_o * np.cos(theta),
                                     k_o * np.sin(theta))
            return k_o / np.abs(u_o) * np.abs(Delta_theta)**2

        if  False:
            theta_vals  = np.linspace(0.0, 2.0 * np.pi, 1001)
            dGamma_vals = np.vectorize(dGamma)(theta_vals)
            Gamma_rad = integrate.trapz(dGamma_vals, theta_vals)
            pl.plot(theta_vals, dGamma_vals)
        else:
            Gamma_rad, eps = integrate.quad(dGamma, 0.0, 2.0 * np.pi)

        print ("o = ", omega, "Gamma_rad = ", Gamma_rad)    
        Gamma_rad *= 0.25 / np.pi
        return Gamma_rad
    
    def Gamma_prime(self, omega):
        def dGamma_prime(theta):
            k_o = self.dispersion.k_omega(omega, theta)
            u_o = self.dispersion.u_omega(omega, theta)
            Delta1_theta = self.Delta_prime(k_o * np.cos(theta),
                                            k_o * np.sin(theta))
            return k_o / np.abs(u_o) * np.abs(Delta1_theta)**2

        if  False:
            theta_vals  = np.linspace(0.0, 2.0 * np.pi, 1001)
            dGamma_vals = np.vectorize(dGamma_prime)(theta_vals)
            Gamma_prime = integrate.trapz(dGamma_vals, theta_vals)
            pl.plot(theta_vals, dGamma_vals)
        else:
            Gamma_prime, eps = integrate.quad(dGamma_prime, 0.0, 2.0 * np.pi)

        print ("o = ", omega, "Gamma_prime = ", Gamma_prime)    
        Gamma_prime *= 0.25 / np.pi
        return Gamma_prime
    
    def Gamma_tilde(self, omega):
        def dGamma_tilde(theta):
            k_o = self.dispersion.k_omega(omega, theta)
            u_o = self.dispersion.u_omega(omega, theta)
            kx = k_o * np.cos(theta)
            ky = k_o * np.sin(theta)
            Delta_theta  = self.Delta(kx, ky)
            Delta1_theta = self.Delta_prime(kx, ky)
            return k_o / np.abs(u_o) * Delta1_theta * Delta_theta

        def dGamma_tilde_re(theta):
            return dGamma_tilde(theta).real
        
        def dGamma_tilde_im(theta):
            return dGamma_tilde(theta).imag

        if  False:
            theta_vals  = np.linspace(0.0, 2.0 * np.pi, 1001)
            dGamma_vals = np.vectorize(dGamma_tilde)(theta_vals)
            Gamma_tilde = integrate.trapz(dGamma_vals, theta_vals)
            pl.plot(theta_vals, dGamma_vals)
        else:
            Gamma_tilde_re, eps_re = integrate.quad(dGamma_tilde_re, 0.0, 2.0 * np.pi)
            Gamma_tilde_im, eps_im = integrate.quad(dGamma_tilde_im, 0.0, 2.0 * np.pi)
            Gamma_tilde = Gamma_tilde_re + 1j * Gamma_tilde_im

        print ("o = ", omega, "Gamma_tilde = ", Gamma_tilde)    
        Gamma_tilde *= 0.25 / np.pi
        return Gamma_tilde

    def find_data_k(self, kx, ky):
        k = np.sqrt(kx**2 + ky**2)
        theta = np.angle(kx + 1j * ky)
        omega = self.dispersion.omega_k(kx, ky).real
        return find_data_theta(omega, theta, k)
       
    def find_data_theta(self, omega, theta, k = -1.0):
        if k < 0: 
           k   = self.dispersion.k_omega(omega, theta)
        kx  = k * np.cos(theta)
        ky  = k * np.sin(theta)
        Delta_k    = self.Delta(kx, ky)
        v_ox, v_oy = self.dispersion.v_omega(omega, theta)
        u_o        = self.dispersion.u_omega(omega, theta)
        v_o = np.sqrt(v_ox**2 + v_oy**2)
        alpha_o = np.angle (v_ox + 1j * v_oy)
        #alpha_o = self.dispersion.alpha(omega, theta)
        alpha_p = self.dispersion.alpha_prime(omega, theta)
        return Delta_k, k, u_o, v_o, alpha_o, alpha_p

# Obsolete. Remove?
class ScatteringProblem_1Dx:
    def __init__ (self, slab, resonator, z_res):
        self.resonator = resonator
        self.z_res = z_res
        self.dispersion = Dispersion(self.omega_k)
        self.theta_or = resonator.theta_or
        
    def describe(self):
        result = dict()
        result.update(self.slab.describe())
        result.update(self.resonator.describe())
        result.update(z_res = self.z_res, theta_or = self.theta_or)
        return result
        
    def omega_k(self, kx, ky):
        return self.mode(kx, ky).omega

    def Delta(self, k):
        kx = k * np.cos(self.theta_or)
        ky = k * np.sin(self.theta_or)
        mode_k = self.mode(kx, ky)
        phi_slab = mode_k.potential(self.z_res)
        K_res = self.resonator.K_1d(kx, ky)
        return K_res.conj() * phi_slab.conj() * mu_0 * 0.25

    def Gamma_rad(self, omega):
        k_R = self.dispersion.k_omega(omega, self.theta_or)
        k_L = self.dispersion.k_omega(omega, self.theta_or + np.pi)
        v_R = self.dispersion.u_omega(omega, self.theta_or)
        v_L = self.dispersion.v_omega(omega, self.theta_or + np.pi)
        Delta_R = self.Delta(k_R)
        Delta_L = self.Delta(-k_L)
        Gamma_R =  0.5 * np.abs(Delta_R)**2 / abs(v_R)
        Gamma_L =  0.5 * np.abs(Delta_L)**2 / abs(v_L)
        return Gamma_R + Gamma_L

    def Gamma_tot(self, omega):
        return self.resonator.gamma_0() + self.Gamma_rad(omega)

    def T_and_R(self, omega):
        k_R = self.dispersion.k_omega(omega, self.theta_or)
        k_L = self.dispersion.k_omega(omega, self.theta_or + np.pi)
        v_R = self.dispersion.u_omega(omega, self.theta_or)
        v_L = self.dispersion.v_omega(omega, self.theta_or + np.pi)
        Delta_R = self.Delta(k_R)
        Delta_L = self.Delta(-k_L)
        Gamma_R =  0.5 * np.abs(Delta_R)**2 / abs(v_R)
        Gamma_L =  0.5 * np.abs(Delta_L)**2 / abs(v_L)
        Gamma_0 =  self.resonator.gamma_0()
        Gamma_tot = Gamma_0 + Gamma_L + Gamma_R
        omega_denom = omega - omega_0 + 1j * Gamma_tot
        T = 1.0 - 2.0j * Gamma_R / omega_denom
        R = -1j * Delta_R * Delta_L.conjugate() / np.sqrt(np.abs(v_R * v_L))
        R *= 1.0 / omega_denom
        return T, R
        
