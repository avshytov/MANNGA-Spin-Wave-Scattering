import numpy as np
from constants import gamma_s, mu_0

def sinc(x):
    if abs(x) > 1e-6: return np.sin(x) / x
    return 1.0

def sinch(x):
    if abs(x) > 1e-6: return (1.0 - np.exp(-2.0 * x)) / 2.0 / x
    return 1.0

class Resonator:
    def __init__ (self, L, W, H, Ms, B, alpha, Nx, Nz, theta_or = 0):
        V = L * W * H
        self.L = L
        self.W = W
        self.H = H
        self.V = V
        self.Ms = Ms
        self.B = B
        self.alpha = alpha
        self.Nx = Nx
        self.Nz = Nz
        self.theta_or = theta_or
        self.theta_cs = np.cos(theta_or)
        self.theta_sn = np.sin(theta_or)
        self.omega_x = gamma_s * (self.B + self.Nx * mu_0 * self.Ms)
        self.omega_z = gamma_s * (self.B + self.Nz * mu_0 * self.Ms)
        self.C = np.sqrt(2.0 * self.V * self.Ms * gamma_s)
        self.ell_xz  = np.sqrt(self.omega_x / self.omega_z)
        if self.omega_x < 0:
            sel.ell_xz *= -1.0
        self.mx = self.C / np.sqrt(np.abs(self.ell_xz))
        self.mz = -1j * self.mx * self.ell_xz

    def describe(self):
        return dict(
            res_Ms    = self.Ms,
            res_Bbias = self.B,
            res_alpha = self.alpha,
            res_Nx    = self.Nx,
            res_Nz    = self.Nz,
            res_V     = self.V,
            res_L     = self.L,
            res_H     = self.H,
            res_W     = self.W, 
            res_omega_0 = self.omega_0(),
            res_gamma_0 = self.gamma_0(),
            res_orientation = self.theta_or)

    def form_factors(self, kx, ky, q):
        #return 1.0, 1.0
        f_x = sinc(-kx * self.L / 2.0)
        #f_y = 1.0
        f_y = sinc(-ky * self.W / 2.0)
        f_z = sinch(q * self.H / 2.0)
        f_xyz = f_x * f_y * f_z
        return f_xyz, 0.0, f_xyz
        
    def K(self, kx, ky):
        #omega_x, omega_y = self.omega_xz()
        k = np.sqrt(kx**2 + ky**2)
        #k_hor = kx * self.theta_cs + ky * self.theta_sn
        kx_1 =   kx * self.theta_cs + ky * self.theta_sn
        ky_1 =  -kx * self.theta_sn + ky * self.theta_cs
        F_x, F_y, F_z = self.form_factors(kx_1, ky_1, k)
        Fk_hor = F_x * kx_1 + F_y * ky_1
        return (1j * Fk_hor * self.mx + F_z * k * self.mz)

    def K_1D(self, k):
        kx_1 =   k
        ky_1 =   0.0
        F_x, F_y, F_z = self.form_factors(kx_1, ky_1, np.abs(k))
        #print ("k = ", kx_1, ky_1, "F = ", F_x, F_y, F_z)
        Fk_hor = F_x * k
        wt = 1.0 / np.sqrt(self.W)
        return (1j * Fk_hor * self.mx + F_z * np.abs(k) * self.mz) * wt

    def Nxz(self):
        return self.Nx, self.Nz

    def omega_xz(self):
        Nx, Nz = self.Nxz()
        omega_x = (self.B + Nx * self.Ms * mu_0) * gamma_s
        omega_z = (self.B + Nz * self.Ms * mu_0) * gamma_s
        return omega_x, omega_z

    def freq_and_damping(self):
        omega_x, omega_z = self.omega_xz()
        omega_0 = np.sqrt(omega_x * omega_z)
        gamma_0 = self.alpha / 2.0 * (omega_x + omega_z)
        return omega_0, gamma_0

    def omega_0(self):
        omega_, gamma_ = self.freq_and_damping()
        return omega_

    def gamma_0(self):
        omega_, gamma_ = self.freq_and_damping()
        return gamma_    

class ResonatorAnti(Resonator):
    def __init__ (self, L, W, H, Ms, B, alpha, Nx, Ny, theta_or = 0):
        Resonator.__init__(self, L, W, H, Ms, B, alpha, Nx, Ny, theta_or)

    def describe(self):
        res_desc = Resonator.describe(self)
        result = {}
        for k in res_desc.keys():
            k_new = str(k)
            if k[0:4] == 'res_':
                k_new = 'res_anti_' + k[4:]
            result[k_new] = res_desc[k]
        return result

    def form_factors(self, kx, ky, q):
        #return 1.0
        f_x = sinc(kx * self.L / 2.0)
        #f_y = 1.0
        f_y = sinc(ky * self.W / 2.0)
        f_zs = 1.0 - np.exp(-q * self.H)
        f_zs /= q * self.H
        f_za = 1.0 - 2.0 * np.exp(-q * self.H/2.0) + np.exp(-q * self.H)
        f_za /= q * self.H
        return f_x * f_y * f_za, 0.0, f_x * f_y * f_zs
    
    
