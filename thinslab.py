import numpy as np
import pylab as pl
from scipy import linalg
from scipy import integrate

from constants import mu_0, gamma_s

class UniformMode:
    def __init__ (self, slab, kx, ky, omega, mx, mz, phi):
        self.slab = slab
        self.kx = kx
        self.ky = ky
        self.omega = omega
        self.mx = mx
        self.mz = mz
        self.phi = phi
        self.a = slab.a
        self.b = slab.b

    def energy(self):
        E = self.slab.energy(self)
        return E

    def potential(self, z_probe):
        k = np.sqrt(self.kx**2 + self.ky**2)
        if z_probe > self.b:
            return np.exp( - k  *  (z_probe - self.b)) * self.phi
        if z_probe < self.a:
            return np.exp( - k * (self.a - z_probe)) * self.phi
        return self.phi

    def field(self, z_probe):
        k = np.sqrt(self.kx**2 + self.ky**2)
        if z_probe > self.b:
            phi_exp = np.exp( - k  *  (z_probe - self.z[-1])) * self.phi
            q_z = -k
        if z_probe < self.z[0]:
            phi_exp = np.exp( - k * (self.a - z_probe)) * self.phi
            q_z = k
        H_x = phi_exp * 1j * self.kx
        H_y = phi_exp * 1j * self.ky
        H_z = phi_exp * q_z
        return H_x, H_y, H_z

    

class ThinSlab:
    def __init__ (self, a, b, Bext, Ms, Jex, alpha):
        self.a = a
        self.b = b
        self.Bext = Bext
        self.Ms = Ms
        self.Jex = Jex
        self.alpha = alpha
        self.d = np.abs(self.b - self.a)

    def describe(self):
        return dict(
               thin_slab_a     = self.a,
               thin_slab_b     = self.b,
               thin_slab_Bext  = self.Bext,
               thin_slab_Ms    = self.Ms,
               thin_slab_Jex   = self.Jex,
               thin_slab_alpha = self.alpha)

    def Nxz(self, kx, ky):
        k = np.sqrt(kx**2 + ky**2)
        exp_kd = np.exp(-k * self.d)
        Nz = (1.0 - exp_kd) / k / self.d
        Nx = kx**2 / k**2 * (1.0 - Nz)
        return Nx, Nz
    
    def energy(self, mode):
        Jex = self.Jex
        Bext = self.Bext
        Ms = self.Ms
        alpha = self.alpha
        kx = mode.kx
        ky = mode.ky
        mx = mode.mx
        mz = mode.mz
        phi = mode.phi
        k2 = kx**2 + ky**2
        k = np.sqrt(k2)
        
        m2 = np.abs(mx)**2 + np.abs(mz)**2

        # Exchange
        #
        # (nabla_x m_x)^2 + (nabla_x m_z)^2 + (nabla_y ...)^2 
        #
        Eex = 0.0
        Eex += 0.5 * Jex  * k2 * m2

        Nx, Nz = self.Nxz(kx, ky)

        # Bias contribution: 
        Ebias = 0.5 * Bext / Ms * m2

        E_m = 0.5 * np.abs(mx)**2 * Nx * mu_0 + 0.5 * np.abs(mz)**2 * Nz * mu_0
        E = (Eex + Ebias + E_m) * self.d 
        return E
        
    def make_modes(self, kx, ky):
        Jex  = self.Jex
        Ms   = self.Ms
        Bext = self.Bext
        alpha = self.alpha
        
        k2 = kx**2 + ky**2
        k = np.sqrt(k2)

        Nx, Nz = self.Nxz(kx, ky)

        Omega_x = gamma_s * (Bext + mu_0 * Nx * Ms + Jex * Ms * k2)
        Omega_z = gamma_s * (Bext + mu_0 * Nz * Ms + Jex * Ms * k2)

        e_xz = np.sqrt(Omega_x / Omega_z)
        Omega_k = np.sqrt(Omega_x * Omega_z)
        gamma_k = alpha / 2.0 * (Omega_x + Omega_z)
        C = np.sqrt(2 * gamma_s * Ms / self.d)
        mx = C / np.sqrt(e_xz)
        mz = -1j * C * np.sqrt(e_xz)
        #print ("product: ", (mz.conjugate() * mx).imag * self.d,
        #       2 * gamma_s * Ms)
        exp_kd = np.exp(-k * self.d)
        phi_plus  = 0.5 / k**2 * (1.0 - exp_kd) * (1j * kx * mx - k * mz)
        phi_minus = 0.5 / k**2 * (1.0 - exp_kd) * (1j * kx * mx + k * mz)
        omega_k = Omega_k - 1j * gamma_k
        mode_plus  = UniformMode(self, kx, ky, omega_k,
                                 mx, mz, phi_plus)
        mode_minus = UniformMode(self, kx, ky, -omega_k.conj(),
                                 mx, -mz, phi_minus)
        E_plus  = self.energy(mode_plus)
        E_minus = self.energy(mode_minus)
        #print ("E/omega: ", E_plus / Omega_k,
        #       E_minus / Omega_k, "k = ", kx, ky)

        return (mode_plus, E_plus), (mode_minus, E_minus)

