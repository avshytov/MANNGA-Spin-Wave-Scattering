import numpy as np
import pylab as pl
from scipy import integrate

class Dispersion:
    def __init__ (self, omega_k):
        self.omega_k = omega_k

    def G(self, omega, kx, ky):
        return 1.0 / (omega - self.omega_k(kx, ky))
    
    # Wavenumber for given frequency and angular position
    def k_omega(self, omega, theta):
        def F(k):
            #print ("k = ", k)
            kx_ = k * np.cos(theta)
            ky_ = k * np.sin(theta)
            #print ("kx_, ky_ = ", kx_, ky_)
            #mode_plus_E, mode_minus_E = slab.make_modes(kx_, ky_)
            #mode_plus, E_plus = mode_plus_E
            o_k = self.omega_k(kx_, ky_)
            return o_k.real - omega
        k_a = 0.0001
        k_b = 200.0
        f_a = F(k_a)
        f_b = F(k_b)


        while abs(k_b - k_a) > 1e-12:
              k_c = 0.5 * (k_a + k_b)
              f_c = F(k_c)
              if  (f_c * f_a <= 0):
                  k_b = k_c
                  f_b = f_c
              elif (f_c * f_b <= 0):
                  k_a = k_c
                  f_a = f_c
              else:
                  raise Exception("cannot find root"
                            " for omega = %g, theta = %g" % (omega, theta))
        k_result =  0.5 * (k_a + k_b)
        return k_result

    #
    # Naive group velocity d omega / d|k|
    #
    def u_omega(self, omega, theta):
        k_o = self.k_omega(omega, theta)
        dk = k_o * 1e-3
        cs = np.cos(theta)
        sn = np.sin(theta)
        omega_p = self.omega_k(cs * (k_o + dk), sn * (k_o + dk)).real
        omega_m = self.omega_k(cs * (k_o - dk), sn * (k_o - dk)).real
        return (omega_p - omega_m) / 2.0 / dk

    #
    # Full group velocity
    #
    #  v_omega = d omega / d|k| (k_hat - 1 / k_omega * theta_hat * dk/dtheta )
    #
    def v_omega(self, omega, theta):
        u_o = self.u_omega(omega, theta)
        cs = np.cos(theta)
        sn = np.sin(theta)
        k_hat = cs + 1j * sn
        theta_hat = 1j * k_hat
        dtheta = 1e-4
        k_o    = self.k_omega(omega, theta)
        k_p    = self.k_omega(omega, theta + dtheta)
        k_m    = self.k_omega(omega, theta - dtheta)
        kprime = (k_p - k_m) / 2.0 / dtheta
        v_o = u_o * (k_hat - theta_hat * kprime / k_o)
        return v_o.real, v_o.imag

    #
    # Propagation direction
    #
    def alpha(self, omega, theta):
        v_ox, v_oy = self.v_omega(omega, theta)
        return np.angle(v_ox + 1j * v_oy)

    #
    # d alpha / d theta: Needed to compute the outgoing wave
    #

    def alpha_prime(self, omega, theta):
        dtheta = 1e-4
        alpha_p = self.alpha(omega, theta + dtheta)
        alpha_m = self.alpha(omega, theta - dtheta)
        while alpha_p > alpha_m + np.pi:
            alpha_p -= 2.0 * np.pi
        while alpha_p < alpha_m - np.pi:
            alpha_p += 2.0 * np.pi
        return (alpha_p - alpha_m) / 2.0 / dtheta

    
