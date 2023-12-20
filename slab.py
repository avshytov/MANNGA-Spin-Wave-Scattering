import numpy as np
import pylab as pl
from scipy import linalg
from scipy import integrate

from constants import mu_0, gamma_s

class Mode:
    def __init__ (self, slab, kx, ky, omega, mx, mz, phi):
        self.slab = slab
        self.kx = kx
        self.ky = ky
        self.omega = omega
        self.mx = mx
        self.mz = mz
        self.phi = phi
        self.z = slab.z

    def energy(self):
        E = self.slab.energy(self)
        return E

    def potential(self, z_probe):
        k = np.sqrt(self.kx**2 + self.ky**2)
        if z_probe > self.z[-1]:
            return np.exp( - k  *  (z_probe - self.z[-1])) * self.phi[-1]
        if z_probe < self.z[0]:
            return np.exp( - k * (self.z[0] - z_probe)) * self.phi[0]
        phi_re_spl = interpolate.splrep(self.z, self.phi.real)
        phi_im_spl = interpolate.splrep(self.z, self.phi.imag)
        phi_re = interpolate.splev(z_probe, phi_re_spl)
        phi_im = interpolate.splev(z_probe, phi_im_spl)
        return phi_re + 1j * phi_im

    def field(self, z_probe):
        k = np.sqrt(self.kx**2 + self.ky**2)
        if z_probe > self.z[-1]:
            phi_exp = np.exp( - k  *  (z_probe - self.z[-1])) * self.phi[-1]
            q_z = -k
        if z_probe < self.z[0]:
            phi_exp = np.exp( - k * (self.z[0] - z_probe)) * self.phi[0]
            q_z = k
        H_x = phi_exp * 1j * self.kx
        H_y = phi_exp * 1j * self.ky
        H_z = phi_exp * q_z
        return H_x, H_y, H_z

    

class Slab:
    def __init__ (self, a, b, Bext, Ms, Jex, alpha, N):
        self.a = a
        self.b = b
        self.Bext = Bext
        self.Ms = Ms
        self.Jex = Jex
        self.alpha = alpha
        self.N = N
        self.z = np.linspace(a, b, N + 1) # Discretised vertical coordinate
        self.wt = np.ones ((N + 1))
        self.wt[0] = self.wt[-1] = 0.5

    def describe(self):
        return dict(
               slab_a     = self.a,
               slab_b     = self.b,
               slab_Bext  = self.Bext,
               slab_Ms    = self.Ms,
               slab_Jex   = self.Jex,
               slab_alpha = self.alpha,
               slab_N     = self.N)
        
    #
    # Energy of the mode
    #
    # There are four contributions:
    #
    #   *  Dot product M.H_bias
    #
    #        - M . H =  - M_y * H_bias = (Mx^2 + M_y^2) * H_bias / (2 Ms) 
    #
    #   *  Exchange term
    #
    #      1/2 J_ex (nabla M)^2
    #
    #   * Magnetic energy
    #
    #     1/2 ( (H_x)^2 + (H_z)^2) = 1/2 (nabla phi)^2
    #
    #   * Magnetodipole interaction - M nabla phi, which we rewrite in terms of
    #     magnetic charges
    #
    #     int (M nabla phi) = - int phi \nabla M + M_boundary * phi_boundary
    #
    #  Discretisation:
    #  * Magnetisation and potential live on the nodes.
    #
    #  * x, y derivatives live on the nodes, z derivatives live on the links (midpoints)
    #
    #  * The slab is split
    #    into boxes centered on the nodes. The boundary boxes are 1/2 of the width:
    #
    #     B--|--N--|--N--|-- ..... --N--|--N--|--B
    #
    #  * This effectively places magnetic charges m_1 - m_2 at each inner boundary, and
    #    the charge +-m at the outer boundaries
    #
    #    E = - sum (m_cur - m_next) * 1/2 (phi_cur + phi_next) - m_0 phi_0 + m_last * phi_last
    #
    
    def energy(self, mode):
        Jex = self.Jex
        Bext = self.Bext
        Ms = self.Ms
        alpha = self.alpha
        z = self.z
        kx = mode.kx
        ky = mode.ky
        mx = mode.mx
        mz = mode.mz
        phi = mode.phi
        k2 = kx**2 + ky**2
        k = np.sqrt(k2)
        h = self.z[1] - self.z[0]
        wt_h = self.wt * h

        m2 = np.abs(mx)**2 + np.abs(mz)**2

        # Exchange
        #
        # (nabla_x m_x)^2 + (nabla_x m_z)^2 + (nabla_y ...)^2 
        #
        Eex = 0.0
        Eex += 0.5 * Jex  * k2 * np.sum(m2 * wt_h)
        
        dz = z[1:] - z[:-1]
        zm = 0.5 * (z[1:] + z[:-1])
        dmx = (mx[1:] - mx[:-1]) / dz
        dmz = (mz[1:] - mz[:-1]) / dz
        phi_m = 0.5 * (phi[1:] + phi[:-1])
        mzm = 0.5 * (mz[1:] + mz[:-1])

        # (nabla_z m_x)^2 + (nabla_z m_z)^2
        Eex += 0.5 * Jex  * np.sum(np.abs(dmx)**2 * dz)
        Eex += 0.5 * Jex  * np.sum(np.abs(dmz)**2 * dz)

        # Bias contribution: 
        Ebias = 0.5 * Bext / Ms * np.sum(wt_h * m2)

        # Magnetic energy
        dphi = (phi[1:] - phi[:-1]) / dz
        E_m = 0.0
        #
        # (nabla_x phi)^2 + (nabla_y phi)^2 
        #
        E_m += -0.5 * k2 * np.sum(np.abs(phi)**2 * wt_h)
        #
        # (nabla_z phi)^2
        #
        E_m += -0.5 * np.sum(np.abs(dphi)**2 * dz)
        #
        # Outer-space contribution: the potential decays as exp(-kz),
        # so that
        #
        #  (nabla_x phi)^2 + (nabla_y phi)^2 + (nabla_z phi)^2 = 2 k^2 exp(-2kz) * phi_0^2
        #
        # Integrating this yields 1/2 |k| phi_0^2
        #
        E_m += -0.5 * k * np.abs(phi[0])**2
        E_m += -0.5 * k * np.abs(phi[-1])**2


        #
        # Magnetodipole energy: first nabla_x m_x phi
        #
        E_mphi = np.sum( (1j * kx * mx.conj() * phi).real * wt_h)
        #
        #  Now nabla_z m_z * phi 
        #
        E_mphi += - np.sum( (dmz.conj () * phi_m).real * dz)
        #
        # Magnetic charges at the boundaries: m * phi
        #
        E_mphi += - (mz[0].conj () * phi[0]).real  
        E_mphi += + (mz[-1].conj() * phi[-1]).real 
        #print ("Exchange energy: ", Eex)
        #print ("Bias: ", Ebias)
        ##print ("magnetostatic: ", E_m)
        #print ("m * dphi = ", E_mphi)
        E = Eex + Ebias + mu_0 * (E_m + E_mphi)
        #print ("total:", E)
        return E
        
    def make_modes(self, kx, ky, n = 1):
        Jex  = self.Jex
        Ms   = self.Ms
        Bext = self.Bext
        alpha = self.alpha
        N = self.N
        z = self.z
        h = self.z[1] - self.z[0]

        k2 = kx**2 + ky**2
        k = np.sqrt(k2)

        #
        #  We have to solve the LL equations
        #
        #  -i omega m_x = Bext * m_z - M_s * H_z
        #  -i omega m_z = M_s * H_x - Bext * m_x
        #
        #  The effective fields should match the formula for energy above
        #
        #  in which the field is taken in the form H = J nabla^2 m - nabla phi,
        #  i.e. 
        #
        #  H_z = J nabla^2 m_z - d phi / dz    H_x = J nabla^2 m_x - i k_x phi 
        #
        #  with nabla^2 = d^2/dz^2 - k_x^2 - k_y^2 
        #
        #  The magnetostatic potential phi obeys the eqn
        #
        #   nabla^2 phi = nabla * m = i k_x m_x + nabla_z m_z
        #
        #  The potential behaves as exp(-|k||z|) outside 
        #
        # Method:
        # 
        # 1. There is 90deg phase shift between m_x and m_z, phi. We therefore
        #    change the variables to m_x, 1j * m_z, 1j * phi
        #
        # 2. The equations are rewritten as
        #
        #        omega m = L_0 * m + P * phi 
        #
        #        Q phi + A m = 0
        #
        #    where L_0, P, Q, and A are discretised operators.
        #
        #  3. This way, one may solve for phi:  phi = - Qinv * A * m
        #
        #     After substituting phi into eqn for m, one finds
        #
        #     omega * m = L * m  with L = L_0 - P . Qinv . A
        #
        #  4. Thus, we compute the operator L and then diagonalise it
        #

        L_0 = np.zeros((2 * (N + 1), 2 * (N + 1)))
        Q = np.zeros((N + 1, N + 1))
        A = np.zeros((N + 1, 2 * (N + 1)))
        P = np.zeros((2 * (N + 1), N + 1))

        # Diagonal terms
        for i in range (0, N + 1):
            ix_cur   = 2 * i
            iz_cur   = ix_cur + 1
            iphi_cur = i
            wt_i = self.wt[i]

            # omega m_x = - Bext * m_z + ...
            # omega m_z = - Bext * m_x + ...
            L_0[ix_cur, iz_cur] += -  Bext * wt_i
            L_0[iz_cur, ix_cur] += -  Bext * wt_i

            #
            # Exchange contribution (diagonal part k^2 from nabla^2)
            #
            L_0[ix_cur, iz_cur] += Ms * Jex * (- k2) * wt_i
            L_0[iz_cur, ix_cur] += Ms * Jex * (- k2) * wt_i

            # The same for nabla^2 phi
            Q[iphi_cur, iphi_cur] +=  -k2 * wt_i

            # nabla^2 phi - k_x m_x 
            A[iphi_cur, ix_cur  ] +=  -kx * wt_i

            # Hx = k_x phi
            P[iz_cur,   iphi_cur] +=   kx * Ms * wt_i

        #
        # Go over the links
        #
        # The best way to understand the terms below is by discretising the
        # energy
        #
        #       J (m_next - m_cur)^2 / 2 h^2
        #
        #    + (m_cur - m_next)/2.0 * (phi_next + phi_cur) - m_0 * phi_0 + m_last * phi_last
        #
        #    + (phi_next - phi_cur)^2 / 2.0
        #
        # and compute the respective fields by computing the variation
        # with respect to the relevant variables
        #
        # Effectively, this discretisation puts magnetic charges (m_cur - m_next) at the
        # middle of each link, and two magnetic charges at the endpoints
        #
        #
        for i in range (0, N):
            i_next = i + 1
            ix_cur = 2 * i
            iz_cur = ix_cur + 1
            iphi_cur = i
            ix_next = 2 * i_next
            iz_next = ix_next + 1
            iphi_next = i_next

            #
            # Exchange field d^2 m_x / dz^2
            #
            L_0[ix_cur,  iz_cur]  += Ms * Jex * (-1.0/h**2)
            L_0[ix_cur,  iz_next] += Ms * Jex * (1.0/h**2)
            L_0[ix_next, iz_cur]  += Ms * Jex * (1.0/h**2)
            L_0[ix_next, iz_next] += Ms * Jex * (-1.0/h**2)

            #
            # Exchange field d^2 m_z / dz^2
            #
            L_0[iz_cur,  ix_cur]  += Ms * Jex * (-1.0/h**2)
            L_0[iz_cur,  ix_next] += Ms * Jex * (1.0/h**2)
            L_0[iz_next, ix_cur]  += Ms * Jex * (1.0/h**2)
            L_0[iz_next, ix_next] += Ms * Jex * (-1.0/h**2)

            #
            # H_z = - dphi / dz
            #
            # Each gradient is taken symmetrically
            #
            ##P[ix_cur, iphi_cur]   += + 0.5 / h * Ms 
            ##P[ix_cur, iphi_next]  += - 0.5 / h * Ms 
            ##P[ix_next, iphi_cur]  += + 0.5 / h * Ms 
            ##P[ix_next, iphi_next] += - 0.5 / h * Ms 

            P[ix_cur, iphi_cur]   += - 0.5 / h * Ms 
            P[ix_cur, iphi_next]  += - 0.5 / h * Ms 
            P[ix_next, iphi_cur]  += + 0.5 / h * Ms 
            P[ix_next, iphi_next] += + 0.5 / h * Ms 

            #
            #  nabla^2 phi
            #
            Q[iphi_cur,  iphi_cur]  += (-1.0/h**2)
            Q[iphi_cur,  iphi_next] += ( 1.0/h**2)
            Q[iphi_next, iphi_cur]  += ( 1.0/h**2)
            Q[iphi_next, iphi_next] += (-1.0/h**2)

            #
            # dm / dz
            #
            A[iphi_cur, iz_cur]   += +0.5 / h  
            A[iphi_cur, iz_next]  += -0.5 / h  # 
            A[iphi_next, iz_cur]  += +0.5 / h  # 
            A[iphi_next, iz_next] += -0.5 / h  

        #
        # Boundary conditions:
        #
        #  -dphi/dz + m_z = -|k| phi
        #
        # The first and the last row of Q is simply
        # the normal magnetic field at the boundary
        #
        Q[0, 0]   -= k / h 
        Q[-1, -1] -= k / h
        #
        # Magnetic charges at the ends
        #
        A[0,   1] += -1.0  / h
        A[-1, -1] +=  1.0  / h

        #
        # Contribution of the endpoints
        #
        P[0,  0]  -= -1.0 / h * Ms
        P[-2, -1] += -1.0 / h * Ms

        # Determine Green's function for the potential: phi = Qinv . A . m
        Qinv = linalg.inv(Q)

        if  False: # Graph the important quantities, for debugging
            QinvA = np.dot(Qinv, A)
            QinvAx = QinvA[:, 0::2]
            QinvAz = QinvA[:, 1::2]
            pl.figure()
            pl.plot(z, Qinv[:, 0], label='0')
            pl.plot(z, Qinv[:, 1], label='1/N')
            pl.plot(z, Qinv[:, 2], label='2/N')
            pl.plot(z, Qinv[:, N//4], label='1/4')
            pl.plot(z, Qinv[:, N//2], label='1/2')
            pl.plot(z, Qinv[:, 3*N//4], label='3/4')
            pl.plot(z, Qinv[:, -3], label='1-2/N')
            pl.plot(z, Qinv[:, -2], label='1-1/N')
            pl.plot(z, Qinv[:, -1], label='1')
            pl.legend()
            pl.title("Q: k = %g %g" % (kx, ky))
            pl.figure()
            pl.title("QAz: k = %g %g" % (kx, ky))
            pl.plot(z, QinvAz[:, 0], label='0')
            pl.plot(z, QinvAz[:, 1], label='1/N')
            pl.plot(z, QinvAz[:, 2], label='2/N')
            pl.plot(z, QinvAz[:, (N//4)], label='1/4')
            pl.plot(z, QinvAz[:, N//2], label='1/2')
            pl.plot(z, QinvAz[:, 3*N//4], label='3/4')
            pl.plot(z, QinvAz[:, -3], label='1-2/N')
            pl.plot(z, QinvAz[:, -2], label='1-1/N')
            pl.plot(z, QinvAz[:, -1], label='1')
            pl.legend()
            pl.figure()
            pl.title("QAx: k = %g %g" % (kx, ky))
            pl.plot(z, QinvAx[:, 0], label='0')
            pl.plot(z, QinvAx[:, 1], label='1/N')
            pl.plot(z, QinvAx[:, 2], label='2/N')
            pl.plot(z, QinvAx[:, (N//4)], label='1/4')
            pl.plot(z, QinvAx[:, N//2], label='1/2')
            pl.plot(z, QinvAx[:, 3*N//4], label='3/4')
            pl.plot(z, QinvAx[:, -3], label='1-2/N')
            pl.plot(z, QinvAx[:, -2], label='1-1/N')
            pl.plot(z, QinvAx[:, -1], label='1')
            pl.legend()
            pl.figure()
            QinvAz1 = np.dot(QinvAz, np.ones(N + 1))
            QinvAx1 = np.dot(QinvAx, np.ones(N + 1))
            pl.title("QA1: k = %g %g" % (kx, ky))
            pl.plot(z, QinvAx1[:], label='mx = 1')
            pl.plot(z, QinvAz1[:], label='mz = 1')
            pl.legend()
            H_x = np.dot(P, QinvAx1)
            H_zx = H_x[0::2]
            H_xx = H_x[1::2]
            H_z = np.dot(P, QinvAz1)
            H_zz = H_z[0::2]
            H_xz = H_z[1::2]
            pl.figure()
            pl.plot(z, H_xx, label='Hx for mx = 1')
            pl.plot(z, H_zx, label='Hz for mx = 1')
            pl.plot(z, H_xz, label='Hx for mz = 1')
            pl.plot(z, H_zz, label='Hz for mz = 1')
            pl.legend()
            pl.show()

        # Construct the operator for the eigensystem
        L = gamma_s * ( L_0 - mu_0 * np.dot(P, np.dot(Qinv, A)) )

        # Compensate half-weights for endpoints
        #W = np.diag(wt)
        #L[0, :]  *= 2.0
        #L[1, :]  *= 2.0
        #L[-2, :] *= 2.0
        #L[-1, :] *= 2.0
        Winv = np.diag(1.0 / self.wt)
        L[ ::2, :] = np.dot(Winv, L[ ::2, :])
        L[1::2, :] = np.dot(Winv, L[1::2, :])

        # Include the Hilbert damping. Because of the damping,
        # the equations are to be modified as
        #
        #  [  1     alpha ]
        #  |              |   dm/dt = H_total
        #  [ -alpha     1 ]
        #
        #  Inverting, we write 
        #
        L_alpha = np.zeros(np.shape(L), dtype=complex)
        C_alpha = 1.0 / (1.0 + alpha**2)
        ialpha = 1j * alpha # Do not forget about the imaginary unit
                            # between mx and mz
        #print ("alpha = ", self.alpha)
        L_alpha[ ::2, ::2] = (L[ ::2,  ::2] + ialpha * L[1::2,  ::2]) * C_alpha
        L_alpha[ ::2,1::2] = (L[ ::2, 1::2] + ialpha * L[1::2, 1::2]) * C_alpha
        L_alpha[1::2, ::2] = (L[1::2,  ::2] + ialpha * L[ ::2,  ::2]) * C_alpha
        L_alpha[1::2,1::2] = (L[1::2, 1::2] + ialpha * L[ ::2, 1::2]) * C_alpha

        # Diagonalise it, to find the modes
        omega, psi = linalg.eig(L_alpha)
        #print ("eigenals: ", omega)


        omega_plus  = [t for t in enumerate(omega) if t[1].real >= 0.0]
        omega_minus = [t for t in enumerate(omega) if t[1].real <  0.0]

        omega_plus.sort(key = lambda x: abs(x[1]))
        omega_minus.sort(key = lambda x: abs(x[1]))


        def get_mode(i_mode):
            psi_i = psi[:, i_mode]
            omega_i = omega[i_mode]
            #print ("omega = ", omega_i, "k = ", kx, ky)
            #print ("check the eigenvector: ", 
            #   linalg.norm( np.dot(L_alpha, psi_i) - omega_i * psi_i))

            # Extract the profile from the eigenvector
            mx = psi_i[0::2] 
            mz = psi_i[1::2] * 1j  # Take care of the 90deg phase
                                 # Determine the potential as above
            phi = -1j * np.dot(Qinv, np.dot(A, psi_i))
            C = integrate.trapz((mx.conj() * mz).imag, z) 
            #np.dot(mx.conj() , mz).imag
            Knorm = 1.0 / np.sqrt(np.abs(C / 2.0 / gamma_s / Ms))
            ix_max = np.argmax(np.abs(mx))
            Knorm *= np.abs(mx[ix_max]) / mx[ix_max]
            mx  *= Knorm
            mz  *= Knorm
            phi *= Knorm
            mode = Mode(self, kx, ky, omega_i, mx, mz, phi)
            E_i = self.energy(mode)
            #print ("energy: ", E_i, E_i / np.abs(omega_i))
            return mode, E_i

        if n == 1:
           return get_mode(omega_plus[0][0]), get_mode(omega_minus[0][0])
        res_plus  = []
        res_minus = []
        for i in range(n):
            res_plus.append(get_mode(omega_plus[i][0]))
            res_minus.append(get_mode(omega_minus[i][0]))

        return res_plus, res_minus
        
    #return np.abs(omega_min), z, mx, mz, phi

