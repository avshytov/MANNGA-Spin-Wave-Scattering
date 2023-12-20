import numpy as np
from constants import gamma_s, mu_0
from modes2 import CellArrayModel, Area, Grid, Material
from scipy import linalg
import constants


def sinc(x):
    if abs(x) > 1e-6: return np.sin(x) / x
    return 1.0

def sinch(x):
    if abs(x) > 1e-6: return (1.0 - np.exp(-2.0 * x)) / 2.0 / x
    return 1.0

class ResonantMode2D:
    def __init__ (self, L, W, H, X, Z, dV, omega, mx, mz, theta_or = 0.0):
        self.omega = omega
        self.mx = mx
        self.mz = mz
        self.mx_dual = mx.conj()
        self.mz_dual = mz.conj()
        self.X = X
        self.Z = Z
        self.L = L
        self.W = W
        self.V = L * W * H
        self.H = H
        self.dV = dV
        self.theta_or = theta_or
        self.theta_cs = np.cos(self.theta_or)
        self.theta_sn = np.sin(self.theta_or) 

    def describe(self):
        return dict(
            res_name  = "res_mode @ " + str(self.omega.real / constants.GHz_2pi) + "GHz",
            res_V     = self.V,
            res_L     = self.L,
            res_H     = self.H,
            res_W     = self.W, 
            res_omega_0 = self.omega_0(),
            res_gamma_0 = self.gamma_0(),
            res_orientation = self.theta_or)
        
    def mk_mode(self, kx, q, m_vals):
        exp_kr = np.exp( - q * self.Z - 1j * kx * self.X)
        return np.sum(self.dV * m_vals * exp_kr) * self.W

    def Mx(self, kx, q):
        return self.mk_mode(kx, q, self.mx)
    
    def Mx_dual(self, kx, q):
        return self.mk_mode(-kx, q, self.mx_dual)
    
    def Mz(self, kx, q):
        return self.mk_mode(kx, q, self.mz)
    
    def Mz_dual(self, kx, q):
        return self.mk_mode(-kx, q, self.mz_dual)
    
    def form_factors(self, kx, ky, q):
        #return 1.0, 1.0    
        #f_x = sinc(kx * self.L / 2.0)
        #f_y = 1.0
        f_y = sinc(ky * self.W / 2.0)
        #f_z = sinch(q * self.H / 2.0)
        #f_xyz = f_x * f_y * f_z
        return f_y, 0.0, f_y
        
    def K(self, kx, ky):
        #omega_x, omega_y = self.omega_xz()
        K_vals = []
        k = np.sqrt(kx**2 + ky**2)
        #k_hor = kx * self.theta_cs + ky * self.theta_sn
        kx_1 =   kx * self.theta_cs + ky * self.theta_sn
        ky_1 =  -kx * self.theta_sn + ky * self.theta_cs
        F_x, F_y, F_z = self.form_factors(kx_1, ky_1, k)
        Fk_hor = F_x * kx_1 + F_y * ky_1
        return (1j * Fk_hor * self.Mx(kx_1, k) + F_z * k * self.Mz(kx_1, k))
    
    def K_dual(self, kx, ky):
        #omega_x, omega_y = self.omega_xz()
        K_vals = []
        k = np.sqrt(kx**2 + ky**2)
        #k_hor = kx * self.theta_cs + ky * self.theta_sn
        kx_1 =   kx * self.theta_cs + ky * self.theta_sn
        ky_1 =  -kx * self.theta_sn + ky * self.theta_cs
        F_x, F_y, F_z = self.form_factors(kx_1, ky_1, k)
        Fk_hor = F_x * kx_1 + F_y * ky_1
        return (-1j * Fk_hor * self.Mx_dual(kx_1, k) + F_z * k * self.Mz_dual(kx_1, k))

    def K_1D(self, k):
        kx_1 =   k
        ky_1 =   0.0
        F_x, F_y, F_z = self.form_factors(kx_1, ky_1, np.abs(k))
        #print ("k = ", kx_1, ky_1, "F = ", F_x, F_y, F_z)
        Fk_hor = F_x * k
        wt = 1.0 / np.sqrt(self.W)
        #print ("mx: ", self.mx)
        return (1j * Fk_hor * self.Mx(kx_1, np.abs(k))
                + F_z * np.abs(k) * self.Mz(kx_1, np.abs(k))) * wt
    
    def K_1D_dual(self, k):
        kx_1 =   k
        ky_1 =   0.0
        F_x, F_y, F_z = self.form_factors(kx_1, ky_1, np.abs(k))
        #print ("k = ", kx_1, ky_1, "F = ", F_x, F_y, F_z)
        Fk_hor = F_x * k
        wt = 1.0 / np.sqrt(self.W)
        #print ("mx: ", self.mx)
        return (-1j * Fk_hor * self.Mx_dual(kx_1, np.abs(k))
                + F_z * np.abs(k) * self.Mz_dual(kx_1, np.abs(k))) * wt

    def freq_and_damping(self):
        return self.omega.real, -self.omega.imag

    def omega_0(self):
        omega_, gamma_ = self.freq_and_damping()
        return omega_

    def gamma_0(self):
        omega_, gamma_ = self.freq_and_damping()
        return gamma_    
        

def make_mode_collection(iomega, mxz, I, f_min, f_max,
                         Xr, Zr, dV_r, Lr, Wr, Hr, theta_or):

    omega = 1j * iomega

    i_pos = [t for t in range(len(omega)) if omega[t].real >= 0]
    i_pos.sort(key = lambda t: omega[t].real)

    modes = []

    for i in i_pos:
        print ("mode freq: ", omega[i] / constants.GHz_2pi)
        if omega[i].real < f_min * constants.GHz_2pi: continue
        if omega[i].real > f_max * constants.GHz_2pi: continue
        mxz_i = np.array(mxz[:, i])
        C_norm = np.dot(mxz_i.conj(), np.dot(I, mxz_i))
        C_omega = np.sqrt(1.0/(1.0 + omega[i].imag**2 / omega[i].real**2))
        mxz_i *= np.sqrt(1.0 / np.abs(C_norm)) 
        mxz_i *= np.sqrt(4.0 / np.sqrt(Wr))
        mxz_i *= C_omega
        i_max = np.argmax(np.abs(mxz_i))
        mxz_i *= np.abs(mxz_i[i_max]) / mxz_i[i_max]
        mx_i = mxz_i[0::2]
        mz_i = mxz_i[1::2]
        theta_or = 0.0
        mode = ResonantMode2D(Lr, Wr, Hr, Xr, Zr, dV_r, omega[i],              
                                mx_i, mz_i, theta_or)
        modes.append(mode)
        
    return modes

def make_resonant_mode_collection(res_model, f_min, f_max,
                                  Lr, Wr, Hr, theta_or):
        result = res_model.solve()
        X, Z = result['coords_all']
        dV = result['dV_all']
        modes = []
        for mode in result['modes']:
            print ("mode freq: ", mode.f)
            if mode.f > f_min and mode.f < f_max:
                print ("append")
                mode.normalize()
                mode.scale(np.sqrt(4.0 / Wr))
                mx, mz = mode.m_all()
                omega = constants.GHz_2pi * mode.f
                res_mode = ResonantMode2D(Lr, Wr, Hr, X, Z, dV,
                                        omega,
                                        np.array(mx), np.array(mz), theta_or)
                modes.append(res_mode)
        return modes

class Resonator2D:
    def __init__ (self, L, W, H, Ms, Jex, B, alpha, Nx, Nz, f_min, f_max,
                  theta_or = 0):
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

        res_material = Material("Resonator", Ms, Jex, alpha, gamma_s)
        res_area = Area("resonator",
                        Grid(-L/2.0, L/2.0, 0, H, Nx, Nz),
                        res_material)
        res_model = CellArrayModel()
        res_model.add_area(res_area, B)
        result = res_model.solve()
        #self.XR, self.ZR = result['coords']['resonator']
        X, Z = result['coords_all']
        dV = result['dV_all']
        self.modes = []
        m_dual_prev = []
        for mode in result['modes']:
            print ("mode freq: ", mode.f)
            if mode.f > f_min and mode.f < f_max:
                print ("append")
                try: 
                    mode.normalize()
                except:
                    import traceback
                    traceback.print_exc()
                    print ("cannot normalize in the usual way")
                #C_omega = np.sqrt(1.0 / (1.0 + (mode.f.imag/mode.f.real)**2))
                #mode.scale(C_omega)
                mode.scale(np.sqrt(4.0 / self.W))
                mx, mz           = mode.m_all()
                #print ("mx = ", mx)
                #print ("mz = ", mz)
                print ("norm(mx), norm(mz)",
                       linalg.norm(mx), linalg.norm(mz))
                mx_dual, mz_dual = mode.m_dual_all()
                print ("norm(mx_dual), norm(mz_dual)",
                       linalg.norm(mx_dual), linalg.norm(mz_dual))
                omega = constants.GHz_2pi * mode.f
                res_mode = ResonantMode2D(L, W, H, X, Z, dV,
                                        omega, mx, mz, theta_or)
                self.modes.append(res_mode)
                res_mode.mx_dual = mx_dual * 1.0 / W
                res_mode.mz_dual = mz_dual * 1.0 / W
                if True:
                    mxz = np.zeros((len(mx) + len(mz)), dtype=complex)
                    mxz_dual = np.zeros((len(mx) + len(mz)), dtype=complex)
                    mxz[0::2] = res_mode.mx
                    mxz[1::2] = res_mode.mz
                    mxz_dual[0::2] = res_mode.mx_dual
                    mxz_dual[1::2] = res_mode.mz_dual
                    I_a = res_model.get_I_alpha()
                    mm = np.dot(mxz_dual, np.dot(I_a, mxz)) * W
                    print ("mm = ", mm)
                    i_dual_max = np.argmax(np.abs(mxz_dual))
                    print ("norm(mxz)", linalg.norm(mxz))
                    print ("norm(dual)", linalg.norm(mxz_dual))
                    try: 
                       C_dual = mxz.conj()[i_dual_max] / mxz_dual[i_dual_max]
                       print ("C_dual = ", C_dual)
                       dual_diff = linalg.norm(mxz.conj() - mxz_dual)
                       dual_diff /= np.sqrt(linalg.norm(mxz))
                       dual_diff /= np.sqrt(linalg.norm(mxz_dual))
                       print ("dual - conj diff: ", dual_diff)
                    except:
                        import traceback
                        traceback.print_exc()
                        print ("cannot check for closeness of the dual")
                    for m_prev in m_dual_prev:
                        print ("overlap with prev: ",
                               np.abs(np.dot(m_prev, np.dot(I_a, mxz))*W))
                    m_dual_prev.append(mxz_dual)
        
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

    def omega_0(self):
        return np.array([t.omega_0() for t in self.modes])

    def gamma_0(self):
        return np.array([t.gamma_0() for t in self.modes])

class GeometrySpec:
     def __init__ (self, L, W, H, Nx, Nz):
         self.L = L; self.W = W; self.H = H
         self.Nx = Nx; self.Nz = Nz

class MaterialSpec:
     def __init__ (self, name, Ms, Jex, alpha, gamma_s = constants.gamma_s):
         self.Ms = Ms
         self.name = name
         self.Jex = Jex
         self.alpha = alpha
         self.gamma_s = gamma_s
         
class Resonator2DNearField:
    def __init__ (self, geometry_res, geometry_slab,
                  material_spec_res, material_spec_slab,
                  B, s, 
                  #L, W, H, Ms, Jex, B, alpha, Nx, Nz,
                  #s, d, Nsx, Nsz, 
                  f_min, f_max,
                  theta_or = 0):
        L = geometry_res.L
        W = geometry_res.W
        H = geometry_res.H
        
        V = L * W * H
        self.L = L
        self.W = W
        self.H = H
        self.V = V
        self.Ms = material_spec_res.Ms #Ms
        self.B = B
        self.alpha = material_spec_res.alpha #alpha
        self.Nx = Nx = geometry_res.Nx #Nx
        self.Nz = Nz = geometry_res.Nz #Nz
        self.Nsx = Nsx = geometry_slab.Nx #Nsx
        self.Nsz = Nsz = geometry_slab.Nz #Nsz
        self.d = d = geometry_slab.H #d
        self.s = s
        self.theta_or = theta_or
        self.theta_cs = np.cos(theta_or)
        self.theta_sn = np.sin(theta_or)

        Ms  = material_spec_res.Ms
        Jex = material_spec_res.Jex
        alpha = material_spec_res.alpha
        gamma_s = material_spec_res.gamma_s
        slab_Ms  = material_spec_slab.Ms
        slab_Jex = material_spec_slab.Jex
        slab_alpha = material_spec_slab.alpha
        slab_gamma_s = material_spec_slab.gamma_s
        
        res_material = Material(material_spec_res.name,
                                Ms, Jex, alpha, gamma_s)
        res_area = Area("resonator",
                        Grid(-L/2.0, L/2.0, 0, H,
                             geometry_res.Nx, geometry_res.Nz),
                        res_material)
        res_model = CellArrayModel()
        res_model.add_area(res_area, B)
        slab_material = Material(material_spec_slab.name, slab_Ms, slab_Jex,
                                 slab_alpha, slab_gamma_s)
        slab_area = Area("slab",
                         Grid(-L/2 - 5 * (H + s), L/2 + 5 * (H + s),
                              -s - d, -s, Nsx, Nsz),
                         slab_material)
        res_model.add_area(slab_area, B)

        res_mask  = res_model.get_area_mask("resonator")
        slab_mask = res_model.get_area_mask("slab")
        #print ("res_mask = ", res_mask, np.sum(res_mask))
        #print ("slab_mask = ", slab_mask, np.sum(slab_mask))

        L = res_model.compute_LLG_operator()
        #iomega_full, mxz_full = linalg.eig(L)
        #print ("full modes: ", iomega_full[:20] * 1j / constants.GHz_2pi)
        #I = res_model.get_I()
        I = res_model.get_I_alpha()
        print ("I = ", I)
        #Iinv = linalg.inv(I)
        IL = np.dot(I, L)
        print ("I . L done")
        #print ("hermiticity of IL: ",
        #       linalg.norm(IL - np.transpose(IL.conj())))
        
        #
        # There must be a way to do it faster and neater
        #
        def restrict(L, mask1, mask2):
            #return L[mask1 > 0, mask2 > 0]
            #out = []
            #N_L, M_L = np.shape(L)
            i_mask1 = [t for t in range (len(mask1)) if mask1[t] > 0]
            j_mask2 = [t for t in range (len(mask2)) if mask2[t] > 0]
            L_r = np.zeros((len(i_mask1), len(j_mask2)), dtype=L.dtype)
            if True:  # hopefully a faster method
              for i_R, i_L in enumerate(i_mask1):
                  L_r[i_R, :] = L[i_L, mask2 > 0]
            else:     # the old restriction method is retained just in case
              for i_R, i_L in enumerate(i_mask1):
                for j_R, j_L in enumerate(j_mask2):
                    L_r[i_R, j_R] = L[i_L, j_L]
            return L_r
            #for i in range(M_L):
            #    if mask1[i] < 0.5: continue
            #    line = []
            #    for j in range(N_L):
            #        if mask2[j] < 0.5: continue
            #        line.append(L[i, j])
            #    line = np.array(line)
            #    out.append(line)
            #return np.array(out)

        def restrict2(L, mask1, mask2):
            Nr = len([t for t in range(len(mask1)) if mask1[t] > 0])
            Mr = len([t for t in range(len(mask2)) if mask2[t] > 0])
            Lr = np.zeros((2 * Nr, 2 * Mr), dtype=L.dtype)
            Lr[0::2, 0::2] = restrict(L[0::2, 0::2], mask1, mask2)
            Lr[0::2, 1::2] = restrict(L[0::2, 1::2], mask1, mask2)
            Lr[1::2, 0::2] = restrict(L[1::2, 0::2], mask1, mask2)
            Lr[1::2, 1::2] = restrict(L[1::2, 1::2], mask1, mask2)
            return Lr
            
                

        Nr = len([t for t in range(len(res_mask)) if res_mask[t] > 0])
        Ns = len([t for t in range(len(slab_mask)) if slab_mask[t] > 0])
        #np.sum(res_mask)
        #Ns = np.sum(slab_mask)
        R  = np.zeros((2 * Nr, 2 * Nr))
        S  = np.zeros((2 * Ns, 2 * Ns))
        Q  = np.zeros((2 * Ns, 2 * Nr))
        K  = np.zeros((2 * Nr, 2 * Ns))
        Ir = np.zeros((2 * Nr, 2 * Nr)) 
        Is = np.zeros((2 * Ns, 2 * Ns))

        Xr = np.zeros((Nr))
        Zr = np.zeros((Nr))

        
        Xr = res_model.box_array.X[res_mask > 0]
        Zr = res_model.box_array.Z[res_mask > 0]
        dV_r = res_model.box_array.dV()[res_mask > 0]

        print ("Restrict Rr")
        R  = restrict2(IL, res_mask, res_mask)
        print ("Restrict Ir")
        Ir = restrict2(I,  res_mask, res_mask) 

        #R[0::2, 0::2] = restrict(IL[0::2, 0::2], res_mask, res_mask)
        #R[0::2, 1::2] = restrict(IL[0::2, 1::2], res_mask, res_mask)
        #R[1::2, 0::2] = restrict(IL[1::2, 0::2], res_mask, res_mask)
        #R[1::2, 1::2] = restrict(IL[1::2, 1::2], res_mask, res_mask)

        #Ir[0::2, 0::2] = restrict(I[0::2, 0::2], res_mask, res_mask)
        #Ir[0::2, 1::2] = restrict(I[0::2, 1::2], res_mask, res_mask)
        #Ir[1::2, 0::2] = restrict(I[1::2, 0::2], res_mask, res_mask)
        #Ir[1::2, 1::2] = restrict(I[1::2, 1::2], res_mask, res_mask)

        print ("restrict S")
        S = restrict2(IL, slab_mask, slab_mask)
        #S[0::2, 0::2] = restrict(IL[0::2, 0::2], slab_mask, slab_mask)
        #S[0::2, 1::2] = restrict(IL[0::2, 1::2], slab_mask, slab_mask)
        #S[1::2, 0::2] = restrict(IL[1::2, 0::2], slab_mask, slab_mask)
        #S[1::2, 1::2] = restrict(IL[1::2, 1::2], slab_mask, slab_mask)
        
        #Is[0::2, 0::2] = restrict(I[0::2, 0::2], slab_mask, slab_mask)
        #Is[0::2, 1::2] = restrict(I[0::2, 1::2], slab_mask, slab_mask)
        #Is[1::2, 0::2] = restrict(I[1::2, 0::2], slab_mask, slab_mask)
        #Is[1::2, 1::2] = restrict(I[1::2, 1::2], slab_mask, slab_mask)
        print ("restict Is")
        Is = restrict2(I, slab_mask, slab_mask)

        print ("restrict Q")
        Q = restrict2(IL, slab_mask, res_mask)
        #Q[0::2, 0::2] = restrict(IL[0::2, 0::2], slab_mask, res_mask)
        #Q[0::2, 1::2] = restrict(IL[0::2, 1::2], slab_mask, res_mask)
        #Q[1::2, 0::2] = restrict(IL[1::2, 0::2], slab_mask, res_mask)
        #Q[1::2, 1::2] = restrict(IL[1::2, 1::2], slab_mask, res_mask)

        print ("restrict K")
        K = restrict2(IL, res_mask, slab_mask)
        #K[0::2, 0::2] = restrict(IL[0::2, 0::2], res_mask, slab_mask)
        #K[0::2, 1::2] = restrict(IL[0::2, 1::2], res_mask, slab_mask)
        #K[1::2, 0::2] = restrict(IL[1::2, 0::2], res_mask, slab_mask)
        #K[1::2, 1::2] = restrict(IL[1::2, 1::2], res_mask, slab_mask)

        #Ir[0::2, 1::2] = - np.eye(Nr)
        #Ir[1::2, 0::2] = - Ir[0::2, 1::2]
        #Is[0::2, 1::2] = - np.eye(Ns)
        #Is[1::2, 0::2] = - Is[0::2, 1::2]

        #Ur, Sr, Vr = linalg.svd(R)
        #print ("svd(R) = ", Sr)
        #Us, Ss, Vs = linalg.svd(S)
        #print ("svd(S) = ", Ss)
        print ("compute the near-field operator")
        Sinv = linalg.inv(S)
        SinvQ = np.dot(Sinv, Q)
        R_nrf = R - np.dot(K, SinvQ)
        I_nrf = Ir + np.dot( np.dot(K, Sinv), np.dot(Is, SinvQ))
        I_nrf_inv = linalg.inv(I_nrf)
        #Lnrf = np.dot(Ir, Rnrf)
        L_nrf = np.dot(I_nrf_inv, R_nrf)

        #Sigma_2 = np.dot(np.dot(np.dot(K, Sinv), Is),
        #                 np.dot(Sinv, np.dot(Is, SinvQ)))

        print ("diagonalize the near-field operator")
        iomega, mxz_l, mxz = linalg.eig(L_nrf, left=True, right=True)
        omega = 1j * iomega

        mxz_dual = np.dot(mxz_l.transpose().conj(), I_nrf_inv)
        i_pos = [t for t in range(len(omega)) if omega[t].real > 0]
        i_pos.sort(key = lambda t: omega[t].real)

        i_neg = [t for t in range(len(omega)) if omega[t].real <= 0]
        i_neg.sort(key = lambda t: -omega[t].real)

        i_all = [t for t in range(len(omega))]
        i_all.sort(key = lambda t: np.abs(omega[t]))
        print ("lowest negative modes: ",
               [omega[t] / constants.GHz_2pi for t in i_neg[0:10]])
        print ("lowest modes: ",
               [omega[t] / constants.GHz_2pi for t in i_all[0:10]])
        print ("lowest positive modes: ",
               [omega[t] / constants.GHz_2pi for t in i_pos[0:10]])

        self.modes = []
        I_nrf_s = 0.5 * (I_nrf - np.transpose(I_nrf))
        I_nrf_a = I_nrf - I_nrf_s
        m_dual_prev = []
        for i in i_pos:
            print ("mode freq: ", omega[i] / constants.GHz_2pi)
            if omega[i].real < f_min * constants.GHz_2pi: continue
            if omega[i].real > f_max * constants.GHz_2pi: continue
            print ("append mode")
            mxz_i = np.array(mxz[:, i])
            mxz_dual_i = np.array(mxz_dual[i, :])
            if True:
                print ("verify m",
                       linalg.norm(np.dot(R_nrf, mxz_i)
                                   + 1j * omega[i] * np.dot(I_nrf, mxz_i)))
                print ("verify dual",
                       linalg.norm(np.dot(mxz_dual_i, R_nrf)
                                + 1j * omega[i] * np.dot(mxz_dual_i, I_nrf)))
            C_norm = np.dot(mxz_i.conj(), np.dot(I_nrf_s, mxz_i))
            C_omega = 1.0 / np.sqrt(1.0 + (omega[i].imag / omega[i].real)**2)
            mxz_i *= np.sqrt(1.0 / np.abs(C_norm))
            #mxz_i *= C_omega
            #corr = np.dot(mxz_i.conj(), np.dot(Sigma_2, mxz_i)) * omega[i]**2
            #corr /= np.dot(mxz_i.conj(), np.dot(I_nrf, mxz_i)) * 1j
            #print ("correction: ", corr)
            mxz_i *= np.sqrt(4.0 / np.sqrt(W))
            i_max = np.argmax(np.abs(mxz_i))
            mxz_i *= np.abs(mxz_i[i_max]) / mxz_i[i_max]

            mm = np.dot(mxz_dual_i, np.dot(I_nrf, mxz_i)) * W
            mxz_dual_i *= 1.0/mm * 4.0 * 1j
            if True:
                print ("overlap with current: ",
                       W * np.abs(np.dot(mxz_dual_i, np.dot(I_nrf, mxz_i))))
                i_dual_max = np.argmax(np.abs(mxz_dual_i))
                C_dual = mxz_i.conj()[i_dual_max] / mxz_dual_i[i_dual_max]
                print ("C_dual = ", C_dual)
                print ("diff:",
                       np.abs(linalg.norm( mxz_i.conj()
                                           - C_dual *  mxz_dual_i))
                      / np.sqrt(linalg.norm(mxz_i) * linalg.norm(mxz_dual_i)))
                for m_prev in m_dual_prev:
                    print ("overlap with prev:",
                           np.abs(np.dot(m_prev, np.dot(I_nrf, mxz_i))))
            m_dual_prev.append(mxz_dual_i)
            
            mx_i = mxz_i[0::2]
            mz_i = mxz_i[1::2]
            mx_dual_i = mxz_dual_i[0::2]
            mz_dual_i = mxz_dual_i[1::2]
                           
            mode = ResonantMode2D(L, W, H, Xr, Zr, dV_r, omega[i],
                                mx_i, mz_i, self.theta_or)
            mode.mx_dual = mx_dual_i
            mode.mz_dual = mz_dual_i
            if False:
               import pylab as pl
               mxz_i_slab = -np.dot(SinvQ, mxz_i)
               mx_s, mz_s = mxz_i_slab[0::2], mxz_i_slab[1::2]
               Xs = res_model.box_array.X[slab_mask > 0]
               Zs = res_model.box_array.Z[slab_mask > 0]
               mx_max = np.max(np.abs(mx_i))
               mz_max = np.max(np.abs(mz_i))
               pl.figure()
               pl.tripcolor(Xr / constants.nm, Zr / constants.nm,
                            np.abs(mx_i), vmin=0.0, vmax=mx_max,
                            cmap='magma')
               pl.tripcolor(Xs / constants.nm, Zs / constants.nm,
                            np.abs(mx_s), vmin=0.0, vmax=mx_max,
                            cmap='magma')
               pl.gca().set_aspect('equal', 'box')
               cb = pl.colorbar()
               cb.set_label(r"Horizontal magnetisation $|m_x|$, a.u.")
               pl.xlabel(r"Position $x$, nm")
               pl.xlabel(r"Position $z$, nm")

               pl.figure()
               pl.tripcolor(Xr / constants.nm, Zr / constants.nm,
                            np.abs(mz_i), vmin=0.0, vmax=mz_max,
                            cmap='magma')
               pl.tripcolor(Xs / constants.nm, Zs / constants.nm,
                            np.abs(mz_s), vmin=0.0, vmax=mz_max,
                            cmap='magma')
               pl.gca().set_aspect('equal', 'box')
               cb = pl.colorbar()
               cb.set_label(r"Vertical magnetisation, $|m_z|$, a.u.")
               pl.xlabel(r"Position $x$, nm")
               pl.xlabel(r"Position $z$, nm")
               pl.figure()
               pl.quiver(Xr / constants.nm, Zr / constants.nm,
                         5 * mx_i.real, 5 * mz_i.real, pivot='middle',
                         color='red')
               pl.quiver(Xs / constants.nm, Zs / constants.nm,
                         5 * mx_s.real, 5 * mz_s.real, pivot='middle',
                         color='red')
               pl.quiver(Xr / constants.nm, Zr / constants.nm,
                         5 * mx_i.imag, 5 * mz_i.imag, pivot='middle',
                         color='blue')
               pl.quiver(Xs / constants.nm, Zs / constants.nm,
                         5 * mx_s.imag, 5 * mz_s.imag, pivot='middle',
                         color='blue')
               pl.xlabel(r"Position $x$, nm")
               pl.xlabel(r"Position $z$, nm")
               pl.gca().set_aspect('equal', 'box')
               pl.show()
            #I_nrf2 = I_nrf - 2.0 * Sigma_2 * omega[i] * 1j
            #R_nrf2 =  - Sigma_2 * omega[i]**2
            #iomega2, mxz2 = linalg.eig(np.dot(linalg.inv(I_nrf2), R_nrf2))
            #omega2 = iomega2 *  1j
            #i_closest = np.argmin(np.abs(omega2))
            #print ("closest ev: ", omega2[i_closest] / constants.GHz_2pi)
            self.modes.append(mode)
        print ("show modes")
        if False: pl.show()
        print ("done")
        
    def describe(self):
        return dict(
            res_Ms    = self.Ms,
            res_Bbias = self.B,
            res_alpha = self.alpha,
            res_Nx    = self.Nx,
            res_Nz    = self.Nz,
            res_Nsx   = self.Nsx,
            res_Nsz   = self.Nsz,
            res_V     = self.V,
            res_L     = self.L,
            res_H     = self.H,
            res_W     = self.W,
            res_s     = self.s,
            res_d     = self.d, 
            res_omega_0 = self.omega_0(),
            res_gamma_0 = self.gamma_0(),
            res_orientation = self.theta_or)

    def omega_0(self):
        return np.array([t.omega_0() for t in self.modes])

    def gamma_0(self):
        return np.array([t.gamma_0() for t in self.modes])
        
