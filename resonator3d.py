import numpy as np

from modes3 import CellArrayModel, Material, Area, Grid
from modes3 import H_patch, VectorField, Mode
import constants
from nearfield import do_solveNearField

class ResonantMode3D:
    def __init__ (self, L, W, H, X, Y, Z, dV, omega,
                  ma, mb, mx, my, mz, theta_or = 0.0):
        self.omega = omega
        self.ma = ma
        self.mb = mb
        self.mx = mx
        self.my = my
        self.mz = mz
        self.X = X
        self.Y = Y
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
        
    def mk_mode(self, kx, ky, q, m_vals):
        exp_kr = np.exp( - q * self.Z - 1j * kx * self.X  - 1j * ky * self.Y)
        return np.sum(self.dV * m_vals * exp_kr) #* self.W

    def Mx(self, kx, ky, q):
        return self.mk_mode(kx, ky, q, self.mx)
    
    def My(self, kx, ky, q):
        return self.mk_mode(kx, ky, q, self.my)
    
    def Mz(self, kx, ky, q):
        return self.mk_mode(kx, ky, q, self.mz)
    
    def form_factors(self, kx, ky, q):
        #return 1.0, 1.0    
        #f_x = sinc(kx * self.L / 2.0)
        #f_y = 1.0
        #f_y = sinc(ky * self.W / 2.0)
        #f_z = sinch(q * self.H / 2.0)
        #f_xyz = f_x * f_y * f_z
        return 1.0, 1.0, 1.0 #f_y, f_y, f_y
        
    def K(self, kx, ky):
        #omega_x, omega_y = self.omega_xz()
        K_vals = []
        k = np.sqrt(kx**2 + ky**2)
        #k_hor = kx * self.theta_cs + ky * self.theta_sn
        kx_1 =   kx * self.theta_cs + ky * self.theta_sn
        ky_1 =  -kx * self.theta_sn + ky * self.theta_cs
        #F_x, F_y, F_z = self.form_factors(kx_1, ky_1, k)
        #Fk_hor = F_x * kx_1 + F_y * ky_1
        Mk_hor = self.Mx(kx_1, ky_1, k) * kx_1 + self.My(kx_1, ky_1, k) * ky_1
        return (1j * Mk_hor + k * self.Mz(kx_1, ky_1, k))

    def K_1D(self, k):
        kx_1 =   k
        ky_1 =   0.0
        #F_x, F_y, F_z = self.form_factors(kx_1, ky_1, np.abs(k))
        #print ("k = ", kx_1, ky_1, "F = ", F_x, F_y, F_z)
        #Fk_hor = F_x * k
        Mk_hor  = self.Mx(kx_1, ky_1, abs(k)) * kx_1
        Mk_hor += self.My(kx_1, ky_1, abs(k)) * ky_1
        wt = 1.0 / np.sqrt(self.W)
        #print ("mx: ", self.mx)
        return (1j * Mk_hor  + np.abs(k) * self.Mz(kx_1, np.abs(k))) * wt

    def freq_and_damping(self):
        return self.omega.real, -self.omega.imag

    def omega_0(self):
        omega_, gamma_ = self.freq_and_damping()
        return omega_

    def gamma_0(self):
        omega_, gamma_ = self.freq_and_damping()
        return gamma_    
        
def make_mode_collection_3d(model, res_mask, iomega, mab, I, f_min, f_max,
                            Xr, Yr, Zr, dV_r, Lr, Wr, Hr, theta_or):

    omega = 1j * iomega

    i_pos = [t for t in range(len(omega)) if omega[t].real > 0]
    i_pos.sort(key = lambda t: omega[t].real)

    modes = []

    res_to_global = dict()
    n_res = 0
    for i in range(len(res_mask)):
        if res_mask[i] > 0:
            res_to_global[n_res] = i
            n_res += 1

    def extend_to_global(res_field):
        out = np.zeros((len(res_mask)), dtype=complex)
        for i in range(len(res_field)):
            out[res_to_global[i]] = res_field[i]
        return out
    
    for i in i_pos:
        #print ("mode freq: ", omega[i] / constants.GHz_2pi)
        if omega[i].real < f_min * constants.GHz_2pi: continue
        if omega[i].real > f_max * constants.GHz_2pi: continue
        mab_i = np.array(mab[:, i])
        C_norm = np.dot(mab_i.conj(), np.dot(I, mab_i))
        mab_i *= np.sqrt(4.0 / np.abs(C_norm))
        #mab_i *= np.sqrt(4.0 / np.sqrt(Wr))
        i_max = np.argmax(np.abs(mab_i))
        mab_i *= np.abs(mab_i[i_max]) / mab_i[i_max]
        ma_i = mab_i[0::2]
        mb_i = mab_i[1::2]
        ma_global_i = extend_to_global(ma_i)
        mb_global_i = extend_to_global(mb_i)
        #mx_i, my_i, mz_i = ma_i, 0.0 * ma_i, mb_i
        
        mxyz_global_i = model.ab_to_xyz(ma_global_i, mb_global_i)
        mx_i = mxyz_global_i.x[res_mask > 0]
        my_i = mxyz_global_i.y[res_mask > 0]
        mz_i = mxyz_global_i.z[res_mask > 0]
        theta_or = 0.0
        #def __init__ (self, L, W, H, X, Y, Z, dV, omega,
        #          ma, mb, mx, my, mz, theta_or = 0.0):
        mode = ResonantMode3D(Lr, Wr, Hr, Xr, Yr, Zr, dV_r,
                                omega[i],              
                                ma_i, mb_i, mx_i, my_i, mz_i,
                                theta_or)
        print ("append: ", mode)
        modes.append(mode)

    print ("modes:", modes)
    return modes


def solveNearField_3d(model, f_min, f_max, Lr, Wr, Hr, theta_or, *areas):
    iomega, mxz, I_nrf, res_mask = do_solveNearField(model, *areas)
    Nr = len([t for t in range(len(res_mask)) if res_mask[t] > 0])
    Xr = np.zeros((Nr))
    Yr = np.zeros((Nr))
    Zr = np.zeros((Nr))
    Xr = model.box_array.X[res_mask > 0]
    Yr = model.box_array.Y[res_mask > 0]
    Zr = model.box_array.Z[res_mask > 0]
    dV_r = model.box_array.dV()[res_mask > 0]
    
    modes = make_mode_collection_3d(model, res_mask,
                                    iomega, mxz, I_nrf,
                                    f_min, f_max, Xr, Yr, Zr, dV_r,
                                    Lr, Wr, Hr, theta_or)

    return modes

class GeometrySpec:
    def __init__ (self, L, W, H, Nx, Ny, Nz):
        self.L = L
        self.W = W
        self.H = H
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz


class MaterialSpec:
    def __init__ (self, name, Ms, Jex, alpha, gamma_s = constants.gamma_s):
        self.name = name
        self.Ms = Ms
        self.Jex = Jex,
        self.alpha = alpha
        self.gamma_s = gamma_s

class Resonator3D:
    def __init__ (self, res_geometry, res_material, B, 
                  f_min, f_max, theta_or = 0):
        L = res_geometry.L
        W = res_geometry.W
        H = res_geometry.H
        Nx = res_geometry.Nx
        Ny = res_geometry.Ny
        Nz = res_geometry.Nz
        Ms = res_material.Ms
        Jex = res_material.Jex
        alpha = res_material.alpha
        V = L * W * H
        self.L = L
        self.H = H
        self.W = W
        self.Ms = Ms
        self.B = B
        self.alpha = alpha
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.theta_or = theta_or
        res_material = Material("Resonator", Ms, Jex, alpha,
                                res_material.gamma_s)
        def mask_funct_rect(x, y, z):
            return True
        def mask_funct_ellipse(x, y, z):
            return ((2 * x/L)**2 + (2 * y / W)**2 < 1)
        mask_funct = mask_funct_ellipse
        res_area = Area("resonator",
                        Grid(-L/2.0, L/2.0, -W/2, W/2, 0, H,
                             Nx, Ny, Nz),
                        res_material, mask_funct)
        res_model = CellArrayModel()
        res_model.add_area(res_area, 0.0, B, 0.0)
        n_static = res_model.relax_magnetisation()
        self._n_static = n_static
        self._B_static = res_model.compute_static_field(n_static)
        res_model.setup_dreibein(n_static)
        result = res_model.solve()
        X, Y, Z = result['coords_all']
        dV = result['dV_all']
        self.modes = []
        for mode in result['modes']:
            if mode.f.real > f_min and mode.f.real < f_max:
                print ("append")
                mode.normalize()
                mode.scale(np.sqrt(4.0))
                ma, mb = mode.m_ab_all()
                m_xyz = res_model.ab_to_xyz(ma, mb)
                omega = mode.f * constants.GHz_2pi

                #def __init__ (self, L, W, H, X, Y, Z, dV, omega,
                #  ma, mb, mx, my, mz, theta_or = 0.0):
                res_mode = ResonantMode3D(L, W, H, X, Y, Z, dV,
                                          omega,
                                          ma, mb,
                                          m_xyz.x, m_xyz.y, m_xyz.z,
                                          theta_or)
                self.modes.append(res_mode)
    def n_static(self):
        return self._n_static
    
    def B_static(self):
        return self._B_static
    
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
    
class Resonator3DNearField:
    def __init__ (self, geometry_res, geometry_slab,
                        material_spec_res, material_spec_slab,
                        B, s,
                        f_min, f_max, theta_or = 0):
        L = geometry_res.L
        W = geometry_res.W
        H = geometry_res.H
        
        V = L * W * H
        
        self.L = L
        self.W = W
        self.H = H
        self.V = V
        self.Ms = material_spec_res.Ms
        self.B = B
        self.alpha = material_spec_res.alpha
        self.Nx = geometry_res.Nx
        self.Ny = geometry_res.Ny
        self.Nz = geometry_res.Nz
        self.Nsx = geometry_slab.Nx
        self.Nsy = geometry_slab.Ny
        self.Nsz = geometry_slab.Nz
        self.d = geometry_slab.H
        self.s = s
        self.theta_or = theta_or
        self.theta_cs = np.cos(theta_or)
        self.theta_sn = np.sin(theta_or)

        #Ws = geometry_slab.W
        Ls, Ws  = geometry_slab.L, geometry_slab.W
        slab_Ms = material_spec_slab.Ms

        def H_inf(x, y, z):
            R_n = np.array([0.0, Ws/2, -self.d/2 - self.s])
            R_s = np.array([0.0, -Ws/2, -self.d/2 - self.s])
            a_n = np.array([Ls, 0, 0])
            b_n = np.array([0, 0, self.d])
            a_s =  a_n
            b_s =  b_n
            R   = np.array([x, y, z])
            H_n = H_patch(a_n, b_n, R - R_n)
            H_s = H_patch(a_s, b_s, R - R_s)
            return (H_n - H_s) * slab_Ms * constants.mu_0 / 4.0 / np.pi
        
        def B_funct_x(x, y, z):
            return 0.0  + 1 * H_inf(x, y, z)[0]
        def B_funct_y(x, y, z):
            return B    + 1 * H_inf(x, y, z)[1]
        def B_funct_z(x, y, z):
            return 0.0  + 1 * H_inf(x, y, z)[2]
        

        
        res_material  = Material("Resonator", material_spec_res.Ms,
                                material_spec_res.Jex, material_spec_res.alpha,
                                material_spec_res.gamma_s)
        slab_material = Material("Resonator", material_spec_slab.Ms,
                                material_spec_slab.Jex,
                                material_spec_slab.alpha,
                                material_spec_slab.gamma_s)
        def mask_funct_rect(x, y, z): return True
        def mask_funct_ellipse(x, y, z):
            return (2.0 * x / L)**2 + (2.0 * y / W)**2 < 1
        res_area = Area("resonator",
                        Grid(-L/2.0, L/2.0, -W/2, W/2, 0, H,
                             self.Nx, self.Ny, self.Nz),
                        res_material, mask_funct_ellipse)
        d = geometry_slab.H
        slab_area = Area("slab",
                        Grid(-Ls/2.0, Ls/2.0, -Ws/2, Ws/2, - s - d, - s,
                             self.Nsx, self.Nsy, self.Nsz),
                        slab_material, mask_funct_rect)
        res_model = CellArrayModel()
        
        #res_model.add_area(res_area, B_funct_x, B_funct_y, B_funct_z)
        res_model.add_area(res_area, B_funct_x, B_funct_y, B_funct_z)
        res_model.add_area(slab_area, B_funct_x, B_funct_y, B_funct_z)
        n_static = res_model.relax_magnetisation()
        B_static = res_model.compute_static_field(n_static)
        self._n_static = n_static
        self._B_static = B_static
        res_model.setup_dreibein(n_static)
        self.modes = solveNearField_3d(res_model, f_min, f_max,
                                       L, W, H, theta_or,
                                       'resonator')
        
    def n_static(self): return self._n_static
    
    def B_static(self): return self._B_static
    
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
            res_s     = self.s,
            res_d     = self.d, 
            res_omega_0 = self.omega_0(),
            res_gamma_0 = self.gamma_0(),
            res_orientation = self.theta_or)
        
    def omega_0(self):
        return np.array([t.omega_0() for t in self.modes])

    def gamma_0(self):
        return np.array([t.gamma_0() for t in self.modes])

    

    
