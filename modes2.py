import numpy as np
import pylab as pl
from scipy import linalg

import constants

class Material:
    def __init__(self, name, Ms, Jex, alpha, gamma_s):
        self.name    = name
        self.Ms      = Ms
        self.Jex     = Jex
        self.alpha   = alpha
        self.gamma_s = gamma_s

    def alpha_func(self, x, z):
        return self.alpha + 0.0 * x + 0.0 * z
    
    def Jex_func(self, x, z):
        return self.Jex + 0.0 * x + 0.0 * z
    
    def Ms_func(self, x, z):
        return self.Ms + 0.0 * x + 0.0 * z
    

def extend_array(X, x_new):
    X = list(X)
    X.extend(x_new)
    return np.array(X)

class Link:
    def __init__ (self, pos_1, pos_2, x_mid, z_mid, dl,
                  material, factors):
        self.pos_1 = pos_1  # nodes to be linked
        self.pos_2 = pos_2
        self.x_mid = x_mid  # link midpoint
        self.z_mid = z_mid
        self.dl    = dl     # link length
        self.material = material
        self.Jex = 0.0      # exchange constant
        self.factors = factors 
        
    def expand_material_constants(self):
        self.Jex = self.material.Jex_func(self.x_mid, self.z_mid)

    def make_Kex(self):
        return self.Jex / self.dl**2 


class LinkArray:
    def __init__ (self):
        #self.h_links = []
        #self.v_links = []
        self.links = []

        
    def expand_material_constants(self):
        for link in self.links:
            link.expand_material_constants()
        #for link in self.v_links:
        #    link.expand_material_constants()

    def add_link(self, pos1, pos2, x_mid, z_mid, dl, material,
                 factors = np.array([[1.0, 1.0], [1.0, 1.0]])):
        self.links.append(Link(pos1, pos2, x_mid, z_mid, dl, material,
                               factors))
            
    def update(self, area):
        Nx = len(area.grid.xc)
        Nz = len(area.grid.zc)
        for i in range (Nx - 1):
            for j in range(Nz):
                pos_o = area.get_pos(i, j)
                pos_e = area.get_pos(i + 1, j)
                dx = area.grid.xc[i + 1] - area.grid.xc[i]
                x_mid = 0.5 *(area.grid.xc[i + 1] + area.grid.xc[i])
                z_mid = area.grid.zc[j]
                self.add_link(pos_o, pos_e, x_mid, z_mid, dx, area.material)
                #self.links.append(Link(pos_o, pos_e, x_mid, z_mid, dx,
                #                         area.material))
                
        for i in range (Nx):
            for j in range(Nz - 1):
                pos_o = area.get_pos(i, j)
                pos_n = area.get_pos(i, j + 1)
                dz = area.grid.zc[j + 1] - area.grid.zc[j]
                z_mid = 0.5 *(area.grid.zc[j + 1] + area.grid.zc[j])
                x_mid = area.grid.xc[i]
                self.add_link(pos_o, pos_n, x_mid, z_mid, dz, area.material)
                #self.links.append(Link(pos_o, pos_n, x_mid, z_mid, dz,
                #                         area.material))

    def compute_J_operator(self, N):
        J = np.zeros((N, N))
        #for pos_o, pos_e, dx in link_array.h_links:
        for link in self.links:
            Kex = link.make_Kex()
            J[link.pos_1, link.pos_1] += Kex * link.factors[0, 0]
            J[link.pos_1, link.pos_2] -= Kex * link.factors[0, 1]
            J[link.pos_2, link.pos_1] -= Kex * link.factors[1, 0]
            J[link.pos_2, link.pos_2] += Kex * link.factors[1, 1]
        #for link in self.v_links:
        #    Kex = link.make_Kex()
        #    J[link.pos_1, link.pos_1] += Kex
        #    J[link.pos_1, link.pos_2] -= Kex
        #    J[link.pos_2, link.pos_1] -= Kex
        #    J[link.pos_2, link.pos_2] += Kex
            #K = 1.0 / dx**2
            #J[pos_o, pos_o] +=  K
            #J[pos_o, pos_e] += -K
            #J[pos_e, pos_o] += -K
            #J[pos_e, pos_e] +=  K

        #for pos_o, pos_n, dz in link_array.v_links:
        #    K = 1.0 / dz**2
        #    J[pos_o, pos_o] +=  K
        #    J[pos_o, pos_n] += -K
        #    J[pos_n, pos_o] += -K
        #    J[pos_n, pos_n] +=  K

        return J

class BoxArray:
    def __init__ (self):
        
        self.Z  = np.array([])
        self.X  = np.array([])
        self.DX = np.array([])
        self.DZ = np.array([])
        self.areas = []
        self.area_dict = {}
        self.masks = []
        self.N = 0
        self.last_id = -1
        self.pos = []
        

    def dV(self):
        return self.DX * self.DZ

    def get_area_id(self, area_name):
        return self.area_dict[area_name]

    def get_area(self, area_name):
        return self.areas[self.get_area_id(self, area_name)]

    def get_areas(self): return self.areas

    def get_area_mask(self, area_name):
        mask =  self.masks[self.get_area_id(area_name)]
        #print ("mask for ", area_name, mask)
        return mask

    def evaluate_all (self, func):
        result = np.vectorize(func)(self.X, self.Z)
        return result

    def evaluate_in_area(self, func, area):
        Xa, Za = area.meshgrid()
        return np.vectorize(func)(Xa, Za)
        
    def integrate_over2(self, field):
        self_dV = self.dV()
        return np.sum(self_dV[:, None] * field * self_dV, axis=(0, 1))
    
    def integrate_over(self, field):
        return np.sum(field * self.dV(), axis=(0))

    def integrate_over_area(self, field, area_name):
        return self.integrate_over(field * self.get_area_mask(area_name))
    
    def integrate_over_area2(self, field, area_name):
        mask = self.get_area_mask(area_name)
        return self.integrate_over2(mask[:, None] * field * mask)

    def area_volume(self, area_name):
        return self.areas[self.get_area_id(area_name)].get_volume()
    
    def extend(self, area):
        x_new  = area.grid.xc
        z_new  = area.grid.zc
        dx_new = area.grid.dx
        dz_new = area.grid.dz

        #print ("new: ", x_new, z_new)
        cur_id = self.last_id + 1
        
        Znew, Xnew = np.meshgrid(z_new, x_new)
        DZnew, DXnew = np.meshgrid(dz_new, dx_new)

        Jnew, Inew = np.meshgrid(range(len(z_new)), range(len(x_new)))
        #print ("JInew: ", Jnew, Inew)
        Jnew = Jnew.flatten()
        Inew = Inew.flatten()
        #print ("JInew: ", Jnew, Inew)
        
        self.X  = extend_array(self.X,  Xnew.flatten())
        self.Z  = extend_array(self.Z,  Znew.flatten())
        self.DX = extend_array(self.DX, DXnew.flatten())
        self.DZ = extend_array(self.DZ, DZnew.flatten())


            
        Nboxes = len(Inew)

        self.pos.extend(zip([area] * Nboxes, Inew, Jnew))
        self.areas.append(area)
        self.area_dict[area.name] = len(self.areas) - 1
        #for t in range(Nboxes):
        positions =  np.array(range(self.N, self.N + Nboxes), dtype=int)
        area.record_positions(Inew, Jnew, positions)

        #print ("add masks")
        masks_new = []
        for mask in self.masks:
            #print ("old mask: ", mask)
            mask = extend_array(mask, np.zeros((len(Xnew.flatten()))))
            masks_new.append(mask)
            #print ("new mask of length", len(mask), mask)
        mask_cur = np.zeros(len(self.X))
        mask_cur[self.N:] = 1
        #print ("cur mask: ", len(mask_cur), mask_cur)
        #print ("len(X)", len(self.X))
        masks_new.append(mask_cur)
        self.masks = masks_new

        self.N += Nboxes
        self.last_id = cur_id
        
        return cur_id, Nboxes

class Box:
    def __init__ (self, i, j, x, z, dx, dz):
        self.i = i
        self.j = j
        self.x = x
        self.z = z
        self.dx = dx
        self.dz = dz
        self.pos = 0
        
    def set_pos(self, pos):
        self.pos = pos
    
class VariableGrid:
    def __init__ (self, xvals, zvals):
        self.x = xvals
        self.z = zvals
        self.xc = 0.5 * (self.x[1:] + self.x[:-1])
        self.zc = 0.5 * (self.z[1:] + self.z[:-1])
        self.dx = self.x[1:] - self.x[:-1]
        self.dz = self.z[1:] - self.z[:-1]
    
    def meshgrid(self):
        Z, X = np.meshgrid(self.zc, self.xc)
        return X, Z
        
    def get_volume(self):
        return (self.x[-1] - self.x[0]) * (self.z[-1] - self.z[0])

    def box(self, i, j):
        return Box(i, j, self.xc[i], self.zc[j], self.dx[i], self.dz[j])
    
    def east(self):
        return [self.box(len(self.xc) - 1, t) for t in range(len(self.zc))]
    
    def west(self):
        return [self.box(0,                t) for t in range(len(self.zc))]
    
    def south(self):
        return [self.box(t,                0) for t in range(len(self.xc))]
    
    def north(self):
        return [self.box(t, len(self.zc) - 1) for t in range(len(self.xc))]

    #def meshgrid(self):
    #    Z, X = np.meshgrid(self.zc, self.xc)
    #    return X, Z
    
    #def get_volume(self):
    #    return (self.x[-1] - self.x[0]) * (self.y[-1] - self.y[0])

class Grid(VariableGrid):
    def __init__ (self, x1, x2, z1, z2, Nx, Nz):
        xvals = np.linspace(x1, x2, Nx)
        zvals = np.linspace(z1, z2, Nz)
        VariableGrid.__init__ (self, xvals, zvals)
        #self.xc = 0.5 * (self.x[1:] + self.x[:-1])
        #self.zc = 0.5 * (self.z[1:] + self.z[:-1])
        #self.dx = self.x[1:] - self.x[:-1]
        #self.dz = self.z[1:] - self.z[:-1]

class AreaBoundary:
    def __init__ (self, area, boxes):
        self.area  = area
        self.boxes = boxes
        for box in self.boxes:
            box.set_pos(self.area.get_pos(box.i, box.j))
        

class Area:
    def __init__ (self, name, grid, material):
        self.name     = name
        self.grid     = grid
        self.material = material
        self.pos = dict()

    def meshgrid(self): return self.grid.meshgrid()
        
    def get_volume(self): return self.grid.get_volume()

    def record_positions(self, I, J, POS):
        for i, j, pos in zip(I, J, POS):
            self.pos[(i, j)] = pos
            
    def record_pos(self, i, j, pos):
        #print ("record pos", i, j, pos)
        self.pos[(i, j)] = pos

    def get_pos(self, i, j):
        #print ("get pos", i, j)
        return self.pos[(i, j)]

    
    def east(self):
        return AreaBoundary(self, self.grid.east())
    
    def west(self):
        return AreaBoundary(self, self.grid.west())
    
    def north(self):
        return AreaBoundary(self, self.grid.north())
    
    def south(self):
        return AreaBoundary(self, self.grid.south())
            
# Field induced by a vertical segment (x_a, z_a) -- (x_a, z_b)
# with unit charge density at the point (x_o, z_o)
def H_xx(x_o, z_o, x_a, z_a, z_b):
    #dx_oa = x_o - x_a
    #atan_d = np.arctan((z_o - z_b)/dx_oa) - np.arctan((z_o - z_a)/dx_oa)
    #atan_d =    np.angle( 1j * (x_o - x_a + 1j * (z_o - z_a))) - np.pi/2.0
    #atan_d +=   np.angle( 1j * (x_o - x_a + 1j * (z_o - z_b))) + np.pi/2.0
    atan_d = np.angle((x_o - x_a + 1j * (z_o - z_a))/(x_o - x_a + 1j * (z_o - z_b)))
    return atan_d

def H_zx(x_o, z_o, x_a, z_a, z_b):
    log_d  = np.log(np.abs((x_a - x_o) + 1j * (z_b - z_o)))
    log_d -= np.log(np.abs((x_a - x_o) + 1j * (z_a - z_o)))
    return -log_d

# Field induced by a horizontal segment (x_a, z_a) -- (x_b, z_a)
# with unit charge density at the point (x_o, z_o)
def H_xz(x_o, z_o, x_a, x_b, z_a):
    log_d  = np.log(np.abs((x_b - x_o) + 1j * (z_a - z_o)))
    log_d -= np.log(np.abs((x_a - x_o) + 1j * (z_a - z_o)))
    return -log_d

def H_zz(x_o, z_o, x_a, x_b, z_a):
    dx_oa = x_o - x_a
    dx_ob = x_o - x_b
    #atan_d = np.arctan(dx_ob/(z_o - z_a)) - np.arctan(dx_oa/(z_o - z_a))
    atan_d = np.angle((-1j * (x_o - x_b) + (z_o - z_a))/(-1j * (x_o - x_a) +  (z_o - z_a)))
    #atan_d  =   np.angle((x_o - x_a + 1j * (z_o - z_a))) 
    #atan_d += - np.angle(-(x_o - x_b + 1j * (z_o - z_a))) + np.pi 
    return  atan_d
        
def compute_H_operator (box_array):
    N = len(box_array.X)
    Hxx = np.zeros ((N, N))
    Hxz = np.zeros ((N, N))
    Hzx = np.zeros ((N, N))
    Hzz = np.zeros ((N, N))

    X_o = box_array.X
    Z_o = box_array.Z

    #print ("X_o, Z_o", X_o, Z_o)
    for i in range(N):
        x_c = box_array.X[i]
        z_c = box_array.Z[i]
        x_l = x_c - 0.5 * box_array.DX[i] 
        x_r = x_c + 0.5 * box_array.DX[i] 
        z_b = z_c - 0.5 * box_array.DZ[i] 
        z_t = z_c + 0.5 * box_array.DZ[i]
        
        Hxx[:, i] += H_xx(X_o, Z_o, x_r, z_b, z_t)
        Hxx[:, i] -= H_xx(X_o, Z_o, x_l, z_b, z_t)
        Hzx[:, i] += H_zx(X_o, Z_o, x_r, z_b, z_t)
        Hzx[:, i] -= H_zx(X_o, Z_o, x_l, z_b, z_t)

        Hzz[:, i] += H_zz(X_o, Z_o, x_l, x_r, z_t)
        Hzz[:, i] -= H_zz(X_o, Z_o, x_l, x_r, z_b)
        Hxz[:, i] += H_xz(X_o, Z_o, x_l, x_r, z_t)
        Hxz[:, i] -= H_xz(X_o, Z_o, x_l, x_r, z_b)

    Hxx /= 2.0 * np.pi
    Hzz /= 2.0 * np.pi
    Hxz /= 2.0 * np.pi
    Hzx /= 2.0 * np.pi
    return Hxx, Hxz, Hzx, Hzz


class Mode:
    def __init__ (self, model, f, mode_mxz, mode_mxz_dual):
        self.f = f
        self.model  = model
        self.mx_all = mode_mxz[0::2]
        self.mz_all = mode_mxz[1::2]
        self.mx_dual_all = mode_mxz_dual[0::2]
        self.mz_dual_all = mode_mxz_dual[1::2]
        self.mx = dict()
        self.mz = dict()
        self.mx_dual = dict()
        self.mz_dual = dict()
        for area in model.areas():
            mx_area = model.to_area_coord(self.mx_all, area)
            mz_area = model.to_area_coord(self.mz_all, area)
            self.mx[area.name] = mx_area
            self.mz[area.name] = mz_area
            mx_dual_area = model.to_area_coord(self.mx_dual_all, area)
            mz_dual_area = model.to_area_coord(self.mz_dual_all, area)
            self.mx_dual[area.name] = mx_dual_area
            self.mz_dual[area.name] = mz_dual_area

    def copy(self):
        mxz      = np.zeros((len(self.mx_all) + len(self.mz_all)),
                       dtype=self.mx_all.dtype)
        mxz_dual = np.zeros((len(self.mx_all) + len(self.mz_all)),
                       dtype=self.mx_all.dtype)
        mxz[0::2] = self.mx_all
        mxz[1::2] = self.mz_all
        mxz_dual[0::2] = self.mx_dual_all
        mxz_dual[1::2] = self.mz_dual_all
        mode_copy = Mode(self.model, self.f, mxz, mxz_dual)
        #print ("check copy: mx_all", linalg.norm(self.mx_all - mode_copy.mx_all))
        #print ("check copy: mz_all", linalg.norm(self.mz_all - mode_copy.mz_all))
        #for k in self.mx.keys():
        #    print ("  check x", k, linalg.norm(self.mx[k] - mode_copy.mx[k]))
        #    print ("  check z", k, linalg.norm(self.mz[k] - mode_copy.mz[k]))
        return mode_copy

    def normalize(self, area = ''):
        C_norm = self.model.dot_product(self, self, area)
        self.scale(1.0 / np.sqrt(np.abs(C_norm)))

    def participation (self, area):
        C_tot  = self.model.dot_product(self, self)
        C_area = self.model.dot_product(self, self, area)
        return np.abs(C_area / C_tot)
    
    def scale(self, scale_factor):
        self.mx_all *= scale_factor
        self.mz_all *= scale_factor
        self.mx_dual_all /= scale_factor
        self.mz_dual_all /= scale_factor
        for k in self.mx.keys():
            self.mx[k] = self.mx[k] * scale_factor
        for k in self.mz.keys():
            self.mz[k] = self.mz[k] * scale_factor
        for k in self.mx_dual.keys():
            self.mx_dual[k] = self.mx_dual[k] / scale_factor
        for k in self.mz_dual.keys():
            self.mz_dual[k] = self.mz_dual[k] / scale_factor
        
    def freq(self):
        return self.f

    def m(self, area_name):
        return self.mx[area_name], self.mz[area_name]
    
    def m_dual(self, area_name):
        return self.mx_dual[area_name], self.mz_dual[area_name]

    def m_all(self):
        return self.mx_all, self.mz_all
    
    def m_dual_all(self):
        return self.mx_dual_all, self.mz_dual_all

class CellArrayModel:
    def __init__ (self):
        self.box_array  = BoxArray()
        self.link_array = LinkArray()
        self.materials  = []
        self.Bbias      = []

    def areas(self):
        return self.box_array.get_areas()
        
    def get_area(self, area_name):
        return self.box_array.get_area(area_name)
    
    def get_area_mask(self, area_name):
        return self.box_array.get_area_mask(area_name)

    def add_area(self, area, Bbias):
        area_id, Nboxes = self.box_array.extend(area)
        self.link_array.update(area)
        print ("add area: nboxes = ", Nboxes)
        print ("material: ", area.material, area.material.name)
        self.materials.extend ([area.material] * Nboxes)
        self.Bbias.extend([Bbias] * Nboxes)

    def connect(self, boundary1, boundary2, Jex, M1, M2):
        if len(boundary1.boxes) != len(boundary2.boxes):
            raise Exception("Connecting boundaries of different length "
                            "is not implemented")
        name1 = boundary1.area.name
        name2 = boundary2.area.name
        joint_name = name1 + ":" + name2

        box1 = boundary1.boxes[0]
        box2 = boundary2.boxes[0]
        box1_c = box1.x + 1j * box1.z
        box2_c = box2.x + 1j * box2.z
        dl = np.abs(box1_c - box2_c)
        J_eff = Jex * dl 
        joint_material = Material(joint_name, 0.0, J_eff, 0.0, 0.0)
        r1 = 1.0
        r2 = 1.0
        if    abs(box1.x - box2.x) < 1e-4: # vertical link
              r1 = np.abs(box1.dz) / dl
              r2 = np.abs(box2.dz) / dl
        elif  abs(box1.z - box1.z) < 1e-4:
              r1 = np.abs(box1.dx) / dl
              r2 = np.abs(box2.dx) / dl
        else:
            raise Exception("cannot connect boxes"
                            "across a tilted boundary:"
                            "not implemented")
                   

        #diag1 = M2 / M1
        #diag2 = M1 / M2
        factors = np.zeros((2, 2))
        factors[0, 0] = M2 / M1 / r1
        factors[1, 1] = M1 / M2 / r2
        factors[0, 1] = 1.0 / r1
        factors[1, 0] = 1.0 / r2
        for box1, box2 in zip(boundary1.boxes, boundary2.boxes):
            x_mid = 0.5 * (box1.x + box2.x)
            z_mid = 0.5 * (box1.z + box2.z)
            self.link_array.add_link(box1.pos, box2.pos, x_mid, z_mid, dl,
                                     joint_material, factors)

    def dot_product(self, mode1, mode2, area = ''):
        mx1, mz1 = mode1.m_all()
        mx2, mz2 = mode2.m_all()
        integrand = mz1.conjugate() * mx2 - mx1.conjugate() * mz2
        integrand /= self.gamma_s * self.Ms * 1j
        if area == '':
            return self.integrate_over(integrand)
        else:
            return self.integrate_over_area(integrand, area)

    def evaluate_all (self, func):
        return self.box_array.evaluate_all(func)

    def evaluate_in_area (self, func, area_name):
        return self.box_array.evaluate_in_area(func, self.get_area(area_name))

    def integrate(self, field, area = ''):
        if type(area) == type([]):
           result = 0.0 + 0.0j
           for area_item in area:
               result += self.integrate_over_area(field, area_item)
           return result
        if type(area) == type(''):
            if area == '': return self.integrate_over(field)
            return self.integrate_over_area(field, area)
        raise Exception("Unknown area specification: ", area)
        
    def integrate2(self, field, area = ''):
        if type(area) == type([]):
           result = 0.0 + 0.0j
           for area_item in area:
               result += self.integrate_over_area2(field, area_item)
           return result
        if type(area) == type(''):
            if area == '': return self.integrate_over2(field)
            return self.integrate_over_area2(field, area)
        raise Exception("Unknown area specification: ", area)

    def integrate_over2(self, field):
        return self.box_array.integrate_over2(field)
    
    def integrate_over(self, field):
        return self.box_array.integrate_over(field)

    def integrate_over_area(self, field, area_name):
        return self.box_array.integrate_over_area(field, area_name)
    
    def integrate_over_area2(self, field, area_name):
        return self.box_array.integrate_over_area2(field, area_name)

    def expand_material_constants(self):

        X = self.box_array.X
        Z = self.box_array.Z
        self.alpha   = np.array([self.materials[t].alpha_func(X[t], Z[t])
                                 for t in range(len(self.materials))])
        #self.alpha = self.alpha_func(self.box_a) 
        self.gamma_s = np.array([t.gamma_s for t in self.materials])
        self.Ms      = np.array([t.Ms      for t in self.materials])
        if  False:
            import pylab as pl
            pl.plot(self.box_array.Z, self.Ms)
        #self.Jex     = np.array([t.Jex     for t in self.materials])
        self.Bbias   = np.array(self.Bbias)
        self.link_array.expand_material_constants()

    def compute_ext_source(self, h_func):
        self.expand_material_constants()

        Ms = self.Ms
        gamma_s = self.gamma_s
        sgn_M = 1.0 + 0.0 * self.Ms
        sgn_M[self.Ms < 0] = -1.0
        alpha = self.alpha * sgn_M
        N = self.box_array.N
        L0s = np.zeros((2 * N), dtype=complex)
        hx, hz = h_func(self.box_array.X, self.box_array.Z)
        print ("shapes: hx, hz", np.shape(hx), np.shape(hz))
        print ("Ms: ", np.shape(Ms))
        L0s[0::2] += - Ms * constants.mu_0 * hz
        L0s[1::2] +=   Ms * constants.mu_0 * hx
        L0s[0::2] *= gamma_s
        L0s[1::2] *= gamma_s
        a0 = 1.0 / (1.0 + alpha * alpha)
        sgn_M = Ms / np.abs(Ms)
        a1 = alpha * a0 #* sgn_M
        Ls = np.zeros((2 * N), dtype=complex)
        Ls[0::2] = a0 *  L0s[0::2] + a1 * L0s[1::2]
        Ls[1::2] = -a1 * L0s[0::2] + a0 * L0s[1::2]

        return Ls

    def get_I(self):
        N = self.box_array.N
        I = np.zeros((2 * N, 2 * N))
        self.expand_material_constants()
        dV = self.box_array.dV()
        I_zx = dV / self.Ms / self.gamma_s
        I[1::2, 0::2] =   np.diag(I_zx)
        I[0::2, 1::2] = - np.diag(I_zx)
        return I
    
    def get_I_alpha(self):
        N = self.box_array.N
        I = np.zeros((2 * N, 2 * N))
        self.expand_material_constants()
        dV = self.box_array.dV()
        I_zx = dV / self.Ms / self.gamma_s
        I_zx_alpha = dV / np.abs(self.Ms) / self.gamma_s * self.alpha 
        I[1::2, 0::2] =   np.diag(I_zx)
        I[0::2, 1::2] = - np.diag(I_zx)
        I[0::2, 0::2] =    -np.diag(I_zx * self.alpha)
        I[1::2, 1::2] =    -np.diag(I_zx * self.alpha)
        return I

        
    def compute_LLG_operator(self):

        self.expand_material_constants()
        if  False:
            import pylab as pl
            pl.figure()
            pl.tripcolor(self.box_array.X, self.box_array.Z,
                         self.Ms)
            pl.colorbar()
            pl.show()
        sgn_M = 1.0 + 0.0 * self.Ms
        sgn_M[self.Ms < 0] = -1.0
        alpha   = self.alpha * sgn_M
        gamma_s = self.gamma_s
        Ms = self.Ms
        #Jex = self.Jex
        Bbias = self.Bbias
    
        Hxx, Hxz, Hzx, Hzz = compute_H_operator(self.box_array)
        J = self.link_array.compute_J_operator(self.box_array.N)
        
        N = len(Ms)
        for area_id in range(len(self.box_array.areas)):
          area_name =  self.box_array.areas[area_id].name
          print ("Area: ", area_name)
          #area_mask = self.box_array.masks[area_id]
          area_vol = self.box_array.areas[area_id].get_volume()
          box = self.box_array
          #print ("volume: ", area_vol)
          #print ("integrate one: ",
          #       self.integrate(np.ones(len(Hxx)), area_name) / area_vol)
          #one = np.ones(len(Hxx))
          #one2 = np.outer(one, one)
          dV = self.box_array.dV() 
          #print ("integrate one2: ",
          #       self.integrate2(one2, area_name) / area_vol**2)
          #print ("  demag factors xx:",
          #       self.integrate2(Hxx/dV, area_name) / area_vol)
          #print ("  demag factors zz:",
          #       self.integrate2(Hzz/dV, area_name) / area_vol)
          #print ("  demag factors xz:",
          #       self.integrate2(Hxz/dV, area_name) / area_vol)
          #print ("  demag factors zx:",
          #       self.integrate2(Hzx/dV, area_name) / area_vol)

        #print ("bias: ", Bbias)
        #print ("Ms: ", Ms)
        #print ("Jex", Jex)
        L0 = np.zeros((2 * N, 2 * N))
        I  = np.eye((N))
        L0[0::2, 1::2] +=   Bbias * I + Ms[:, None] * J
        L0[1::2, 0::2] += - Bbias * I - Ms[:, None] * J
        L0[0::2, 0::2] += - Ms[:, None] * constants.mu_0 * Hzx
        L0[0::2, 1::2] += - Ms[:, None] * constants.mu_0 * Hzz
        L0[1::2, 0::2] +=   Ms[:, None] * constants.mu_0 * Hxx
        L0[1::2, 1::2] +=   Ms[:, None] * constants.mu_0 * Hxz
        L0[0::2, :] *= gamma_s[:, None]
        L0[1::2, :] *= gamma_s[:, None]

        L0 *= 1.0 #/ constants.GHz_2pi

        L = np.zeros((2 * N, 2 * N))
        a0 = 1.0 / (1.0 + alpha * alpha)
        sgn_M = Ms / np.abs(Ms)
        a1 = alpha * a0 # * sgn_M
        a0 = a0[:, None]
        a1 = a1[:, None]
        L[0::2, 0::2] = a0 * L0[0::2, 0::2]  + a1 * L0[1::2, 0::2]
        L[0::2, 1:: 2] = a0 * L0[0::2, 1::2]  + a1 * L0[1::2, 1::2]
        L[1::2, 0::2] = -a1 * L0[0::2, 0::2] + a0 * L0[1::2, 0::2]
        L[1::2, 1::2] = -a1 * L0[0::2, 1::2] + a0 * L0[1::2, 1::2]
        
        return L

    def solve(self):
        L = self.compute_LLG_operator()
        I_alpha = self.get_I_alpha()
        I_s = 0.5 *  (I_alpha - np.transpose(I_alpha))
        iomega, mxz_l, mxz = linalg.eig(L, left=True, right=True)
        mxz_dual = np.dot(mxz_l.transpose().conj(), linalg.inv(I_alpha))
        H = np.dot(I_alpha, L)
        if True:
           #I = self.get_I_alpha()
           IL = np.dot(I_alpha, L)
           print ("Hermiticity: ", linalg.norm(IL - np.transpose(IL).conj()))
        f = 1j * iomega / constants.GHz_2pi
        f_all = list(f)
        f_all.sort(key = lambda x: np.abs(x))
        print ("all freqs: ", f_all[0:20])
        modes_pos = [t for t in range(len(f)) if f[t].real > 0]
        modes_pos.sort(key = lambda t: f[t].real)
        f_pos = np.array([f[t] for t in modes_pos])
        
        modes_unpacked = []
        m_dual_prev = []
        for t in modes_pos:
            print ("append mode", t, f[t])
            mxz_t = mxz[:, t]
            mxz_dual_t = mxz_dual[t, :]
            diff   = np.dot(H, mxz_t)    - iomega[t] * np.dot(I_alpha, mxz_t)
            diff /= linalg.norm(mxz_t)
            diff_d = np.dot(mxz_dual_t, H) - iomega[t] * np.dot(mxz_dual_t, I_alpha)
            diff_d /= linalg.norm(mxz_dual_t)
            #print ("verify: ", linalg.norm(diff), linalg.norm(diff_d))
            C_norm = np.dot(mxz_t.conj(), np.dot(I_s, mxz_t))
            mxz_t *= 1.0 / np.sqrt(np.abs(C_norm))
            mm = np.dot(mxz_dual_t, np.dot(I_alpha, mxz_t))
            mxz_dual_t *= 1.0 / mm * 4.0 * 1j
            #print ("<dual * orig> = ",
            #       np.dot(mxz_dual_t, np.dot(I_alpha, mxz_t)))
            modes_unpacked.append(Mode(self, f[t], mxz_t, mxz_dual_t))
            for m_prev in m_dual_prev:
                pass
                #print ("verify _|_: ",
                #       np.abs(np.dot(m_prev, np.dot(I_alpha, mxz_t))))
            m_dual_prev.append(mxz_dual_t)
            
        coords = dict()
        for area in self.areas():
            coords[area.name] = self.area_coords(area)

        area_names = list([t.name for t in self.areas()])

        result = dict()
        result['area_names'] = area_names
        result['modes'] = modes_unpacked
        result['coords'] = coords
        result['coords_all'] = self.box_array.X, self.box_array.Z
        result['dV_all'] = self.box_array.dV()
        result['frequencies'] = f_pos

        return result
    
    def solve_response(self, omegas, h_func):
        L = self.compute_LLG_operator()
        S = self.compute_ext_source(h_func)
        N = self.box_array.N
        I = np.eye(2 * N)
        modes = []
        for omega in omegas:
            print ("solve f = ", omega / constants.GHz_2pi)
            mxz = linalg.solve( -1j * omega * I - L, S)
            mxz_dual = mxz.conj()
            #print ("done")
            mode = Mode(self, omega / constants.GHz_2pi, mxz, mxz_dual)
            modes.append(mode)

        area_names = list([t.name for t in self.areas()])

        coords = dict()
        for area in self.areas():
            coords[area.name] = self.area_coords(area)
            
        result = dict()
        result['area_names'] = area_names
        result['modes'] = modes
        result['coords'] = coords
        result['coords_all'] = self.box_array.X, self.box_array.Z
        result['dV_all'] = self.box_array.dV()

        return result
        
    def area_coords(self, area):
        return area.meshgrid()
        
    def to_area_coord(self, field, area):
        Nx = len(area.grid.xc)
        Nz = len(area.grid.zc)
        FIELD = np.zeros ((Nx, Nz), dtype=field.dtype)
        for i in range(Nx):
            for j in range(Nz):
                FIELD[i, j] = field[area.get_pos(i, j)]
        return FIELD
        
if __name__ == '__main__':

    YIG_Ms = 140 * constants.kA_m
    YIG_alpha = 1 * 0.001
    YIG_gamma_s = constants.gamma_s
    Aex = 2 * 3.5 * constants.pJ_m
    YIG_Jex = Aex / YIG_Ms**2
    YIG = Material("YIG", YIG_Ms, YIG_Jex, YIG_alpha, YIG_gamma_s)

    Bext = 5 * constants.mT
    a = 200 * constants.nm
    #a = 50 * constants.nm
    b = 30 * constants.nm
    #b = 20 * constants.nm
    Nx = 50
    Nz = 7

    def test_bare_resonator():
        resonator = Area ("resonator", Grid(-a/2, a/2, -b/2, b/2, Nx, Nz), YIG)
        model = CellArrayModel()
        model.add_area (resonator, Bext)
        result = model.solve()

        
        box = model.box_array

        XR, ZR = result['coords']['resonator']
        X, Z = result['coords_all']
        
        def show_mode(mode):
            pl.figure()
            mx, mz = mode.m('resonator')
            mx_all, mz_all = mode.m_all()
            pl.quiver(X, Z,
                      mx_all.real, mz_all.real, pivot='middle')
            pl.gca().set_aspect('equal', 'box')
            pl.title("omega = %g %g" % (mode.f.real, mode.f.imag))
            pl.xlim(-a - 5 * b, a + 5 * b)
            pl.figure()
            mxmax = np.max(np.abs(mx))
            mzmax = np.max(np.abs(mz))
            re = model.integrate(np.abs(mx_all.real)**2
                                 + np.abs(mz_all.real)**2, "resonator")
            im = model.integrate(np.abs(mx_all.imag)**2
                                 + np.abs(mz_all.imag)**2, "resonator")
            print ("ell = ", np.sqrt(np.abs(im/re)))
            ell2  = model.integrate(np.abs(mz_all)**2, 'resonator')
            ell2 /= model.integrate(np.abs(mx_all)**2, 'resonator')
            print ("ell2 = ", np.sqrt(ell2))

            pl.pcolormesh(XR, ZR,
                          mx.real, cmap='bwr', vmin=-mxmax, vmax=mxmax)
            pl.gca().set_aspect('equal', 'box')
            pl.colorbar()
            pl.title("Re m_x")

            if False:
                pl.figure()
                pl.pcolormesh(XR, ZR,
                              mz.real, cmap='bwr', vmin=-mzmax, vmax=mzmax)
                pl.gca().set_aspect('equal', 'box')
                pl.colorbar()
                pl.title("Re m_z")

                pl.figure()
                pl.pcolormesh(XR, ZR,
                              mx.imag, cmap='bwr', vmin=-mxmax, vmax=mxmax)
                pl.gca().set_aspect('equal', 'box')
                pl.colorbar()
                pl.title("Im m_x")

            pl.figure()
            pl.pcolormesh(XR, ZR, 
                          mz.imag, cmap='bwr', vmin=-mzmax, vmax=mzmax)
            pl.gca().set_aspect('equal', 'box')
            pl.colorbar()
            pl.title("Im m_z")

            if False:
                pl.figure()
                pl.plot(XR[:, 0], mx[:, 0].real, label='Re Mx')
                pl.plot(XR[:, 0], mx[:, 0].imag, label='Im Mx')
                pl.plot(XR[:, 0], mz[:, 0].real, label='Re Mz')
                pl.plot(XR[:, 0], mz[:, 0].imag, label='Im Mz')
                pl.legend()
                pl.title ("Bottom face")

            if False:
                pl.figure()
                pl.plot(XR[:, -1], mx[:, -1].real, label='Re Mx')
                pl.plot(XR[:, -1], mx[:, -1].imag, label='Im Mx')
                pl.plot(XR[:, -1], mz[:, -1].real, label='Re Mz')
                pl.plot(XR[:, -1], mz[:, -1].imag, label='Im Mz')
                pl.legend()
                pl.title ("top face")

            pl.figure()
            i0 = np.argmin(np.abs(XR[:, 0] - 0.5 * (XR[0, 0] + XR[-1, 0])))
            j0 = np.argmin(np.abs(ZR[0, :] - 0.5 * (ZR[0, 0] + ZR[0, -1])))
            print ("i0, j0 = ", i0, j0)
            pl.plot(XR[:, j0], mx[:, j0].real, label='Re Mx')
            pl.plot(XR[:, j0], mx[:, j0].imag, label='Im Mx')
            pl.plot(XR[:, j0], mz[:, j0].real, label='Re Mz')
            pl.plot(XR[:, j0], mz[:, j0].imag, label='Im Mz')
            pl.legend()
            pl.title ("Centerline, f = %g" % mode.f.real)

            pl.figure()
            theta_ell = np.linspace(0.0, 2.0 * np.pi, 1001)
            mx_ell = mx[i0, j0] * np.exp(1j * theta_ell)
            mz_ell = mz[i0, j0] * np.exp(1j * theta_ell)
            pl.plot(mx_ell.real, mz_ell.real, label='centre')
            mx_ell = mx[0, j0] * np.exp(1j * theta_ell)
            mz_ell = mz[0, j0] * np.exp(1j * theta_ell)
            pl.plot(mx_ell.real, mz_ell.real, '.', ms=1.0, label='left')
            mx_ell = mx[-1, j0] * np.exp(1j * theta_ell)
            mz_ell = mz[-1, j0] * np.exp(1j * theta_ell)
            pl.plot(mx_ell.real, mz_ell.real, '--', label='right')
            pl.gca().set_aspect('equal', 'box')
            pl.title("ellipticity, f = %g" % mode.f.real)
            pl.legend()
            
        for mode in result['modes']:
            print ("mode: ", mode.f)
            #show_mode(mode)
            #pl.show()
            if mode.f.real > 5.0: break
        pl.show()


    def test_response():

        from scipy import optimize
        def f_wave_fit(x, k, A, B):
            return A * np.exp(1j * k * x) + B * np.exp(-1j * k * x)
        
        def get_amplitudes(x_a, x_b, x, psi, k_min = 0.0, k_max = 2.0):
            i_fit   = [t for t in range(len(x)) if x[t] > x_a and x[t] < x_b]
            x_fit   = np.array([x[t]   for t in i_fit])
            psi_fit = np.array([psi[t] for t in i_fit])
            psi_fit /= np.max(np.abs(psi))

            if False:
                from scipy.fft import fft, fftfreq, fftshift
                fft_psi  = fft(psi_fit)
                fft_freq = fftfreq( len(fft_psi), x_fit[1] - x_fit[0] )
                fft_freq = fftshift( fft_freq )
                fft_psi  = fftshift( fft_psi )
                pl.figure()
                pl.semilogy(fft_freq * 2.0 * np.pi, np.abs(fft_psi)**2)
                fft_max = np.argmax(np.abs(fft_psi))
                k_fft_max = fft_freq[fft_max] * 2.0 * np.pi
                print ("k_fft_max = ", k_fft_max)
                if k_fft_max < 0: k_fft_max *= -1
                print ("k_fft_max = ", k_fft_max)
            
        
            def f_opt(x):
                k = x[0]; A = x[1] + 1j * x[2]; B = x[3] + 1j * x[4]
                #psi_test = A * np.exp(1j * k * x_fit) + B * np.exp(-1j * k * x_fit)
                return np.sum(np.abs(f_wave_fit(x_fit, k, A, B) - psi_fit)**2)

            k_approx = 0.5 * (k_min + k_max)
            x0 = np.array([k_approx, 1.0, 0.0, 0.0, 1.0])
            result = optimize.minimize(f_opt, x0, tol=1e-10, method='Powell',
                                         bounds=((k_min, k_max),
                                                 (-3, 3), (-3, 3), (-3, 3), (-3, 3)))
            print ("fit: ", result) 
            k_fit, A_fit_re, A_fit_im, B_fit_re, B_fit_im = result['x']
            A_fit = A_fit_re + 1j * A_fit_im
            B_fit = B_fit_re + 1j * B_fit_im
            
            if k_fit < 0:
                A_fit, B_fit = B_fit, A_fit
                k_fit *= -1
            return k_fit, A_fit, B_fit

        
        print ("test wave fit")
        x_test = np.linspace(-np.pi, np.pi, 101)
        k = 1.17
        y_test = 0.57 * np.exp(1j * k * x_test) + 0.41 * np.exp(-1j * k * x_test) + 0.01 * np.cos(2*k*x_test)
        print (get_amplitudes(-1.0, 2.5, x_test, y_test))
        
        s = 50 * constants.nm
        d = 20 * constants.nm
        W = a + 400 * b
        Wd = 50 * constants.nm
        #Nsx = 1000
        Nsx = 600
        Nsz = 5
        w_off = 0
        resonator      = Area ("resonator", Grid(-a/2, a/2, 0.0, b, Nx, Nz), YIG)
        resonator_bare = Area ("resonator", Grid(-a/2, a/2, 0.0, b, Nx, Nz), YIG)
        YIG_d = Material("YIG", YIG_Ms, YIG_Jex, 0.0001, YIG_gamma_s)
        W_round = W/2 - W / 10
        K_round = 1.0 / (W/2 - W_round) 
        def alpha_slab(x, z):
            if np.abs(x) < W_round: return 0.0001
            return 0.0001 + 0.25 * (np.abs(x) - W_round) * K_round 
        YIG_d.alpha_func = alpha_slab

        slab = Area("slab", Grid(-W/2 + w_off, W/2 + w_off, -s - d, -s, Nsx, Nsz), YIG_d)
        #damp_l = Area("damp-l", Grid(-W/2 - Wd, -W/2, -s - d, -s, 3, 3), YIG_d)
        #damp_r = Area("damp-r", Grid( W/2, W/2 + Wd, -s - d, -s, 3, 3), YIG_d)
        model_bare = CellArrayModel()
        model_bare.add_area(resonator_bare, Bext)
        
        model = CellArrayModel()
        model.add_area (resonator, Bext)
        model.add_area (slab,      Bext)
        #model.add_area (damp_l,    Bext)
        #model.add_area (damp_r,    Bext)

        h0 = 1.0
        x0 = - W_round
        sigma = 5.0 * constants.nm
        def h_src_fwd(x, z):
            exp_x = np.exp(-(x - x0)**2 / 2.0 / sigma**2)
            return h0 * exp_x, 0.0 * exp_x
    
        from constants import GHz_2pi
        #omega_tab = np.linspace(1.0, 4.5, 351) * constants.GHz_2pi
        #omega_tab = np.linspace(2.9, 3.3, 201) * constants.GHz_2pi
        omega_tab = np.linspace(3.0, 3.3, 151) * constants.GHz_2pi
        #omega_tab = np.linspace(2.9, 3.3, 5) * constants.GHz_2pi
        #omega_tab = np.linspace(3.8, 4.2, 201) * constants.GHz_2pi
        #omega_tab = np.linspace(3.8, 4.2, 5) * constants.GHz_2pi
        #omega_tab = np.array([3.61]) * constants.GHz_2pi
        #omega_tab = np.linspace(1.0, 4.0, 301) * constants.GHz_2pi
        #omega_tab = np.array([1.0, 1.3, 1.5, 1.7, 1.9, 2.0, 2.2, 2.5, 2.7, 3.0, 3.1, 3.2, 3.5, 4.0]) * GHz_2pi

        x_in_a  = -W/4
        x_in_b  = -5 * (s + b)
        x_out_a =  5 * (s + b)
        x_out_b =  W_round - 3 * b

        from scipy import optimize

        
        #for omega in omega_tab:
        result = model.solve_response(omega_tab, h_src_fwd)


        T_o = []
        R_o = []
        mode_s_mx = []
        mode_s_mz = []
        mode_r_mx = []
        mode_r_mz = []
        xs = result['coords']['slab'][0][:, -1]
        xr = result['coords']['resonator'][0][:, 0]
        for i_omega, omega in enumerate(omega_tab):
            XS, ZS = result['coords']['slab']
            XR, ZR = result['coords']['resonator']
            MX_s, MZ_s = result['modes'][i_omega].m('slab')
            MX_r, MZ_r = result['modes'][i_omega].m('resonator')
            mode_s_mx.append(MX_s[:, -1])
            mode_s_mz.append(MZ_s[:, -1])
            mode_r_mx.append(MX_r[:, 0])
            mode_r_mz.append(MZ_r[:, 0])
            n_changes = 0
            for i in range(len(XS[:, -1]) - 1):
                if (MX_s[i, -1].real) * (MX_s[i + 1, -1].real) < 0: n_changes += 1
            if n_changes > 0:
               k_min = np.pi * (n_changes - 1) / (XS[-1, -1] - XS[0, -1])
            else:
               k_min = 0.0
            k_max =  np.pi * (n_changes + 1) / (XS[-1, -1] - XS[0, -1])
            print ("k_approx: ", 0.5 * (k_min + k_max))
            k_inc, A_inc, B_inc = get_amplitudes(x_in_a, x_in_b, XS[:, -1], MX_s[:, -1], k_min, k_max)
            k_out, A_out, B_out = get_amplitudes(x_out_a, x_out_b, XS[:, -1], MX_s[:, -1], k_min, k_max)
            print ("incident: ", k_inc, A_inc, B_inc)
            print ("transmitted: ", k_out, A_out, B_out)
            print ("Transmission: ", A_out / A_inc, "reflection: ", B_inc / A_inc)
            T_o.append(A_out / A_inc)
            R_o.append(B_inc / A_inc)
            continue
            mx_max = max(np.max(np.abs(MX_r[:, -1])), np.max(np.abs(MX_s[:, 0])))
            pl.figure()
            pl.pcolormesh(XR, ZR, np.abs(MX_r)**2, cmap='magma', vmin=0.0, vmax=mx_max)
            pl.gca().set_aspect('equal', 'box')
            pl.colorbar()
            pl.pcolormesh(XS, ZS, np.abs(MX_s)**2, cmap='magma', vmin=0.0, vmax=mx_max)
            pl.xlim(-a/2 - 2 * s - 3 * b, a/2 + 2 * s + 3 * b)
            pl.gca().set_aspect('equal', 'box')
            pl.title("f = %g" % (omega / GHz_2pi))
            pl.figure()
            pl.plot(XS[:, -1], MX_s[:, -1].real, label='Re mx')
            pl.plot(XS[:, -1], MX_s[:, -1].imag, label='Im mx')
            pl.plot(XS[:, -1], MZ_s[:, -1].real, label='Re mz')
            pl.plot(XS[:, -1], MZ_s[:, -1].imag, label='Im mz')
            x_in = np.linspace(x_in_a, x_in_b, 100)
            x_out = np.linspace(x_out_a, x_out_b, 100)
            psi_inc = f_wave_fit(x_in,  k_inc, A_inc, B_inc)
            psi_out = f_wave_fit(x_out, k_out, A_out, B_out)
            #psi_inc = A_inc * np.exp(1j * k_inc * x_in) + B_inc * np.exp(-1j * k_inc * x_in)
            #psi_out = A_out * np.exp(1j * k_out * x_out) + B_out * np.exp(-1j * k_out * x_in)
            pl.plot(x_in, psi_inc.real, '--')
            pl.plot(x_in, psi_inc.imag, '--')
            pl.plot(x_out, psi_out.real, '--')
            pl.plot(x_out, psi_out.imag, '--')
            pl.legend()
            pl.show()
        np.savez("TR-response-1way-a=%gnm-s=%gnm-%g-%gGHz-B=%g.npz" % (a / constants.nm, s / constants.nm, omega_tab[0] / GHz_2pi,
                                                                       omega_tab[-1] / GHz_2pi, Bext / constants.mT),
                 omega=omega_tab, T=np.array(T_o), R=np.array(R_o),
                 Bext = Bext, a = a, b = b, s = s, d = d, xs = xs, xr = xr,
                 mode_s_mx = np.array(mode_s_mx), mode_s_mz = np.array(mode_s_mz),
                 mode_r_mx = np.array(mode_r_mx), mode_r_mz = np.array(mode_r_mz))
        
        def h_src_bk(x, z):
            exp_x = np.exp(-(x + x0)**2 / 2.0 / sigma**2)
            return h0 * exp_x, 0.0 * exp_x

        result = model.solve_response(omega_tab, h_src_bk)
        
        x_in_b  =  W/4
        x_in_a  =  5 * (s + b)
        x_out_b =  -5 * (s + b)
        x_out_a =  - W_round + 3 * b
        T_o_bk = []
        R_o_bk = []
        mode_s_mx_bk = []
        mode_s_mz_bk = []
        mode_r_mx_bk = []
        mode_r_mz_bk = []
        #xs = result['coords']['slab'][0][:, -1]
        #xr = result['coords']['resonator'][0][:, 0]
        for i_omega, omega in enumerate(omega_tab):
            XS, ZS = result['coords']['slab']
            XR, ZR = result['coords']['resonator']
            MX_s, MZ_s = result['modes'][i_omega].m('slab')
            MX_r, MZ_r = result['modes'][i_omega].m('resonator')
            mode_s_mx_bk.append(MX_s[:, -1])
            mode_s_mz_bk.append(MZ_s[:, -1])
            mode_r_mx_bk.append(MX_r[:, 0])
            mode_r_mz_bk.append(MZ_r[:, 0])
            n_changes = 0
            for i in range(len(XS[:, -1]) - 1):
                if (MX_s[i, -1].real) * (MX_s[i + 1, -1].real) < 0: n_changes += 1
            if n_changes > 0:
               k_min = np.pi * (n_changes - 1) / (XS[-1, -1] - XS[0, -1])
            else:
               k_min = 0.0
            k_max =  np.pi * (n_changes + 1) / (XS[-1, -1] - XS[0, -1])
            print ("k_approx: ", 0.5 * (k_min + k_max))
            k_inc, A_inc, B_inc = get_amplitudes(x_in_a, x_in_b, XS[:, -1], MX_s[:, -1], k_min, k_max)
            k_out, A_out, B_out = get_amplitudes(x_out_a, x_out_b, XS[:, -1], MX_s[:, -1], k_min, k_max)
            print ("incident: ", k_inc, A_inc, B_inc)
            print ("transmitted: ", k_out, A_out, B_out)
            print ("Transmission: ", B_out / B_inc, "reflection: ", A_inc / B_inc)
            T_o_bk.append(B_out / B_inc)
            R_o_bk.append(A_inc / B_inc)
            continue
            mx_max = max(np.max(np.abs(MX_r[:, -1])), np.max(np.abs(MX_s[:, 0])))
            pl.figure()
            pl.pcolormesh(XR, ZR, np.abs(MX_r)**2, cmap='magma', vmin=0.0, vmax=mx_max)
            pl.gca().set_aspect('equal', 'box')
            pl.colorbar()
            pl.pcolormesh(XS, ZS, np.abs(MX_s)**2, cmap='magma', vmin=0.0, vmax=mx_max)
            pl.xlim(-a/2 - 2 * s - 3 * b, a/2 + 2 * s + 3 * b)
            pl.gca().set_aspect('equal', 'box')
            pl.title("f = %g" % (omega / GHz_2pi))
            pl.figure()
            pl.plot(XS[:, -1], MX_s[:, -1].real, label='Re mx')
            pl.plot(XS[:, -1], MX_s[:, -1].imag, label='Im mx')
            pl.plot(XS[:, -1], MZ_s[:, -1].real, label='Re mz')
            pl.plot(XS[:, -1], MZ_s[:, -1].imag, label='Im mz')
            x_in = np.linspace(x_in_a, x_in_b, 100)
            x_out = np.linspace(x_out_a, x_out_b, 100)
            psi_inc = f_wave_fit(x_in,  k_inc, A_inc, B_inc)
            psi_out = f_wave_fit(x_out, k_out, A_out, B_out)
            #psi_inc = A_inc * np.exp(1j * k_inc * x_in) + B_inc * np.exp(-1j * k_inc * x_in)
            #psi_out = A_out * np.exp(1j * k_out * x_out) + B_out * np.exp(-1j * k_out * x_in)
            pl.plot(x_in, psi_inc.real, '--')
            pl.plot(x_in, psi_inc.imag, '--')
            pl.plot(x_out, psi_out.real, '--')
            pl.plot(x_out, psi_out.imag, '--')
            pl.legend()
            pl.show()
        np.savez("TR-response-2way-a=%gnm-s=%gnm-%g-%gGHz-B=%g.npz" % (a / constants.nm, s / constants.nm, omega_tab[0] / GHz_2pi,
                                                                       omega_tab[-1] / GHz_2pi, Bext / constants.mT),
                 omega=omega_tab,
                 T=np.array(T_o), R=np.array(R_o),
                 T_bk=np.array(T_o_bk), R_bk=np.array(R_o_bk),
                 Bext = Bext, 
                 a = a, b = b, s = s, d = d, xs = xs, xr = xr,
                 mode_s_mx = np.array(mode_s_mx), mode_s_mz = np.array(mode_s_mz),
                 mode_r_mx = np.array(mode_r_mx), mode_r_mz = np.array(mode_r_mz),
                 mode_s_mx_bk = np.array(mode_s_mx_bk), mode_s_mz_bk = np.array(mode_s_mz_bk),
                 mode_r_mx_bk = np.array(mode_r_mx_bk), mode_r_mz_bk = np.array(mode_r_mz_bk))

        
    def test_coupled_resonator():
        s = 20 * constants.nm
        d = 20 * constants.nm
        W = a + 200 * b
        Wd = 50 * constants.nm
        Nsx = 200
        Nsz = 5
        w_off = 0
        resonator      = Area ("resonator", Grid(-a/2, a/2, 0.0, b, Nx, Nz), YIG)
        resonator_bare = Area ("resonator", Grid(-a/2, a/2, 0.0, b, Nx, Nz), YIG)
        YIG_d = Material("YIG", YIG_Ms, YIG_Jex, 0.0001, YIG_gamma_s)
        W_round = W/2 - W / 6
        K_round = 1.0 / (W/2 - W_round) 
        #def alpha_slab(x, z):
        #    if np.abs(x) < W_round: return 0.0001
        #    return 0.0001 + 0.05 * (np.abs(x) - W_round) * K_round 
        #YIG_d.alpha_func = alpha_slab

        slab = Area("slab", Grid(-W/2 + w_off, W/2 + w_off, -s - d, -s, Nsx, Nsz), YIG_d)
        #damp_l = Area("damp-l", Grid(-W/2 - Wd, -W/2, -s - d, -s, 3, 3), YIG_d)
        #damp_r = Area("damp-r", Grid( W/2, W/2 + Wd, -s - d, -s, 3, 3), YIG_d)
        model_bare = CellArrayModel()
        model_bare.add_area(resonator_bare, Bext)
        
        model = CellArrayModel()
        model.add_area (resonator, Bext)
        model.add_area (slab,      Bext)
        #model.add_area (damp_l,    Bext)
        #model.add_area (damp_r,    Bext)

        result_bare = model_bare.solve()
        
        result = model.solve()        
        box = model.box_array
        XR, ZR = result['coords']['resonator']
        XS, ZS = result['coords']['slab']
        X, Z = result['coords_all']
        #pl.figure()
        #pl.plot(XS[:, 0], np.vectorize(YIG_d.alpha_func)(XS[:, 0], ZS[:, 0]))
        #pl.title("alpha(x)")
        
        box = model.box_array
        #mode_set = []

        #def dot_product(mode1, mode2, area_name):
        #    mx1, mz1 = mode1.m_all()
        #    mx2, mz2 = mode2.m_all()
        #    C1   = box.integrate_over_area(mz1.conjugate() * mx2, area_name)
        #    #print ("C1 = ", C1)
        #    C1  -= box.integrate_over_area(mx1.conjugate() * mz2, area_name)
        #    #print ("C1 = ", C1)
        #    return C1
        
        def compute_mode_overlap(mode_set):
            mode_norm = []
            for mode in mode_set:
                mx_i, mz_i = mode.m_all()
                C_res = model.dot_product(mode, mode, "resonator")
                #print ("C_res = ", C_res)

                mode_copy = mode.copy()
                mode_copy.scale(1.0 / np.sqrt(np.abs(C_res)))
                mode_norm.append(mode_copy)

            C_ij = np.zeros ((len(mode_norm), len(mode_norm)), dtype=complex)
            S_ij = np.zeros ((len(mode_norm), len(mode_norm)), dtype=complex)
            for i in range(len(mode_norm)):
                for j in range(len(mode_norm)):
                    C_ij[i, j] = model.dot_product(mode_norm[i], mode_norm[j], "resonator")
                    S_ij[i, j] = model.dot_product(mode_norm[i], mode_norm[j], "slab")
            print ("mode matrix: ", np.abs(C_ij))
            print ("mode matrix for slab: ", np.abs(S_ij))
            return mode_norm, C_ij, S_ij

        def clusterise_modes(modes_norm, C_ij):
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

            return clusters

        def analyze_cluster(cluster, mode_norm, C_ij, S_ij):
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
                part_c = mode_c.participation('resonator')
                f_av += part_c * mode_c.f
                part_tot += part_c
                f_vals.append(mode_c.f)
                p_vals.append(part_c)
            f_av /= part_tot
            p_vals = np.array(p_vals)
            f_vals = np.array(f_vals)
            print ("average f: ", f_av)
            return f_vals, p_vals, C_c, S_c
                
        def analyze_clusters(clusters, mode_norm, C_ij, S_ij):
            for cluster in clusters:
                analyze_cluster(cluster, mode_norm, C_ij, S_ij)

            

        XR, ZR = result['coords']['resonator']
        XS, ZS = result['coords']['slab']
        X, Z = result['coords_all']

        mode_set = []
        
        def show_mode(mode):
            mx, mz = mode.m_all()
            C_total = model.integrate(mz.conjugate() * mx).imag
            #C_res   = model.integrate(mz.conjugate() * mx, "resonator").imag
            ones    = model.evaluate_all(lambda x, y: 1.0)
            vol_tot = model.integrate(ones)
            vol_res = model.integrate(ones, "resonator")
            vol_ratio = vol_res / vol_tot
            #participation =  np.abs(C_res / C_total) * vol_tot / vol_res
            participation =  mode.participation('resonator') / vol_ratio
            print ("participation: ", participation, "f = ", mode.f)
            if participation < 0.1: return

            mode.normalize('resonator')
            #mode.scale(1.0/np.sqrt(np.abs(C_res)))
            mx, mz = mode.m_all()
            #mode_set.append(mode)
            #compute_mode_matrix()  
            re = model.integrate(np.abs(mx.real)**2
                                 + np.abs(mz.real)**2, "resonator")
            im = model.integrate(np.abs(mx.imag)**2
                                 + np.abs(mz.imag)**2, "resonator")
            print ("ell = ", np.sqrt(np.abs(im/re)))
            ell2  = model.integrate(np.abs(mz)**2, 'resonator')
            ell2 /= model.integrate(np.abs(mx)**2, 'resonator')
            print ("ell2 = ", np.sqrt(ell2))

            dx_scale = min(abs(resonator.grid.dx[0]),
                           abs(resonator.grid.dz[0]))
            mxmax = np.max(np.abs(mx))
            mzmax = np.max(np.abs(mz))
            m_scale = max (mxmax, mzmax)
            q_scale = m_scale / dx_scale * 1.2
            
            pl.figure()
            pl.quiver(X, Z,
                      mx.real, mz.real,
                      scale = q_scale, scale_units='x',
                      pivot='middle', color='red')
            pl.quiver(X, Z,
                      mx.imag, mz.imag,
                      scale = q_scale, scale_units='x', 
                      pivot='middle', color='blue')
            pl.xlim(-a/2 - 5 * b, a/2 + 5 * b)
            pl.gca().set_aspect('equal', 'box')
            pl.title("omega = %g %g p = %g" % (mode.f.real,
                                               mode.f.imag,
                                               participation))

            pl.figure()
            MXS, MZS = mode.m("slab")
            MXR, MZR = mode.m("resonator")

            pl.pcolormesh(XS, ZS, np.abs(MXS), cmap='magma',
                          vmin=0.0, vmax=mxmax)
            pl.pcolormesh(XR, ZR, np.abs(MXR), cmap='magma',
                          vmin=0.0, vmax=mxmax)
            pl.xlim(-a - 5 * b, a + 5 * b)
            pl.ylim(-s - b/2, b)
            pl.gca().set_aspect('equal', 'box')
            pl.colorbar()

            if False:
                pl.figure()
                pl.pcolormesh(XS, ZS, np.abs(MZS), cmap='magma',
                              vmin=0.0, vmax=mzmax)
                pl.pcolormesh(XR, ZR, np.abs(MZR), cmap='magma',
                              vmin=0.0, vmax=mzmax)
                pl.colorbar()
                pl.gca().set_aspect('equal', 'box')

            pl.figure()
            pl.pcolormesh(XR, ZR,
                          MXR.real, cmap='bwr', vmin=-mxmax, vmax=mxmax)
            pl.gca().set_aspect('equal', 'box')
            pl.colorbar()
            pl.title("Re m_x")

            if False:
                pl.figure()
                pl.pcolormesh(XR, ZR,
                              MZR.real, cmap='bwr', vmin=-mzmax, vmax=mzmax)
                pl.gca().set_aspect('equal', 'box')
                pl.colorbar()
                pl.title("Re m_z")


            if False:
                pl.figure()
                pl.pcolormesh(XR, ZR,
                              MXR.imag, cmap='bwr', vmin=-mxmax, vmax=mxmax)
                pl.gca().set_aspect('equal', 'box')
                pl.colorbar()
                pl.title("Im m_x")

            pl.figure()
            pl.pcolormesh(XR, ZR, 
                          MZR.imag, cmap='bwr', vmin=-mzmax, vmax=mzmax)
            pl.gca().set_aspect('equal', 'box')
            pl.colorbar()
            pl.title("Im m_z")


            if False:
                pl.figure()
                pl.plot(XR[:, 0], MXR[:, 0].real, label='Re Mx')
                pl.plot(XR[:, 0], MXR[:, 0].imag, label='Im Mx')
                pl.plot(XR[:, 0], MZR[:, 0].real, label='Re Mz')
                pl.plot(XR[:, 0], MZR[:, 0].imag, label='Im Mz')
                pl.legend()
                pl.title ("Bottom face")

                pl.figure()
                pl.plot(XR[:, -1], MXR[:, -1].real, label='Re Mx')
                pl.plot(XR[:, -1], MXR[:, -1].imag, label='Im Mx')
                pl.plot(XR[:, -1], MZR[:, -1].real, label='Re Mz')
                pl.plot(XR[:, -1], MZR[:, -1].imag, label='Im Mz')
                pl.legend()
                pl.title ("top face")

            pl.figure()
            i0 = np.argmin(np.abs(XR[:, 0] - 0.5 * (XR[0, 0] + XR[-1, 0])))
            j0 = np.argmin(np.abs(ZR[0, :] - 0.5 * (ZR[0, 0] + ZR[0, -1])))
            print ("i0, j0 = ", i0, j0)
            pl.plot(XR[:, j0], MXR[:, j0].real, label='Re Mx')
            pl.plot(XR[:, j0], MXR[:, j0].imag, label='Im Mx')
            pl.plot(XR[:, j0], MZR[:, j0].real, label='Re Mz')
            pl.plot(XR[:, j0], MZR[:, j0].imag, label='Im Mz')
            pl.legend()
            pl.title ("Centerline")

            pl.figure()
            pl.plot(XS[:, -1], MXS[:, -1].real, label='Re m_x')
            pl.plot(XS[:, -1], MXS[:, -1].imag, label='Im m_x')
            pl.plot(XS[:, -1], MZS[:, -1].real, label='Re m_z')
            pl.plot(XS[:, -1], MZS[:, -1].imag, label='Im m_z')
            pl.legend()
            pl.title("top of the slab")

            pl.figure()
            theta_ell = np.linspace(0.0, 2.0 * np.pi, 1001)
            mx_ell = MXR[i0, j0] * np.exp(1j * theta_ell)
            mz_ell = MZR[i0, j0] * np.exp(1j * theta_ell)
            pl.plot(mx_ell.real, mz_ell.real, label='centre')
            mx_ell = MXR[0, j0] * np.exp(1j * theta_ell)
            mz_ell = MZR[0, j0] * np.exp(1j * theta_ell)
            pl.plot(mx_ell.real, mz_ell.real, '.', ms=1.0, label='left')
            mx_ell = MXR[-1, j0] * np.exp(1j * theta_ell)
            mz_ell = MZR[-1, j0] * np.exp(1j * theta_ell)
            pl.plot(mx_ell.real, mz_ell.real, '--', label='right')
            pl.gca().set_aspect('equal', 'box')
            pl.legend()
            pl.show()

        #f_vals = []
        #p_vals = []
        ones    = model.evaluate_all(lambda x, y: 1.0)
        vol_tot = model.integrate(ones)
        vol_res = model.integrate(ones, "resonator")
        vol_ratio = vol_res / vol_tot
        for mode in result['modes']:
            #show_mode(mode)
            mode_set.append(mode)
            #pl.show()
            #f_vals.append(mode.f)
            #p_vals.append(mode.participation('resonator'))
            if mode.f.real > 5.0: break

        mode_norm, C_modes, S_modes = compute_mode_overlap(mode_set)
        clusters = clusterise_modes(mode_norm, C_modes)

        modes_bare = []
        for mode in result_bare['modes']:
            if mode.f.real > 5.0: break
            print ("mode freq: ", mode.f)
            mode.normalize()
            modes_bare.append(mode)
        
        #f_vals = np.array(f_vals)
        #p_vals = np.array(p_vals)

        def overlap2d(mode, mode_bare):
            MX, MZ = mode.m('resonator')
            MX_bare, MZ_bare = mode_bare.m('resonator')
            S_12 = np.sum(MZ.conjugate() * MX_bare - MX.conjugate() * MZ_bare, axis=(0, 1)) / 1j
            S_11 = np.sum(MZ.conjugate() * MX - MX.conjugate() * MZ, axis=(0, 1)) / 1j
            S_22 = np.sum(MZ_bare.conjugate() * MX_bare - MX_bare.conjugate() * MZ_bare, axis=(0, 1)) / 1j
            return S_12 / np.sqrt(S_11 * S_22)
        
        pl.figure()
        for cluster in clusters:
            f_vals, p_vals, C_cluster, S_cluster = analyze_cluster(cluster, mode_norm, C_modes, S_modes)
            pl.plot(f_vals.real, p_vals / vol_ratio, '-o', ms=3.0)
            for mode_bare in modes_bare:
                print ("Bare mode @ ", mode_bare.f)
                for c in cluster:
                    ovr = overlap2d(mode_norm[c], mode_bare)
                    print ("  overlap with bare mode: ", ovr)
        pl.title("p(f) vs f")


        pl.figure()
        for mode_bare in modes_bare:
            ovr  = []
            freq = []
            for mode in mode_norm:
                ovr.append(np.abs(overlap2d(mode, mode_bare))**2)
                freq.append(mode.f.real)
            pl.plot(freq, ovr, label='bare mode %g' % mode_bare.f.real)
        pl.legend()
        pl.title("overlap with each of the bare modes")
        #pl.show()
        #pl.plot(f_vals.real, p_vals / vol_ratio, '-o', ms=3.0)
        #pl.show()

        clusters_bare = []
        for mode_bare in modes_bare:
            clusters_bare.append([])

        for i_mode, mode in enumerate(mode_norm):
            overlaps = []
            for mode_bare in modes_bare:
                overlaps.append(overlap2d(mode, mode_bare))
            overlaps = np.array(overlaps)
            i_bare = np.argmax(np.abs(overlaps))
            clusters_bare[i_bare].append((i_mode, overlaps[i_bare]))

        pl.figure()
        for i_bare, cluster_bare in enumerate(clusters_bare):
            freqs = np.array([mode_norm[c[0]].f.real for c in cluster_bare])
            overlaps = np.array([overlap2d(mode_norm[c[0]], modes_bare[i_bare]) for c in cluster_bare])
            pl.plot(freqs, np.abs(overlaps), label='overlap with mode @ %g' % modes_bare[i_bare].f.real)
        pl.legend()
        pl.title("overlap for clusterised modes")

        
        pl.figure()
        for i_bare, cluster_bare in enumerate(clusters_bare):
            freqs = np.array([mode_norm[c[0]].f.real for c in cluster_bare])
            participation = np.array([mode_norm[c[0]].participation("resonator") for c in cluster_bare])
            pl.plot(freqs, participation, label='cluster around @ %g' % modes_bare[i_bare].f.real)
        pl.legend()
        pl.title("participation for clusterised modes")
        pl.show()

    #test_bare_resonator()
    test_coupled_resonator()
    #test_response()


    



