import numpy as np
import pylab as pl
from scipy import linalg
import constants

def meshgrid3(z, y, x):
    Nx = len(x)
    Ny = len(y)
    Nz = len(z)
    Zm, Ym = np.meshgrid(z, y)
    X = np.zeros((Nx, Ny, Nz))
    Y = np.zeros((Nx, Ny, Nz))
    Z = np.zeros((Nx, Ny, Nz))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                X[i, j, k] = x[i]
                Y[i, j, k] = y[j]
                Z[i, j, k] = z[k]
        #X[i, :, :] = x[i]
        #Y[i, :, :] = Ym
        #Z[i, :, :] = Zm
    #print ("X = ", X, "Y = ", Y, "Z = ", Z)
    return Z, Y, X
    
def dot(a_x, a_y, a_z, b_x, b_y, b_z):
    return a_x * b_x + a_y * b_y + a_z * b_z
    
def cross(a_x, a_y, a_z, b_x, b_y, b_z):
    c_x = a_y * b_z - a_z * b_y
    c_y = a_z * b_x - a_x * b_z
    c_z = a_x * b_y - a_y * b_x
    return c_x, c_y, c_z

class VectorField:
    def __init__ (self, *args):
        if len(args) == 1:
            self.N = args[0]
            self.x = np.zeros((self.N))
            self.y = np.zeros((self.N))
            self.z = np.zeros((self.N))
        elif len(args) == 3:
            self.N = len(args[0])
            self.x = np.array(args[0])
            self.y = np.array(args[1])
            self.z = np.array(args[2])
        else:
            raise Exception("VectorField.__init__ requires one or three argsx")
    
    def dot(self, v):
        return dot(self.x, self.y, self.z, v.x, v.y, v.z)

    def cross(self, v):
        r_x, r_y, r_z = cross(self.x, self.y, self.z, v.x, v.y, v.z)
        return VectorField(r_x, r_y, r_z)

    def scale(self, factor):
        self.x *= factor
        self.y *= factor
        self.z *= factor

    def normalize(self, value = 1.0):
        r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        self.scale( value / r)

    def __add__ (self, other):
        return VectorField(self.x + other.x,
                           self.y + other.y,
                           self.z + other.z)
    
    def __sub__ (self, other):
        return VectorField(self.x - other.x,
                           self.y - other.y,
                           self.z - other.z)
    
    def __mul__ (self, other):
        return VectorField(self.x * other,
                           self.y * other,
                           self.z * other)
    
    def __truediv__ (self, other):
        return VectorField(self.x / other,
                           self.y / other,
                           self.z / other)

    def copy(self):
        return VectorField(self.x.copy(),
                           self.y.copy(),
                           self.z.copy())

class ScalarKernel:
    def __init__ (self, N):
        self.N = N
        self.K = np.zeros((N, N))

    def scale(self, factor):
        self.K *= factor

    def apply(self, field):
        return np.dot(self.K, field)
    
        
class TensorKernel:
    def __init__ (self, N):
        self.N = N
        self.xx = np.zeros((N, N))
        self.xy = np.zeros((N, N))
        self.xz = np.zeros((N, N))
        self.yx = np.zeros((N, N))
        self.yy = np.zeros((N, N))
        self.yz = np.zeros((N, N))
        self.zx = np.zeros((N, N))
        self.zy = np.zeros((N, N))
        self.zz = np.zeros((N, N))

    def diag(self, Kxx, Kyy, Kzz):
        self.xx = Kxx
        self.yy = Kyy
        self.zz = Kzz
        self.xy[:] = 0.0
        self.xz[:] = 0.0
        self.yx[:] = 0.0
        self.yz[:] = 0.0
        self.zx[:] = 0.0
        self.zy[:] = 0.0

    def scale(self, factor):
        self.xx *= factor
        self.xy *= factor
        self.xz *= factor
        self.yx *= factor
        self.yy *= factor
        self.yz *= factor
        self.zx *= factor
        self.zy *= factor
        self.zz *= factor

    def dot2(self, u, v):
        Kv = self.apply(v)
        return np.sum(u.dot(Kv))
        
    def apply(self, v_f):
        result = VectorField(self.N)
        result.x += np.dot(self.xx, v_f.x)
        result.x += np.dot(self.xy, v_f.y)
        result.x += np.dot(self.xz, v_f.z)
        
        result.y += np.dot(self.yx, v_f.x)
        result.y += np.dot(self.yy, v_f.y)
        result.y += np.dot(self.yz, v_f.z)

        result.z += np.dot(self.zx, v_f.x)
        result.z += np.dot(self.zy, v_f.y)
        result.z += np.dot(self.zz, v_f.z)

        return result

    def set_row_x(self, i, v):
        self.xx[i, :] = v.x
        self.xy[i, :] = v.y
        self.xz[i, :] = v.z
        
    def set_row_y(self, i, v):
        self.yx[i, :] = v.x
        self.yy[i, :] = v.y
        self.yz[i, :] = v.z
        
    def set_row_z(self, i, v):
        self.zx[i, :] = v.x
        self.zy[i, :] = v.y
        self.zz[i, :] = v.z
        
    def set_col_x(self, i, v):
        self.xx[:, i] = v.x
        self.yx[:, i] = v.y
        self.zx[:, i] = v.z
        
    def set_col_y(self, i, v):
        self.xy[:, i] = v.x
        self.yy[:, i] = v.y
        self.zy[:, i] = v.z
        
    def set_col_z(self, i, v):
        self.xz[:, i] = v.x
        self.yz[:, i] = v.y
        self.zz[:, i] = v.z
        
        
    def row_x(self, i):
        return VectorField(self.xx[i, :], self.xy[i, :], self.xz[i, :])

    def row_y(self, i):
        return VectorField(self.yx[i, :], self.yy[i, :], self.yz[i, :])
    
    def row_z(self, i):
        return VectorField(self.zx[i, :], self.zy[i, :], self.zz[i, :])
        
    def col_x(self, i):
        return VectorField(self.xx[:, i], self.yx[:, i], self.zx[:, i])
    
    def col_y(self, i):
        return VectorField(self.xy[:, i], self.yy[:, i], self.zy[:, i])

    def col_z(self, i):
        return VectorField(self.xz[:, i], self.yz[:, i], self.zz[:, i])
    
class Material:
    def __init__(self, name, Ms, Jex, alpha, gamma_s):
        self.name    = name
        self.Ms      = Ms
        self.Jex     = Jex
        self.alpha   = alpha
        self.gamma_s = gamma_s

    def alpha_func(self, x, y, z):
        return self.alpha + 0.0 * x + 0.0 * y + 0.0 * z
    
    def Jex_func(self, x, y, z):
        return self.Jex + 0.0 * x + 0.0 * y + 0.0 * z
    
    def Ms_func(self, x, y, z):
        return self.Ms + 0.0 * x + 0.0 * y + 0.0 * z

def extend_array(X, x_new):
    X = list(X)
    X.extend(x_new)
    return np.array(X)

class Link:
    def __init__ (self, pos_1, pos_2, x_mid, y_mid, z_mid, dl,
                  material, factors):
        self.pos_1 = pos_1  # nodes to be linked
        self.pos_2 = pos_2
        self.x_mid = x_mid  # link midpoint
        self.y_mid = y_mid
        self.z_mid = z_mid
        self.dl    = dl     # link length
        self.material = material
        self.Jex = 0.0      # exchange constant
        self.factors = factors 
        
    def expand_material_constants(self):
        self.Jex = self.material.Jex_func(self.x_mid, self.y_mid, self.z_mid)

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

    def add_link(self, pos1, pos2, x_mid, y_mid, z_mid, dl, material,
                 factors = np.array([[1.0, 1.0], [1.0, 1.0]])):
        self.links.append(Link(pos1, pos2, x_mid, y_mid, z_mid, dl, material,
                               factors))
            
    def update(self, area):
        Nx = len(area.grid.xc)
        Ny = len(area.grid.yc)
        Nz = len(area.grid.zc)
        for i in range (Nx - 1):
            for j in range(Ny):
                for k in range(Nz): 
                  pos_o = area.get_pos(i, j, k)
                  pos_e = area.get_pos(i + 1, j, k)
                  if pos_o < 0 or pos_e < 0: continue
                  dx = area.grid.xc[i + 1] - area.grid.xc[i]
                  x_mid = 0.5 *(area.grid.xc[i + 1] + area.grid.xc[i])
                  y_mid = area.grid.yc[j]
                  z_mid = area.grid.zc[k]
                  self.add_link(pos_o, pos_e, x_mid, y_mid, z_mid,
                              dx, area.material)
                #self.links.append(Link(pos_o, pos_e, x_mid, z_mid, dx,
                #                         area.material))
                
        for i in range (Nx):
            for j in range(Ny - 1):
              for k in range(Nz):
                pos_o = area.get_pos(i, j, k)
                pos_n = area.get_pos(i, j + 1, k)
                if pos_o < 0 or pos_n < 0: continue
                dy = area.grid.yc[j + 1] - area.grid.yc[j]
                y_mid = 0.5 *(area.grid.yc[j + 1] + area.grid.yc[j])
                x_mid = area.grid.xc[i]
                z_mid = area.grid.zc[k]
                self.add_link(pos_o, pos_n, x_mid, y_mid, z_mid, dy,
                              area.material)
        for i in range (Nx):
            for j in range(Ny):
              for k in range(Nz - 1):
                pos_o = area.get_pos(i, j, k)
                pos_t = area.get_pos(i, j, k + 1)
                if pos_o < 0 or pos_t < 0: continue
                dz = area.grid.zc[k + 1] - area.grid.zc[k]
                z_mid = 0.5 *(area.grid.zc[k + 1] + area.grid.zc[k])
                x_mid = area.grid.xc[i]
                y_mid = area.grid.yc[j]
                self.add_link(pos_o, pos_t, x_mid, y_mid, z_mid, dz,
                              area.material)

    def compute_J_operator(self, N):
        J = ScalarKernel(N)
        for link in self.links:
            Kex = link.make_Kex()
            J.K[link.pos_1, link.pos_1] += Kex * link.factors[0, 0]
            J.K[link.pos_1, link.pos_2] -= Kex * link.factors[0, 1]
            J.K[link.pos_2, link.pos_1] -= Kex * link.factors[1, 0]
            J.K[link.pos_2, link.pos_2] += Kex * link.factors[1, 1]
        return J

class BoxArray:
    def __init__ (self):
        
        self.X  = np.array([])
        self.Y  = np.array([])
        self.Z  = np.array([])
        self.DX = np.array([])
        self.DY = np.array([])
        self.DZ = np.array([])
        self.areas = []
        self.area_dict = {}
        self.masks = []
        self.N = 0
        self.last_id = -1
        self.pos = []
        

    def dV(self):
        return self.DX * self.DY * self.DZ

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
        result = np.vectorize(func)(self.X, self.Y, self.Z)
        return result

    def evaluate_in_area(self, func, area):
        Xa, Ya, Za = area.meshgrid()
        return np.vectorize(func)(Xa, Ya, Za)

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
        y_new  = area.grid.yc
        z_new  = area.grid.zc
        dx_new = area.grid.dx
        dy_new = area.grid.dy
        dz_new = area.grid.dz

        #print ("new: ", x_new, z_new)
        cur_id = self.last_id + 1
        
        Znew,  Ynew,  Xnew  = meshgrid3(z_new,  y_new,  x_new)
        DZnew, DYnew, DXnew = meshgrid3(dz_new, dy_new, dx_new)

        Knew, Jnew, Inew = meshgrid3(range(len(z_new)), range(len(y_new)),
                                     range(len(x_new)))
        #print ("JInew: ", Jnew, Inew)
        Knew = Knew.flatten()
        Jnew = Jnew.flatten()
        Inew = Inew.flatten()
        #print ("JInew: ", Jnew, Inew)

        to_extend = [t for t in range(len(Knew))
                     if area.belongs(int(Inew[t]), int(Jnew[t]), int(Knew[t]))]
        Xnew  = Xnew.flatten()
        Ynew  = Ynew.flatten()
        Znew  = Znew.flatten()
        DXnew = DXnew.flatten()
        DYnew = DYnew.flatten()
        DZnew = DZnew.flatten()
        
        X_extend = np.array([Xnew[t] for t in to_extend])
        Y_extend = np.array([Ynew[t] for t in to_extend])
        Z_extend = np.array([Znew[t] for t in to_extend])
        DX_extend = np.array([DXnew[t] for t in to_extend])
        DY_extend = np.array([DYnew[t] for t in to_extend])
        DZ_extend = np.array([DZnew[t] for t in to_extend])
        I_extend  = np.array([Inew[t] for t in to_extend])
        J_extend  = np.array([Jnew[t] for t in to_extend])
        K_extend  = np.array([Knew[t] for t in to_extend])
        self.X  = extend_array(self.X,  X_extend)
        self.Y  = extend_array(self.Y,  Y_extend)
        self.Z  = extend_array(self.Z,  Z_extend)
        self.DX = extend_array(self.DX, DX_extend)
        self.DY = extend_array(self.DY, DY_extend)
        self.DZ = extend_array(self.DZ, DZ_extend)


            
        Nboxes = len(I_extend)

        self.pos.extend(zip([area] * Nboxes, I_extend, J_extend, K_extend))
        self.areas.append(area)
        self.area_dict[area.name] = len(self.areas) - 1
        #for t in range(Nboxes):
        positions =  np.array(range(self.N, self.N + Nboxes), dtype=int)
        area.record_positions(I_extend, J_extend, K_extend, positions)

        #print ("add masks")
        masks_new = []
        for mask in self.masks:
            #print ("old mask: ", mask)
            mask = extend_array(mask, np.zeros((len(X_extend))))
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
    def __init__ (self, i, j, k, x, y, z, dx, dy, dz):
        self.i = i
        self.j = j
        self.k = k
        self.x = x
        self.y = y
        self.z = z
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.pos = 0
        
    def set_pos(self, pos):
        self.pos = pos
    
class VariableGrid:
    def __init__ (self, xvals, yvals, zvals):
        self.x = xvals
        self.y = yvals
        self.z = zvals
        self.xc = 0.5 * (self.x[1:] + self.x[:-1])
        self.yc = 0.5 * (self.y[1:] + self.y[:-1])
        self.zc = 0.5 * (self.z[1:] + self.z[:-1])
        self.dx = self.x[1:] - self.x[:-1]
        self.dy = self.y[1:] - self.y[:-1]
        self.dz = self.z[1:] - self.z[:-1]
    
    def meshgrid(self):
        Z, Y, X = meshgrid3(self.zc, self.yc, self.xc)
        return X, Y, Z
        
    def get_volume(self):
        dx = self.x[-1] - self.x[0]
        dy = self.y[-1] - self.y[0]
        dz = self.z[-1] - self.z[0]
        return dx * dy * dz

    def box(self, i, j, k):
        return Box(i, j, k, self.xc[i], self.yc[j], self.zc[k],
                   self.dx[i], self.dy[j], self.dz[k])
    
    def east(self):
        return [self.box(len(self.xc) - 1, t, u) for t in range(len(self.yc))
                for u in range (len(zc))]
    
    def west(self):
        return [self.box(0,                t, u) for t in range(len(self.yc))
                for u in range(len(self.zc))]
    
    def south(self):
        return [self.box(t,                0, u) for t in range(len(self.xc))
                for u in range(len(self.zc))]
    
    def north(self):
        return [self.box(t, len(self.yc) - 1, u) for t in range(len(self.xc))
                for u in range(len(self.zc))]
    
    def top(self):
        return [self.box(t, u, len(zc) - 1) for t in range(len(self.xc))
                                            for u in range(len(yc))]
    def bottom(self):
        return [self.box(t, u, 0) for t in range(len(self.xc))
                                            for u in range(len(yc))]
    

    #def meshgrid(self):
    #    Z, X = np.meshgrid(self.zc, self.xc)
    #    return X, Z
    
    #def get_volume(self):
    #    return (self.x[-1] - self.x[0]) * (self.y[-1] - self.y[0])

class Grid(VariableGrid):
    def __init__ (self, x1, x2, y1, y2, z1, z2, Nx, Ny, Nz):
        xvals = np.linspace(x1, x2, Nx)
        yvals = np.linspace(y1, y2, Ny)
        zvals = np.linspace(z1, z2, Nz)
        VariableGrid.__init__ (self, xvals, yvals, zvals)
        #self.xc = 0.5 * (self.x[1:] + self.x[:-1])
        #self.zc = 0.5 * (self.z[1:] + self.z[:-1])
        #self.dx = self.x[1:] - self.x[:-1]
        #self.dz = self.z[1:] - self.z[:-1]

class AreaBoundary:
    def __init__ (self, area, boxes):
        self.area  = area
        self.boxes = boxes
        for box in self.boxes:
            box_pox = self.area.get_pos(box.i, box.j, box.k)
            if box_pos < 0: continue
            box.set_pos(box_pos)
        

class Area:
    def __init__ (self, name, grid, material,
                  mask_func = lambda x, y, z: True):
        self.name     = name
        self.grid     = grid
        self.material = material
        self.pos = dict()
        self.mask_func = mask_func

        if  False:
            mask_f = open("mask-%s.dat" % self.name, "w")
            #Z, Y, X = self.meshgrid()
            #Nx, Ny, Nz = np.shape(X)
            Nx = len(self.grid.xc)
            Ny = len(self.grid.yc)
            Nz = len(self.grid.zc)
            k0 = Nz // 2
            for j in range(Ny):
                s = ""
                for i in range(Nx):
                    if self.belongs(i, j, k0): s += "1"
                    else:                      s += "0"
                mask_f.write(s + "\n")
            mask_f.close()
            

    def meshgrid(self): return self.grid.meshgrid()
        
    def get_volume(self): return self.grid.get_volume()

    def belongs(self, i, j, k):
        #print ("belong:", i, j, k)
        x_i = self.grid.xc[i]
        y_j = self.grid.yc[j]
        z_k = self.grid.zc[k]
        return self.mask_func(x_i, y_j, z_k)

    def record_positions(self, I, J, K, POS):
        for i, j, k, pos in zip(I, J, K, POS):
            self.pos[(i, j, k)] = pos
            
    def record_pos(self, i, j, k, pos):
        #print ("record pos", i, j, pos)
        self.pos[(i, j, k)] = pos

    def get_pos(self, i, j, k):
        #print ("get pos", i, j)
        if (i, j, k) not in self.pos.keys(): return -1
        return self.pos[(i, j, k)]

    
    def east(self):
        return AreaBoundary(self, self.grid.east())
    
    def west(self):
        return AreaBoundary(self, self.grid.west())
    
    def north(self):
        return AreaBoundary(self, self.grid.north())
    
    def south(self):
        return AreaBoundary(self, self.grid.south())

def r(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)

#
# Solution for a face perpendicular to z
# x, y are varied
#
#
def H0_x(x, y, z):
    return  np.log((y + r(x, y, z))) #/np.sqrt(x**2 + z**2))

def H0_y(x, y, z):
    return  np.log((x + r(x, y, z))) #/np.sqrt(y**2 + z**2))

def H0_z(x, y, z):
    s = x / np.sqrt(x**2 + z**2) * y / np.sqrt(y**2 + z**2)
    t = x * y / z / r(x, y, z)
    #sgn_z = np.sign(z)
    return - np.arctan(t) #* sgn_z

def H_patch(a, b, R):
    a_b_orth = np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))
    a_x, a_y, a_z = a[0], a[1], a[2]
    b_x, b_y, b_z = b[0], b[1], b[2]
    R_x, R_y, R_z = R[0], R[1], R[2]
    if a_b_orth > 1e-6: raise Exception("a and b must be orthogonal")
    c_x, c_y, c_z = cross(a_x, a_y, a_z, b_x, b_y, b_z)
    c = np.array([c_x, c_y, c_z])
    e_a = a / np.sqrt(np.dot(a, a))
    e_b = b / np.sqrt(np.dot(b, b))
    e_c = c / np.sqrt(np.dot(c, c))
    wid = np.sqrt(np.dot(a, a))
    hei = np.sqrt(np.dot(b, b))
    R_a = np.dot(R, e_a)
    R_b = np.dot(R, e_b)
    R_c = np.dot(R, e_c)

    H_a  = H0_x(R_a + 0.5 * wid, R_b + 0.5 * hei, R_c)
    H_a += H0_x(R_a - 0.5 * wid, R_b - 0.5 * hei, R_c)
    H_a -= H0_x(R_a + 0.5 * wid, R_b - 0.5 * hei, R_c)
    H_a -= H0_x(R_a - 0.5 * wid, R_b + 0.5 * hei, R_c)

    H_b  = H0_x(R_a + 0.5 * wid, R_b + 0.5 * hei, R_c)
    H_b += H0_y(R_a - 0.5 * wid, R_b - 0.5 * hei, R_c)
    H_b -= H0_y(R_a + 0.5 * wid, R_b - 0.5 * hei, R_c)
    H_b -= H0_y(R_a - 0.5 * wid, R_b + 0.5 * hei, R_c)

    H_c  = H0_z(R_a + 0.5 * wid, R_b + 0.5 * hei, R_c)
    H_c += H0_z(R_a - 0.5 * wid, R_b - 0.5 * hei, R_c)
    H_c -= H0_z(R_a + 0.5 * wid, R_b - 0.5 * hei, R_c)
    H_c -= H0_z(R_a - 0.5 * wid, R_b + 0.5 * hei, R_c)

    H_xyz = e_a * H_a + e_b * H_b + e_c * H_c
    
    return H_xyz

# patch perp to x: exchange x, y, z -> z, -y, x
def H_xx(x_o, y_o, z_o, x_a, y_a, y_b, z_a, z_b):
    #print ("H_xx", x_a, y_a, y_b, z_a, z_b,
    #       "min: x:", np.min(np.abs(x_o - x_a)),
    #       "y:", np.min(np.abs(y_o - y_a)),
    #       np.min(np.abs((y_o - y_b))),
    #       "z:", np.min(np.abs(z_o - z_a)),
    #       np.min(np.abs(z_o - z_b)))
    res =   0.0
    res +=  H0_z(z_b - z_o, -y_b + y_o, x_a - x_o)
    res -=  H0_z(z_b - z_o, -y_a + y_o, x_a - x_o)
    res -=  H0_z(z_a - z_o, -y_b + y_o, x_a - x_o)
    res +=  H0_z(z_a - z_o, -y_a + y_o, x_a - x_o)
    return -res

def H_yx(x_o, y_o, z_o, x_a, y_a, y_b, z_a, z_b):

    #print ("H_yx", x_a, y_a, y_b, z_a, z_b,
    #       "min: x:", np.min(np.abs(x_o - x_a)),
    #       "y:", np.min(np.abs(y_o - y_a)),
    #       np.min(np.abs((y_o - y_b))),
    #       "z:", np.min(np.abs(z_o - z_a)),
    #       np.min(np.abs(z_o - z_b)))
    res =   0.0
    res +=  H0_y(z_b - z_o, -y_b + y_o, x_a - x_o)
    res -=  H0_y(z_b - z_o, -y_a + y_o, x_a - x_o)
    res -=  H0_y(z_a - z_o, -y_b + y_o, x_a - x_o)
    res +=  H0_y(z_a - z_o, -y_a + y_o, x_a - x_o)
    return res

def H_zx(x_o, y_o, z_o, x_a, y_a, y_b, z_a, z_b):

    #print ("H_zx", x_a, y_a, y_b, z_a, z_b,
    #       "min: x:", np.min(np.abs(x_o - x_a)),
    #       "y:", np.min(np.abs(y_o - y_a)),
    #       np.min(np.abs((y_o - y_b))),
    #       "z:", np.min(np.abs(z_o - z_a)),
    #       np.min(np.abs(z_o - z_b)))
    res =   0.0
    res +=  H0_x(z_b - z_o, -y_b + y_o, x_a - x_o)
    res -=  H0_x(z_b - z_o, -y_a + y_o, x_a - x_o)
    res -=  H0_x(z_a - z_o, -y_b + y_o, x_a - x_o)
    res +=  H0_x(z_a - z_o, -y_a + y_o, x_a - x_o)
    return -res

# patch perp to y: exchange x, y, z -> -x, z, y
def H_xy(x_o, y_o, z_o, x_a, x_b, y_a, z_a, z_b):
    #print ("H_xy", x_a, x_b, y_a, z_a, z_b,
    #       "min: x:", np.min(np.abs(x_o - x_a)),
    #       np.min(np.abs(x_o - x_b)),
    #       "y:", np.min(np.abs(y_o - y_a)),
    #       "z:", np.min(np.abs(z_o - z_a)),
    #       np.min(np.abs(z_o - z_b)))

    res =   0.0
    res +=  H0_x(-x_b + x_o, z_b - z_o, y_a - y_o)
    res -=  H0_x(-x_b + x_o, z_a - z_o, y_a - y_o)
    res -=  H0_x(-x_a + x_o, z_b - z_o, y_a - y_o)
    res +=  H0_x(-x_a + x_o, z_a - z_o, y_a - y_o)
    return res

def H_yy(x_o, y_o, z_o, x_a, x_b, y_a, z_a, z_b):

    #print ("H_yy", x_a, x_b, y_a, z_a, z_b,
    #       "min: x:", np.min(np.abs(x_o - x_a)),
    #       np.min(np.abs(x_o - x_b)),
    #       "y:", np.min(np.abs(y_o - y_a)),
    #       "z:", np.min(np.abs(z_o - z_a)),
    #       np.min(np.abs(z_o - z_b)))
    res =   0.0
    res +=  H0_z(-x_b + x_o, z_b - z_o, y_a - y_o)
    res -=  H0_z(-x_b + x_o, z_a - z_o, y_a - y_o)
    res -=  H0_z(-x_a + x_o, z_b - z_o, y_a - y_o)
    res +=  H0_z(-x_a + x_o, z_a - z_o, y_a - y_o)
    return -res

def H_zy(x_o, y_o, z_o, x_a, x_b, y_a, z_a, z_b):

    #print ("H_zy", x_a, x_b, y_a, z_a, z_b,
    #       "min: x:", np.min(np.abs(x_o - x_a)),
    #       np.min(np.abs(x_o - x_b)),
    #       "y:", np.min(np.abs(y_o - y_a)),
    #       "z:", np.min(np.abs(z_o - z_a)),
    #       np.min(np.abs(z_o - z_b)))
    res =   0.0
    res +=  H0_y(-x_b + x_o, z_b - z_o, y_a - y_o)
    res -=  H0_y(-x_b + x_o, z_a - z_o, y_a - y_o)
    res -=  H0_y(-x_a + x_o, z_b - z_o, y_a - y_o)
    res +=  H0_y(-x_a + x_o, z_a - z_o, y_a - y_o)
    return -res

# patch orthogonal to z
def H_xz(x_o, y_o, z_o, x_a, x_b, y_a, y_b, z_a):

    #print ("H_xz", x_a, x_b, y_a, y_b, z_a,
    #       "min: x:", np.min(np.abs(x_o - x_a)),
    #       np.min(np.abs(x_o - x_b)),
    #       "y:", np.min(np.abs(y_o - y_a)), np.min(np.abs(y_o - y_b)),
    #       "z:", np.min(np.abs(z_o - z_a)))
    res =   0.0
    res +=  H0_x(x_b - x_o, y_b - y_o, z_a - z_o)
    res -=  H0_x(x_b - x_o, y_a - y_o, z_a - z_o)
    res -=  H0_x(x_a - x_o, y_b - y_o, z_a - z_o)
    res +=  H0_x(x_a - x_o, y_a - y_o, z_a - z_o)
    return res

def H_yz(x_o, y_o, z_o, x_a, x_b, y_a, y_b, z_a):

    #print ("H_yz", x_a, x_b, y_a, y_b, z_a,
    #       "min: x:", np.min(np.abs(x_o - x_a)),
    #       np.min(np.abs(x_o - x_b)),
    #       "y:", np.min(np.abs(y_o - y_a)), np.min(np.abs(y_o - y_b)),
    #       "z:", np.min(np.abs(z_o - z_a)))
    res =   0.0
    res +=  H0_y(x_b - x_o, y_b - y_o, z_a - z_o)
    res -=  H0_y(x_b - x_o, y_a - y_o, z_a - z_o)
    res -=  H0_y(x_a - x_o, y_b - y_o, z_a - z_o)
    res +=  H0_y(x_a - x_o, y_a - y_o, z_a - z_o)
    return res

def H_zz(x_o, y_o, z_o, x_a, x_b, y_a, y_b, z_a):

    #print ("H_zz", x_a, y_a, y_a, y_b, z_a,
    #       "min: x:", np.min(np.abs(x_o - x_a)),
    #       np.min(np.abs(x_o - x_b)),
    #       "y:", np.min(np.abs(y_o - y_a)), np.min(np.abs(y_o - y_b)),
    #       "z:", np.min(np.abs(z_o - z_a)))
    res =   0.0
    res +=  H0_z(x_b - x_o, y_b - y_o, z_a - z_o)
    res -=  H0_z(x_b - x_o, y_a - y_o, z_a - z_o)
    res -=  H0_z(x_a - x_o, y_b - y_o, z_a - z_o)
    res +=  H0_z(x_a - x_o, y_a - y_o, z_a - z_o)
    return res


if  False:
    x = np.linspace (-5.0, 5.0, 100)
    y = np.array(x)
    z = np.array(x)
    dx = 1.0
    dy = 1.0
    dz = 1.0
    Y_xy, X_xy = np.meshgrid(y, x)
    Z_yz, Y_yz = np.meshgrid(z, y)
    Z_xz, X_xz = np.meshgrid(z, x)
    H_xyp_x = H_xz(X_xy, Y_xy, 0.5 + 0.0*X_xy,  -dx, dx, -dy, dy, 0.0)
    H_xyp_y = H_yz(X_xy, Y_xy, 0.5 + 0.0*X_xy,  -dx, dx, -dy, dy, 0.0)
    H_xyp_z = H_zz(X_xy, Y_xy, 0.5 + 0.0*X_xy,   -dx, dx, -dy, dy, 0.0)
    H_xym_x = H_xz(X_xy, Y_xy, -0.5 + 0.0*X_xy, -dx, dx, -dy, dy, 0.0)
    H_xym_y = H_yz(X_xy, Y_xy, -0.5 + 0.0*X_xy, -dx, dx, -dy, dy, 0.0)
    H_xym_z = H_zz(X_xy, Y_xy, -0.5 + 0.0*X_xy, -dx, dx, -dy, dy, 0.0)
    H_yzp_x = H_xz(0.5 + Y_yz*0, Y_yz, Z_yz,    -dx, dx, -dy, dy, 0.0)
    H_yzp_y = H_yz(0.5 + Y_yz*0, Y_yz, Z_yz,    -dx, dx, -dy, dy, 0.0)
    H_yzp_z = H_zz(0.5 + Y_yz*0, Y_yz, Z_yz,    -dx, dx, -dy, dy, 0.0)
    H_yzm_x = H_xz(-0.5 + Y_yz*0, Y_yz, Z_yz,   -dx, dx, -dy, dy, 0.0)
    H_yzm_y = H_yz(-0.5 + Y_yz*0, Y_yz, Z_yz,   -dx, dx, -dy, dy, 0.0)
    H_yzm_z = H_zz(-0.5 + Y_yz*0, Y_yz, Z_yz, -dx, dx, -dy, dy, 0.0)
    H_xzp_x = H_xz(X_xz, 0.5 + 0.0*X_xz, Z_xz,  -dx, dx, -dy, dy, 0.0)
    H_xzp_y = H_yz(X_xz, 0.5 + 0.0*X_xz, Z_xz,  -dx, dx, -dy, dy, 0.0)
    H_xzp_z = H_zz(X_xz, 0.5 + 0.0*X_xz, Z_xz,  -dx, dx, -dy, dy, 0.0)
    H_xzm_x = H_xz(X_xz, -0.5 + 0.0*X_xz, Z_xz, -dx, dx, -dy, dy, 0.0)
    H_xzm_y = H_yz(X_xz, -0.5 + 0.0*X_xz, Z_xz, -dx, dx, -dy, dy, 0.0)
    H_xzm_z = H_zz(X_xz, -0.5 + 0.0*X_xz, Z_xz, -dx, dx, -dy, dy, 0.0)

    pl.figure()
    pl.quiver(X_xy, Y_xy, H_xyp_x, H_xyp_y, pivot='mid')
    pl.gca().set_aspect('equal', 'box')
    pl.title("xy, z = +")
    pl.figure()
    pl.quiver(X_xy, Y_xy, H_xym_x, H_xym_y, pivot='mid')
    pl.gca().set_aspect('equal', 'box')
    pl.title("xy, z = -")
    pl.figure()
    pl.quiver(X_xz, Z_xz, H_xzp_x, H_xzp_z, pivot='mid')
    pl.gca().set_aspect('equal', 'box')
    pl.title("xz, y = +")
    pl.figure()
    pl.quiver(X_xz, Z_xz, H_xzm_x, H_xzm_z, pivot='mid')
    pl.gca().set_aspect('equal', 'box')
    pl.title("xz, y = -")
    pl.figure()
    pl.quiver(Y_yz, Z_yz, H_yzp_y, H_yzp_z, pivot='mid')
    pl.gca().set_aspect('equal', 'box')
    pl.title("yz, x = +")
    pl.figure()
    pl.quiver(Y_yz, Z_yz, H_yzm_y, H_yzm_z, pivot='mid')
    pl.gca().set_aspect('equal', 'box')
    pl.title("yz, x = -")

    pl.show()
    
    H_xyp_x = H_xx(X_xy, Y_xy, 0.5 + 0.0*X_xy,  0.0, -dy, dy, -dz, dz)
    H_xyp_y = H_yx(X_xy, Y_xy, 0.5 + 0.0*X_xy,  0.0, -dy, dy, -dz, dz)
    H_xyp_z = H_zx(X_xy, Y_xy, 0.5 + 0.0*X_xy,  0.0, -dy, dy, -dz, dz)
    H_xym_x = H_xx(X_xy, Y_xy, -0.5 + 0.0*X_xy, 0.0, -dy, dy, -dz, dz)
    H_xym_y = H_yx(X_xy, Y_xy, -0.5 + 0.0*X_xy, 0.0, -dy, dy, -dz, dz)
    H_xym_z = H_zx(X_xy, Y_xy, -0.5 + 0.0*X_xy, 0.0, -dy, dy, -dz, dz)
    H_yzp_x = H_xx(0.5 + Y_yz*0, Y_yz, Z_yz,    0.0, -dy, dy, -dz, dz)
    H_yzp_y = H_yx(0.5 + Y_yz*0, Y_yz, Z_yz,    0.0, -dy, dy, -dz, dz)
    H_yzp_z = H_zx(0.5 + Y_yz*0, Y_yz, Z_yz,    0.0, -dy, dy, -dz, dz)
    H_yzm_x = H_xx(-0.5 + Y_yz*0, Y_yz, Z_yz,   0.0, -dy, dy, -dz, dz)
    H_yzm_y = H_yx(-0.5 + Y_yz*0, Y_yz, Z_yz,   0.0, -dy, dy, -dz, dz)
    H_yzm_z = H_zx(-0.5 + Y_yz*0, Y_yz, Z_yz,   0.0, -dy, dy, -dz, dz)
    H_xzp_x = H_xx(X_xz, 0.5 + 0.0*X_xz, Z_xz,  0.0, -dy, dy, -dz, dz)
    H_xzp_y = H_yx(X_xz, 0.5 + 0.0*X_xz, Z_xz,  0.0, -dy, dy, -dz, dz)
    H_xzp_z = H_zx(X_xz, 0.5 + 0.0*X_xz, Z_xz,  0.0, -dy, dy, -dz, dz)
    H_xzm_x = H_xx(X_xz, -0.5 + 0.0*X_xz, Z_xz, 0.0, -dy, dy, -dz, dz)
    H_xzm_y = H_yx(X_xz, -0.5 + 0.0*X_xz, Z_xz, 0.0, -dy, dy, -dz, dz)
    H_xzm_z = H_zx(X_xz, -0.5 + 0.0*X_xz, Z_xz, 0.0, -dy, dy, -dz, dz)

    pl.figure()
    pl.quiver(X_xy, Y_xy, H_xyp_x, H_xyp_y, pivot='mid')
    pl.gca().set_aspect('equal', 'box')
    pl.title("xy, z = +")
    pl.figure()
    pl.quiver(X_xy, Y_xy, H_xym_x, H_xym_y, pivot='mid')
    pl.gca().set_aspect('equal', 'box')
    pl.title("xy, z = -")
    pl.figure()
    pl.quiver(X_xz, Z_xz, H_xzp_x, H_xzp_z, pivot='mid')
    pl.gca().set_aspect('equal', 'box')
    pl.title("xz, y = +")
    pl.figure()
    pl.quiver(X_xz, Z_xz, H_xzm_x, H_xzm_z, pivot='mid')
    pl.gca().set_aspect('equal', 'box')
    pl.title("xz, y = -")
    pl.figure()
    pl.quiver(Y_yz, Z_yz, H_yzp_y, H_yzp_z, pivot='mid')
    pl.gca().set_aspect('equal', 'box')
    pl.title("yz, x = +")
    pl.figure()
    pl.quiver(Y_yz, Z_yz, H_yzm_y, H_yzm_z, pivot='mid')
    pl.gca().set_aspect('equal', 'box')
    pl.title("yz, x = -")

    pl.show()

    H_xyp_x = H_xy(X_xy, Y_xy, 0.5 + 0.0*X_xy,  -dx, dx, 0.0, -dz, dz)
    H_xyp_y = H_yy(X_xy, Y_xy, 0.5 + 0.0*X_xy,  -dx, dx, 0.0, -dz, dz)
    H_xyp_z = H_zy(X_xy, Y_xy, 0.5 + 0.0*X_xy,  -dx, dx, 0.0, -dz, dz)
    H_xym_x = H_xy(X_xy, Y_xy, -0.5 + 0.0*X_xy, -dx, dx, 0.0, -dz, dz)
    H_xym_y = H_yy(X_xy, Y_xy, -0.5 + 0.0*X_xy, -dx, dx, 0.0, -dz, dz)
    H_xym_z = H_zy(X_xy, Y_xy, -0.5 + 0.0*X_xy, -dx, dx, 0.0, -dz, dz)
    H_yzp_x = H_xy(0.5 + Y_yz*0, Y_yz, Z_yz,    -dx, dx, 0.0, -dz, dz)
    H_yzp_y = H_yy(0.5 + Y_yz*0, Y_yz, Z_yz,    -dx, dx, 0.0, -dz, dz)
    H_yzp_z = H_zy(0.5 + Y_yz*0, Y_yz, Z_yz,    -dx, dx, 0.0, -dz, dz)
    H_yzm_x = H_xy(-0.5 + Y_yz*0, Y_yz, Z_yz,   -dx, dx, 0.0, -dz, dz)
    H_yzm_y = H_yy(-0.5 + Y_yz*0, Y_yz, Z_yz,   -dx, dx, 0.0, -dz, dz)
    H_yzm_z = H_zy(-0.5 + Y_yz*0, Y_yz, Z_yz,   -dx, dx, 0.0, -dz, dz)
    H_xzp_x = H_xy(X_xz, 0.5 + 0.0*X_xz, Z_xz,  -dx, dx, 0.0, -dz, dz)
    H_xzp_y = H_yy(X_xz, 0.5 + 0.0*X_xz, Z_xz,  -dx, dx, 0.0, -dz, dz)
    H_xzp_z = H_zy(X_xz, 0.5 + 0.0*X_xz, Z_xz,  -dx, dx, 0.0, -dz, dz)
    H_xzm_x = H_xy(X_xz, -0.5 + 0.0*X_xz, Z_xz, -dx, dx, 0.0, -dz, dz)
    H_xzm_y = H_yy(X_xz, -0.5 + 0.0*X_xz, Z_xz, -dx, dx, 0.0, -dz, dz)
    H_xzm_z = H_zy(X_xz, -0.5 + 0.0*X_xz, Z_xz, -dx, dx, 0.0, -dz, dz)

    pl.figure()
    pl.quiver(X_xy, Y_xy, H_xyp_x, H_xyp_y, pivot='mid')
    pl.gca().set_aspect('equal', 'box')
    pl.title("xy, z = +")
    pl.figure()
    pl.quiver(X_xy, Y_xy, H_xym_x, H_xym_y, pivot='mid')
    pl.gca().set_aspect('equal', 'box')
    pl.title("xy, z = -")
    pl.figure()
    pl.quiver(X_xz, Z_xz, H_xzp_x, H_xzp_z, pivot='mid')
    pl.gca().set_aspect('equal', 'box')
    pl.title("xz, y = +")
    pl.figure()
    pl.quiver(X_xz, Z_xz, H_xzm_x, H_xzm_z, pivot='mid')
    pl.gca().set_aspect('equal', 'box')
    pl.title("xz, y = -")
    pl.figure()
    pl.quiver(Y_yz, Z_yz, H_yzp_y, H_yzp_z, pivot='mid')
    pl.gca().set_aspect('equal', 'box')
    pl.title("yz, x = +")
    pl.figure()
    pl.quiver(Y_yz, Z_yz, H_yzm_y, H_yzm_z, pivot='mid')
    pl.gca().set_aspect('equal', 'box')
    pl.title("yz, x = -")

    pl.show()
    

        
def compute_H_operator (box_array):
    N = len(box_array.X)
    H = TensorKernel(N)
    #Hxx = np.zeros ((N, N))
    #Hxy = np.zeros ((N, N))
    #Hxz = np.zeros ((N, N))
    #Hyx = np.zeros ((N, N))
    #Hyy = np.zeros ((N, N))
    #Hyz = np.zeros ((N, N))
    #Hzx = np.zeros ((N, N))
    #Hzy = np.zeros ((N, N))
    #Hzz = np.zeros ((N, N))

    X_o = box_array.X
    Y_o = box_array.Y
    Z_o = box_array.Z

    #print ("X_o, Z_o", X_o, Z_o)
    for i in range(N):
        x_c = box_array.X[i]
        y_c = box_array.Y[i]
        z_c = box_array.Z[i]
        x_w = x_c - 0.5 * box_array.DX[i] 
        x_e = x_c + 0.5 * box_array.DX[i] 
        y_s = y_c - 0.5 * box_array.DY[i] 
        y_n = y_c + 0.5 * box_array.DY[i] 
        z_b = z_c - 0.5 * box_array.DZ[i] 
        z_t = z_c + 0.5 * box_array.DZ[i]
        
        H.xx[:, i] += H_xx(X_o, Y_o, Z_o, x_e, y_s, y_n, z_b, z_t)
        H.xx[:, i] -= H_xx(X_o, Y_o, Z_o, x_w, y_s, y_n, z_b, z_t)
        H.yx[:, i] += H_yx(X_o, Y_o, Z_o, x_e, y_s, y_n, z_b, z_t)
        H.yx[:, i] -= H_yx(X_o, Y_o, Z_o, x_w, y_s, y_n, z_b, z_t)
        H.zx[:, i] += H_zx(X_o, Y_o, Z_o, x_e, y_s, y_n, z_b, z_t)
        H.zx[:, i] -= H_zx(X_o, Y_o, Z_o, x_w, y_s, y_n, z_b, z_t)

        H.xy[:, i] += H_xy(X_o, Y_o, Z_o, x_w, x_e, y_n, z_b, z_t)
        H.xy[:, i] -= H_xy(X_o, Y_o, Z_o, x_w, x_e, y_s, z_b, z_t)
        H.yy[:, i] += H_yy(X_o, Y_o, Z_o, x_w, x_e, y_n, z_b, z_t)
        H.yy[:, i] -= H_yy(X_o, Y_o, Z_o, x_w, x_e, y_s, z_b, z_t)
        H.zy[:, i] += H_zy(X_o, Y_o, Z_o, x_w, x_e, y_n, z_b, z_t)
        H.zy[:, i] -= H_zy(X_o, Y_o, Z_o, x_w, x_e, y_s, z_b, z_t)

        H.xz[:, i] += H_xz(X_o, Y_o, Z_o, x_w, x_e, y_s, y_n, z_t)
        H.xz[:, i] -= H_xz(X_o, Y_o, Z_o, x_w, x_e, y_s, y_n, z_b)
        H.yz[:, i] += H_yz(X_o, Y_o, Z_o, x_w, x_e, y_s, y_n, z_t)
        H.yz[:, i] -= H_yz(X_o, Y_o, Z_o, x_w, x_e, y_s, y_n, z_b)
        H.zz[:, i] += H_zz(X_o, Y_o, Z_o, x_w, x_e, y_s, y_n, z_t)
        H.zz[:, i] -= H_zz(X_o, Y_o, Z_o, x_w, x_e, y_s, y_n, z_b)
        

    H.scale(1.0 / (4.0 * np.pi))
    #Hxx /= 4.0 * np.pi
    #Hyy /= 4.0 * np.pi
    #Hzz /= 4.0 * np.pi
    #Hxz /= 4.0 * np.pi
    #Hyz /= 4.0 * np.pi
    #Hyx /= 4.0 * np.pi
    #Hzx /= 4.0 * np.pi
    #Hxy /= 4.0 * np.pi
    #Hzy /= 4.0 * np.pi
    return H # Hxx, Hxy, Hxz, Hyx, Hyy, Hyz, Hzx, Hzy, Hzz


class Mode:
    def __init__ (self, model, f, mode_mab):
        self.f = f
        self.model  = model
        self.ma_all = mode_mab[0::2]
        self.mb_all = mode_mab[1::2]
        ones = 0.0 * self.ma_all + 1.0
        ea_all = model.ab_to_xyz(ones, 0.0 * ones)
        eb_all = model.ab_to_xyz(0.0 * ones, ones)
        self.m_all = ea_all * self.ma_all + eb_all * self.mb_all 
        self.ma = dict()
        self.mb = dict()
        self.mx = dict()
        self.my = dict()
        self.mz = dict()
        for area in model.areas():
            ma_area = model.to_area_coord(self.ma_all, area)
            mb_area = model.to_area_coord(self.mb_all, area)
            e_xa_area = model.to_area_coord(ea_all.x, area)
            e_ya_area = model.to_area_coord(ea_all.y, area)
            e_za_area = model.to_area_coord(ea_all.z, area)
            e_xb_area = model.to_area_coord(eb_all.x, area)
            e_yb_area = model.to_area_coord(eb_all.y, area)
            e_zb_area = model.to_area_coord(eb_all.z, area)
            self.ma[area.name] = ma_area
            self.mb[area.name] = mb_area
            self.mx[area.name] = ma_area * e_xa_area + mb_area * e_xb_area
            self.my[area.name] = ma_area * e_ya_area + mb_area * e_yb_area
            self.mz[area.name] = ma_area * e_za_area + mb_area * e_zb_area

    def copy(self):
        mab = np.zeros((len(self.ma_all) + len(self.mb_all)),
                       dtype=self.ma_all.dtype)
        mab[0::2] = self.ma_all
        mab[1::2] = self.mb_all
        mode_copy = Mode(self.model, self.f, mab)
        #print ("check copy: mx_all", linalg.norm(self.mx_all - mode_copy.mx_all))
        #print ("check copy: mz_all", linalg.norm(self.mz_all - mode_copy.mz_all))
        #for k in self.mx.keys():
        #    print ("  check x", k, linalg.norm(self.mx[k] - mode_copy.mx[k]))
        #    print ("  check z", k, linalg.norm(self.mz[k] - mode_copy.mz[k]))
        return mode_copy

    def normalize(self, area = ''):
        C_norm = self.model.dot_product(self, self, area)
        self.scale(1.0 / np.sqrt(np.abs(C_norm)))

    def participation (self, *areas):
        C_tot  = self.model.dot_product(self, self)
        C_area = 0.0
        for area in areas:
            #print ("area", area)
            C_area += self.model.dot_product(self, self, area)
        return np.abs(C_area / C_tot)
    
    def scale(self, scale_factor):
        self.ma_all *= scale_factor
        self.mb_all *= scale_factor
        for k in self.mx.keys():
            self.ma[k] = self.ma[k] * scale_factor
        for k in self.mz.keys():
            self.mb[k] = self.mb[k] * scale_factor
        for k in self.mx.keys():
            self.mx[k] = self.mx[k] * scale_factor
        for k in self.mz.keys():
            self.my[k] = self.my[k] * scale_factor
        for k in self.mz.keys():
            self.mz[k] = self.mz[k] * scale_factor
        
    def freq(self):
        return self.f

    def m(self, area_name):
        return self.mx[area_name], self.my[area_name], self.mz[area_name]
    
    def m_ab(self, area_name):
        return self.ma[area_name], self.mb[area_name]

    def m_all(self):
        return self.mx_all, self.my_all, self.mz_all
    
    def m_ab_all(self):
        return self.ma_all, self.mb_all


    
class Dreibein:
    def __init__ (self, N):
        # Trivial dreibein
        self.N = N
        self.ea = VectorField(N)
        self.eb = VectorField(N)
        self.ec = VectorField(N)
        #self.eax = np.ones  ((N))
        #self.eay = np.zeros ((N))
        #self.eaz = np.zeros ((N))
        #self.ebx = np.zeros ((N))
        #self.eby = np.ones  ((N))
        #self.ebz = np.zeros ((N))
        #self.ecx = np.zeros ((N))
        #self.ecy = np.zeros ((N))
        #self.ecz = np.ones  ((N))

    def setup (self, Ms_v):
        #Ms = VectorField(self.N)
        zhat = VectorField(self.N)
        zhat.x[:] = 0.0
        zhat.y[:] = 0.0
        zhat.z[:] = 1.0
        #Ms.x = Ms_x
        #Ms.y = Ms_y
        #Ms.z = Ms_z
        Ms = Ms_v.copy()
        Ms.normalize()
        
        self.ec = Ms.copy()
        self.ea = self.ec.cross(zhat)
        self.ea.normalize()
        self.eb = self.ea.cross(self.ec)
        self.eb.normalize()
        #angle_a_xy = np.angle(Ms_y - 1j * Ms_x)
        #self.eax = np.cos(angle_a_xy)
        #self.eay = np.sin(angle_a_xy)
        #self.eaz = 0.0 * Ms_z / Ms
        #print ("check ea perp ec",
        #       np.sum((self.ea.x * self.ec.x + self.ea.y * self.ec.y
        #               + self.ea.z * self.ec.z)**2))
        #self.ebx = self.eay * self.ecz - self.eaz * self.ecy
        #self.eby = self.eaz * self.ecx - self.eax * self.ecz
        #self.ebz = self.eax * self.ecy - self.eay * self.ecx
        #print ("check eb perp ec",
        #       np.sum((self.eb.x * self.ec.x + self.eb.y * self.ec.y
        #               + self.eb.z * self.ec.z)**2))
        #print ("check eb perp ea",
        #       np.sum((self.eb.x * self.ea.x + self.eb.y * self.ea.y
        #               + self.eb.z * self.ea.z)**2))
        #print ("check |ea| = 1",
        #       np.sum((self.ea.x**2 + self.ea.y**2 + self.ea.z**2 - 1.0)**2))
        #print ("check |eb| = 1",
        #       np.sum((self.eb.x**2 + self.eb.y**2 + self.eb.z**2 - 1.0)**2))
        #print ("check |ec| = 1",
        #       np.sum((self.ec.x**2 + self.ec.y**2 + self.ec.z**2 - 1.0)**2))
        

    def abc_to_xyz(self, Vabc):
        result = self.ea * Vabc.x + self.eb * Vabc.y + self.ec * Vabc.z
        return result
        
    def proj3(self, V):
        return VectorField(V.dot(self.ea), V.dot(self.eb), V.dot(self.ec))

    def proj9(self, K):
        result = TensorKernel(self.N)
        tmp = TensorKernel(self.N)
        for i in range(self.N):            
            tmp.set_row_x(i, self.proj3(K.row_x(i)))
            tmp.set_row_y(i, self.proj3(K.row_y(i)))
            tmp.set_row_z(i, self.proj3(K.row_z(i)))
            
        for j in range(self.N):
            result.set_col_x(j, self.proj3(tmp.col_x(j)))
            result.set_col_y(j, self.proj3(tmp.col_y(j)))
            result.set_col_z(j, self.proj3(tmp.col_z(j)))
        return result

class CellArrayModel:
    def __init__ (self):
        self.box_array  = BoxArray()
        self.dreibein   = Dreibein((0))
        self.link_array = LinkArray()
        self.materials  = []
        self.Bbias_x    = []
        self.Bbias_y    = []
        self.Bbias_z    = []
        self.cache      = dict()

    def invalidate_cache(self, *args):
        if len(args) == 0:
           self.cache = dict()
        else:
           for arg in args:
               if arg in self.cache.keys():
                  del self.cache[arg]

    def retrieve_and_cache(self, key, retrieve_proc):
        if key not in self.cache.keys():
            #print ("retrieve: ", key)
            self.expand_material_constants()
            self.cache[key] = retrieve_proc()
            #print ("got: ", self.cache[key])
        #print ("retrieve from cache: ", self.cache[key], key)
        return self.cache[key]

        
    def get_I(self):
        return self.retrieve_and_cache('I',
                        lambda: self.compute_I())
        
    def get_H(self):
        return self.retrieve_and_cache('H',
                    lambda: compute_H_operator(self.box_array))
        
    def get_J(self):
        N = len(self.box_array.X)
        return self.retrieve_and_cache('J',
                    lambda: self.link_array.compute_J_operator(N))
                   
    def setup_dreibein(self, n):
        self.dreibein.setup(n)

    def ab_to_xyz(self, ma, mb):
        if self.dreibein == None:
            return ma, mb, 0.0 * ma
        v_m = VectorField((len(ma)))
        v_m.x = ma
        v_m.y = mb
        v_m.z = 0.0 * ma
        return self.dreibein.abc_to_xyz(v_m)

    def areas(self):
        return self.box_array.get_areas()
        
    def get_area(self, area_name):
        return self.box_array.get_area(area_name)
    
    def get_area_mask(self, area_name):
        return self.box_array.get_area_mask(area_name)

    def add_area(self, area, Bbias_x, Bbias_y, Bbias_z):
        area_id, Nboxes = self.box_array.extend(area)
        self.link_array.update(area)
        self.materials.extend ([area.material] * Nboxes)
        self.Bbias_x.extend([Bbias_x] * Nboxes)
        self.Bbias_y.extend([Bbias_y] * Nboxes)
        self.Bbias_z.extend([Bbias_z] * Nboxes)
        self.dreibein = Dreibein(len(self.box_array.X))
        self.invalidate_cache()
    
    #TODO
    def connect(self, boundary1, boundary2, Jex, M1, M2):
        if len(boundary1.boxes) != len(boundary2.boxes):
            raise Exception("Connecting boundaries of different length "
                            "is not implemented")
        name1 = boundary1.area.name
        name2 = boundary2.area.name
        joint_name = name1 + ":" + name2

        box1   = boundary1.boxes[0]
        box2   = boundary2.boxes[0]
        dx = box2.x - box1.x
        dy = box2.y - box1.y
        dz = box2.z - box1.z
        dl = np.sqrt(dx**2 + dy**2 + dz**2)
        #box1_c = box1.x + 1j * box1.z
        #box2_c = box2.x + 1j * box2.z
        #dl = np.abs(box1_c - box2_c)
        J_eff = Jex * dl 
        joint_material = Material(joint_name, 0.0, J_eff, 0.0, 0.0)
        r1 = 1.0
        r2 = 1.0
        if  abs(dx) + abs(dy) < 1e-4: # vertical link
              r1 = np.abs(box1.dz) / dl
              r2 = np.abs(box2.dz) / dl
        elif  abs(dy) + abs(dz) < 1e-4: # link along x
              r1 = np.abs(box1.dx) / dl
              r2 = np.abs(box2.dx) / dl
        elif  abs(dx) + abs(dz) < 1e-4: # link along x
              r1 = np.abs(box1.dy) / dl
              r2 = np.abs(box2.dy) / dl
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
            y_mid = 0.5 * (box1.y + box2.y)
            z_mid = 0.5 * (box1.z + box2.z)
            self.link_array.add_link(box1.pos, box2.pos,
                                     x_mid, y_mid, z_mid, dl,
                                     joint_material, factors)

        self.invalidate_cache()

    def dot_product(self, mode1, mode2, area = ''):
        ma1, mb1 = mode1.m_ab_all()
        ma2, mb2 = mode2.m_ab_all()
        integrand = mb1.conjugate() * ma2 - ma1.conjugate() * mb2
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
        Y = self.box_array.Y
        Z = self.box_array.Z
        self.alpha   = np.array([self.materials[t].alpha_func(X[t], Y[t], Z[t])
                                 for t in range(len(self.materials))])
        #self.alpha = self.alpha_func(self.box_a) 
        self.gamma_s = np.array([t.gamma_s for t in self.materials])
        self.Ms      = np.array([t.Ms      for t in self.materials])
        #self.Jex     = np.array([t.Jex     for t in self.materials])

        func_type = type(lambda x: 0.0)
        def eval_b(b, x, y, z):
            if isinstance(b, float):   return b
            if isinstance(b, int):     return float(b)
            if isinstance(b, complex): return b
            if isinstance(b, func_type): return b(x, y, z)
            raise Exception("unknown b type: " + str(b))
        
        for t in range(len(self.Bbias_x)):
            self.Bbias_x[t] = eval_b(self.Bbias_x[t], X[t], Y[t], Z[t])
            self.Bbias_y[t] = eval_b(self.Bbias_y[t], X[t], Y[t], Z[t])
            self.Bbias_z[t] = eval_b(self.Bbias_z[t], X[t], Y[t], Z[t])
            
        self.Bbias_x  = np.array(self.Bbias_x)
        self.Bbias_y  = np.array(self.Bbias_y)
        self.Bbias_z  = np.array(self.Bbias_z)
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
        hx, hy, hz = h_func(self.box_array.X,
                            self.box_array.Y,
                            self.box_array.Z)
        print ("shapes: hx, hz", np.shape(hx), np.shape(hz))
        print ("Ms: ", np.shape(Ms))
        h_abc = self.dreibein.proj3(VectorField(hx, hy, hz)) 
        L0s[0::2] += - Ms * constants.mu_0 * h_abc.y
        L0s[1::2] +=   Ms * constants.mu_0 * h_abc.x
        L0s[0::2] *= gamma_s
        L0s[1::2] *= gamma_s
        a0 = 1.0 / (1.0 + alpha * alpha)
        a1 = alpha * a0
        Ls = np.zeros((2 * N), dtype=complex)
        Ls[0::2] = a0 *  L0s[0::2] + a1 * L0s[1::2]
        Ls[1::2] = -a1 * L0s[0::2] + a0 * L0s[1::2]

        return Ls

    def compute_I(self):
        N = self.box_array.N
        I = np.zeros((2 * N, 2 * N))
        self.expand_material_constants()
        dV = self.box_array.dV()
        I_zx = dV / self.Ms / self.gamma_s
        I[1::2, 0::2] =   np.diag(I_zx)
        I[0::2, 1::2] = - np.diag(I_zx)
        return I

    def compute_static_energy(self, n):
        self.expand_material_constants()
        sgn_M = 1.0 + 0.0 * self.Ms
        sgn_M[self.Ms < 0] = -1.0
        alpha   = self.alpha * sgn_M
        gamma_s = self.gamma_s
        Ms = self.Ms
        #Jex = self.Jex
        Bbias = VectorField(self.Bbias_x, self.Bbias_y, self.Bbias_z)

        H = self.get_H()
        #H = compute_H_operator(self.box_array)
        #Hxx, Hxy, Hxz, Hyx, Hyy, Hyz, Hzx, Hzy, Hzz = H_9
        if  False:
            x0 = 0.5 * (np.max(self.box_array.X) + np.min(self.box_array.X))
            y0 = 0.5 * (np.max(self.box_array.Y) + np.min(self.box_array.Y))
            z0 = 0.5 * (np.max(self.box_array.Z) + np.min(self.box_array.Z))
            dist = (self.box_array.X - x0)**2 +  (self.box_array.Y - y0)**2 \
                +  (self.box_array.Z - z0)**2
            i0 = np.argmin(dist)
            print ("i0 = ", i0, self.box_array.X[i0], self.box_array.Y[i0],
                   self.box_array.Z[i0])
            print ("Hxx_av = ", np.sum(H.xx[i0, :]))
            print ("Hyy_av = ", np.sum(H.yy[i0, :]))
            print ("Hzz_av = ", np.sum(H.zz[i0, :]))
            print ("Hxy_av = ", np.sum(H.xy[i0, :]))
            print ("Hxz_av = ", np.sum(H.xz[i0, :]))
            print ("Hyz_av = ", np.sum(H.yz[i0, :]))
            print ("Hyx_av = ", np.sum(H.yx[i0, :]))
            print ("Hzy_av = ", np.sum(H.zy[i0, :]))
            print ("Hzx_av = ", np.sum(H.zx[i0, :]))

        J = self.get_J()
        #J = self.link_array.compute_J_operator(self.box_array.N)
        
        N = len(Ms)
        #ones = np.ones(len(self.box_array.X))

        C_dip = 1.0
        C_ex  = 1.0
        E_total = 0.0

        dV = self.box_array.dV()
        JK = TensorKernel(n.N)
        JK.diag(J.K, J.K, J.K)
        
        E_total += - np.sum(dV * Bbias.dot (n * Ms))
        E_total += 0.5 * C_ex * JK.dot2(n * dV * Ms, n * Ms)
        E_total += -0.5 * C_dip * constants.mu_0 * H.dot2(n * dV * Ms,
                                                          n * Ms)
        return E_total * 1e+5
        
    def compute_static_field(self, n, C_dip = 1.0, C_ex = 1.0):
        self.expand_material_constants()
        sgn_M = 1.0 + 0.0 * self.Ms
        sgn_M[self.Ms < 0] = -1.0
        alpha   = self.alpha * sgn_M
        gamma_s = self.gamma_s
        Ms = self.Ms
        Bbias = VectorField(self.Bbias_x, self.Bbias_y, self.Bbias_z)

        #H = compute_H_operator(self.box_array)
        H = self.get_H()

        #J = self.link_array.compute_J_operator(self.box_array.N)
        J = self.get_J()
        JK = TensorKernel(n.N)
        JK.diag(J.K, J.K, J.K)
        
        N = len(Ms)
        #ones = np.ones(len(self.box_array.X))

        #C_dip = 1.0
        #C_ex  = 1.0
        B_total  = Bbias
        B_total += H.apply(n * Ms) * C_dip * constants.mu_0 
        B_total -= JK.apply(n * Ms) * C_ex
        return B_total

    def relax_magnetisation(self):

        N = len(self.box_array.X)
        n = VectorField(N)
        desired_type = 'X'
        n.x[:] = 0.0
        n.y[:] = 1.0
        n.z[:] = 0.0
        if desired_type == 'X':
           n.x[:] = 0.0
           n.y[:] = 1.0
           n.z[:] = 0.0
        if desired_type == 'S':
           n.y[:] = 1.0
           n.x[:] = 0.5; #0 * 0.1 * self.box_array.X / np.max(self.box_array.X)
           n.z[:] = 0.0; # - 0 * 0.1 * (self.box_array.Z / np.max(self.box_array.Z))**2
        if desired_type == 'O':
            n.y = 1.0
            X = self.box_array.X
            Y = self.box_array.Y
            x0 = 0.5 * (np.min(X) + np.max(X))
            xw = np.max(X) - np.min(X)
            y0 = 0.5 * (np.min(Y) + np.max(Y))
            yw = np.max(Y) - np.min(Y)
            n.x = -0.7 * (X - x0) / (xw / 2.0) * (Y - y0) / (yw/2.0)
        n.normalize()
        tau = 0.5
        import random
        def anneal(n_x, n_y, n_z, xi):
            #return n_x, n_y, n_z
            N = len(n_x)

            print ("***** ANNEAL *******")
            new_n = n.copy()
            for i in range(N):
                nx_i = n.x[i]
                ny_i = n.y[i]
                nz_i = n.z[i]
                phi = random.random() * 2.0 * np.pi
                rnd = random.random()
                cs = (1.0 - 2.0 * xi * rnd)
                sn = np.sqrt(1.0 - cs**2)
                nz_x = 0.0
                nz_y = 0.0
                nz_z = 1.0
                na_x, na_y, na_z = cross(nz_x, nz_y, nz_z, nx_i, ny_i, nz_i)
                na = np.sqrt(dot(na_x, na_y, na_z, na_x, na_y, na_z))
                na_x /= na
                na_y /= na
                na_z /= na
                nb_x, nb_y, nb_z = cross(na_x, na_y, na_z, nx_i, ny_i, nz_i)
                nb = np.sqrt(dot(nb_x, nb_y, nb_z, nb_x, nb_y, nb_z))
                nb_x /= nb
                nb_y /= nb
                nb_z /= nb
                cs_phi = np.cos(phi)
                sn_phi = np.sin(phi)
                new_n.x[i] = nx_i * cs + na_x * sn * cs_phi + nb_x * sn * sn_phi 
                new_n.y[i] = ny_i * cs + na_y * sn * cs_phi + nb_y * sn * sn_phi 
                new_n.z[i] = nz_i * cs + na_z * sn * cs_phi + nb_z * sn * sn_phi
            new_n.normalize()
            return new_n

                
        def show_state(H):
           Z_min = np.min(self.box_array.Z)
           Z_max = np.max(self.box_array.Z)
           Z_b = self.box_array.Z
           Hs_3 = self.compute_static_field(n, 1.0, 0.0)
           Hs = n.dot(Hs_3)
           #H_x, H_y, H_z = self.compute_static_field(n_x, n_y, n_z);
           print ("Hy = ", np.min(H.y), np.max(H.y))
           for z in [Z_min]: #[Z_min, Z_max]:
               i_z = [t for t in range(len(self.box_array.Z)) if abs(Z_b[t] - z) < 1e-6]
               X_z = np.array([self.box_array.X[t] for t in i_z])
               Y_z = np.array([self.box_array.Y[t] for t in i_z])
               nz_x = np.array([n.x[t] for t in i_z])
               nz_y = np.array([n.y[t] for t in i_z])
               nz_z = np.array([n.z[t] for t in i_z])
               hz_x = np.array([H.x[t] for t in i_z])
               hz_y = np.array([H.y[t] for t in i_z])
               hz_z = np.array([H.z[t] for t in i_z])
               hs_z = np.array([Hs[t]  for t in i_z])
               nz_x_scale = np.max(np.abs(nz_x))
               nz_z_scale = np.max(np.abs(nz_z))
               pl.figure()
               pl.tripcolor(X_z, Y_z, hs_z)
               pl.gca().set_aspect('equal', 'box')
               pl.colorbar()
               pl.title('hs')
               pl.figure()
               pl.tripcolor(X_z, Y_z, nz_x, cmap='bwr', vmin=-nz_x_scale, vmax=nz_x_scale)
               pl.gca().set_aspect('equal', 'box')
               pl.colorbar()
               pl.title('n_x')
               pl.figure()
               pl.tripcolor(X_z, Y_z, nz_y, cmap='bwr', vmin=-1.0, vmax=1.0)
               pl.gca().set_aspect('equal', 'box')
               pl.colorbar()
               pl.title("n_y")
               pl.figure()
               pl.tripcolor(X_z, Y_z, nz_z, cmap='bwr', vmin=-nz_z_scale, vmax=nz_z_scale)
               pl.gca().set_aspect('equal', 'box')
               pl.colorbar()
               pl.title("n_z")
               pl.figure()
               pl.quiver(X_z, Y_z, nz_x, nz_y, pivot='mid') 
               pl.gca().set_aspect('equal', 'box')
               pl.title("nx ny")
               pl.figure()
               pl.quiver(X_z, Y_z, hz_x, hz_y, pivot='mid')
               pl.gca().set_aspect('equal', 'box')
               pl.title("Hx Hy")
               pl.figure()
               Hx_scale=np.max(np.abs(hz_x))
               pl.tripcolor(X_z, Y_z, hz_x, cmap='bwr', vmin=-Hx_scale, vmax=Hx_scale)
               pl.gca().set_aspect('equal', 'box')
               pl.colorbar()
               pl.title("H_x")
               pl.figure()
               Hy_scale=np.max(np.abs(hz_y))
               pl.tripcolor(X_z, Y_z, hz_y, cmap='bwr', vmin=-Hy_scale, vmax=Hy_scale)
               pl.gca().set_aspect('equal', 'box')
               pl.colorbar()
               pl.title("H_y")
               pl.figure()
               Hz_scale=np.max(np.abs(nz_z))
               pl.tripcolor(X_z, Y_z, hz_z, cmap='bwr', vmin=-Hz_scale, vmax=Hz_scale)
               pl.gca().set_aspect('equal', 'box')
               pl.colorbar()
               pl.title("H_z")
               NxH_x = nz_y * hz_z - nz_z * hz_y
               NxH_y = nz_z * hz_x - nz_x * hz_z
               NxH_z = nz_x * hz_y - nz_y * hz_x
               NxH = np.sqrt(NxH_x**2 + NxH_y**2 + NxH_z**2)
               pl.figure()
               pl.tripcolor(X_z, Y_z, NxH)
               pl.gca().set_aspect('equal', 'box')
               pl.colorbar()
               pl.title("NxH")
               
               
           pl.show()
        n_it = 0
        #H_x, H_y, H_z = self.compute_static_field(n_x, n_y, n_z);
        #T_x = n_y * H_z - n_z * H_y
        #T_y = n_z * H_x - n_x * H_z
        #T_z = n_x * H_y - n_y * H_x
        #T2 = np.sum(T_x**2 + T_y**2 + T_z**2)
        def field_and_torque(n):
            H = self.compute_static_field(n)
            T = H.cross(n)
            return H, T
        #n_x, n_y, n_z = anneal(n_x, n_y, n_z, 0.01)
        H, T = field_and_torque(n)
        def T_mag(T):
            return np.sum(T.x**2 + T.y**2 + T.z**2)
        def n_polar(theta, phi):
            n_x = np.sin(theta) * np.cos(phi)
            n_y = np.cos(theta)
            n_z = np.sin(theta) * np.sin(phi)
            return n_x, n_y, n_z
        global n_f3;
        n_f3 = 0
        def f_min3(x):
            global n_f3
            n_f3 += 1
            phi   = x[0::2]
            theta = x[1::2]
            n_x, n_y, n_z = n_polar(theta, phi)
            n = VectorField(len(n_x))
            n.x = n_x
            n.y = n_y
            n.z = n_z
            #n.normalize()
            #n_x = x[0::3]
            #n_y = x[1::3]
            #n_z = x[2::3]
            n2 = np.sum((n.x**2 + n.y**2 + n.z**2 - 1.0)**2)
            H, T = field_and_torque(n_x, n_y, n_z)
            t_mag =  T_mag(T)
            E = self.compute_static_energy(n)
            print ("n2 = ", n2, "t_mag = ", t_mag, "E = ", E)
            if n_f3 % 1000 == 0: show_state(H)
            return E # n2 * 100 + t_mag
        x0 = np.zeros((2 * n.N))
        bounds = []
        for i in range(n.N):
            bounds.append([0.0, np.pi])
            bounds.append([-np.pi, np.pi])
        from scipy import optimize
        #res3 = optimize.minimize(f_min3, x0, bounds=bounds)
        #print ("res3 =  ", res3)
        #x3 = res3['x']
        #n_x, n_y, n_z = n_polar(x3[0::2], x3[1::2])
        H, T = field_and_torque(n)
        #show_state(H)
        E = self.compute_static_energy(n)
        while False:
            print ("n[0]", n.x[0], n.y[0], n.z[0], "E = ", E)
            print ("Hy = ", np.min(H.y), np.max(H.y))
            n_it += 1
            if (n_it % 50 == 0):
                show_state(H)

            print (T, type(T))
            T2 = T_mag(T)
            #if (n_it == 1):
            #    show_state(H_x, H_y, H_z)
            Hmag = np.sqrt(H.x**2 + H.y**2 + H.z**2)
            #print ("demag total: ", np.sum(H))
            #dn_x = n_x + H_x / H
            #dn_y = n_y + H_y / H
            #dn_z = n_z + H_z / H
            #n_H = dot (n_x, n_y, n_z, H_x, H_y, H_z)
            #n_x * H_x + n_y * H_y + n_z * H_z
            #n2 = dot (n_x, n_y, n_z, n_x, n_y, n_z) #n_x * n_x + n_y * n_y + n_z * n_z
            dnL = T.copy()
            dnG = n.cross(T)
            #dnG_x = n_x * n_H - H_x * n2
            #dnG_y = n_y * n_H - H_y * n2
            #dnG_z = n_z * n_H - H_z * n2
            #dn_rnd = 0.01 *
            def new_n(uv):
                u = uv[0]
                #v = uv[1]
                #X_b = self.box_array.X
                #Y_b = self.box_array.X
                #Z_b = self.box_array.X
                #X_n = (X_b - np.min(X_b)) / (np.max(X_b) - np.min(X_b))
                #Y_n = (Y_b - np.min(Y_b)) / (np.max(Y_b) - np.min(Y_b))
                #Z_n = (Z_b - np.min(Z_b)) / (np.max(Z_b) - np.min(Z_b))
                #A_x = uv[2] * np.sin(2.0 * np.pi * X_n)
                #A_y = uv[3] * np.sin(2.0 * np.pi * Y_n)
                #A_z = uv[4] * np.sin(2.0 * np.pi * Z_n)
                A = 1.0 #(1.0 + A_x + A_y + A_z)
                tau_A = tau * A
                new_nn = n + dnG * u * tau_A 
                new_nn.normalize()
                return new_nn
            def f_min(uv):
                new_nn = new_n(uv)
                new_H, new_T = field_and_torque(new_nn)               
                new_T2 = T_mag(new_T)
                return new_T2
            from scipy import optimize
            #opt_res = optimize.minimize(f_min, np.array([0.5, 0]),
            #                        bounds = [(-1, 1), (-1, 1)])
            opt_res = optimize.minimize(f_min, np.array([0.5]),
                                    bounds = [(-1, 1)])
            #opt_res = optimize.minimize(f_min, np.array([0.5, 0, 0, 0, 0]),
            #                        bounds = [(-1, 1), (-1, 1),
            #                                  (-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)])
            print ("opt_res = ", opt_res)
            #u = opt_res['x'][0]
            #v = opt_res['x'][1]
            new_nn = new_n(opt_res['x'])
            new_H, new_T = field_and_torque(new_nn)
            new_T2 = T_mag(new_T)
            E_old = E
            E = self.compute_static_energy(new_nn)
            print ("T2: old = ", T2, "new = ", new_T2, "diff = ", new_T2 - T2,
                   "diff E = ", E - E_old)
            if new_T2 - T2 < -1e-2:
                print ("copy fields")
                n = new_nn.copy()
                H = new_H.copy()
                T = new_T.copy()
            else:
                #if new_T2 > 0.1:
                #    print ("***** ANNEAL *******")
                #    n_x, n_y, n_z = anneal(n_x, n_y, n_z, 0.001)
                #    H_x, H_y, H_z, T_x, T_y, T_z = field_and_torque(n_x, n_y, n_z)
                #else:
                    print ("cannot decrease T2, bail out")
                    break
            print ("n[0]", n.x[0], n.y[0], n.z[0])
            H2 = H.x**2 + H.y**2 + H.z**2
            err = T_mag(T) / T_mag(H)
            err = np.sqrt(err)
            print ("err = ", err)
            #err = np.sum(dnG_x**2) + np.sum(dnG_y**2) + np.sum(dnG_z**2)
            #err = np.sqrt(err / N)
            #if err < 0.00001: break
            #print ("err n: ", np.sum(dnG_x**2),
            #       np.sum(dnG_y**2), np.sum(dnG_z**2), err)
            if err < 0.001:
                print ("converged")
                break

        
        H, T = field_and_torque(n)
        T2 = T_mag(T)
        #show_state(H_x, H_y, H_z)
        E = self.compute_static_energy(n)
        
        while True:
          E_old = E
          print ("old E = ", E_old)
          self.setup_dreibein(n)
          L = self.compute_LLG_operator()
          T2_old = T_mag(T)
          print ("torque: old = ", T2_old)
          T_abc = self.dreibein.proj3(T)
          b = np.zeros ((2 * len(T_abc.x)))
          b[0::2] = - T_abc.x * self.gamma_s * self.Ms
          b[1::2] = - T_abc.y * self.gamma_s * self.Ms
          print ("Ta, Tb = ", np.sum(T_abc.x**2), np.sum(T_abc.y**2),
                 "Tc = ", np.sum(T_abc.z**2))
          dm_ab = linalg.solve(L, b)
          dm_a = dm_ab[0::2]; dm_b = dm_ab[1::2]
          dn_a = dm_a / self.Ms
          dn_b = dm_b / self.Ms
          print ("na, nb = ", np.sum(dn_a**2), np.sum(dn_b**2))
          dn = self.ab_to_xyz(dn_a, dn_b)
          tau_LLG = 1.0
          print ("dn: ", np.sum(dn.x**2), np.sum(dn.y**2), np.sum(dn.z**2))

          G = n.cross(T)
          def new_n2(tuv):
              #t = tuv[0];v = 0.0; u = 0.0 
              t = tuv[0]; u = 0.0; v = tuv[1];  #u = tuv[1]; v = tuv[2]
              new_nn = n + dn * t * tau_LLG + T * u + G * v
              new_nn.normalize()
              return new_nn
          
          def f_min2(tuv):
              new_nn = new_n2(tuv)
              new_H, new_T = field_and_torque(new_nn)
              T2_new = T_mag(new_T)
              return T2_new / T2_old
          #res = optimize.minimize(f_min2, np.array([-1.0]),
          #                        bounds=[(-2.0, 2.0)])
          res = optimize.minimize(f_min2, np.array([1.0, 0.0]),
                                  bounds=[(-2.0, 2.0), (-1.0, 1.0)])
          #res = optimize.minimize(f_min2, np.array([0.0, 0.0, 0.0]),
          #                        bounds=[(-2.0, 2.0), (-1.0, 1.0), (-1.0, 1.0)])
          print ("opt: ", res)
          tuv = res['x']
          new_n = new_n2(tuv)
          #new_nx = n_x + t * tau_LLG * dn_x
          #new_ny = n_y + t * tau_LLG * dn_y
          #new_nz = n_z + t * tau_LLG * dn_z
          #new_n = np.sqrt(new_nx**2 + new_ny**2 + new_nz**2)
          #new_nx /= new_n
          #new_ny /= new_n
          #new_nz /= new_n
          newH, newT = field_and_torque(new_n)
          T2_new = T_mag(newT)
          print ("torque: new = ", T2_new, T2_new - T2_old)
          E = self.compute_static_energy(new_n)
          print ("Energy: old = ", E_old, "new = ", E, "diff = ", E - E_old)
          if T2_new - T2_old < -1e-8:
              n = new_n.copy()
              H = newH.copy()
              T = newT.copy()
          else:
              print ("new iteration does not minimize the torque")
              break

        #show_state(H)
        return n
            
        
    def compute_LLG_operator(self):

        self.expand_material_constants()
        sgn_M = 1.0 + 0.0 * self.Ms
        sgn_M[self.Ms < 0] = -1.0
        alpha   = self.alpha * sgn_M
        gamma_s = self.gamma_s
        Ms = self.Ms
        #Jex = self.Jex
        Bbias = VectorField(self.Bbias_x, self.Bbias_y, self.Bbias_z)

        #H = compute_H_operator(self.box_array)
        H = self.get_H()
        H_ab = self.dreibein.proj9(H)
        Haa  = H_ab.xx
        Hab  = H_ab.xy
        Hba  = H_ab.yx
        Hbb  = H_ab.yy
        
        #J = self.link_array.compute_J_operator(self.box_array.N)
        J = self.get_J()
        JK = TensorKernel(self.box_array.N)
        JK.diag(J.K, J.K, J.K)
        J_ab = self.dreibein.proj9(JK)
        Jaa = J_ab.xx
        Jab = J_ab.xy
        Jba = J_ab.yx
        Jbb = J_ab.yy
        #Jaa, Jab, Jac, Jba, Jbb, Jbc, Jca, Jcb, Jcc = Jp_9
        
        N = len(Ms)
        ones = np.ones(len(self.box_array.X))
        Bstatic = self.compute_static_field(self.dreibein.ec, 1.0, 1.0)
        Bst_abc = self.dreibein.proj3(Bstatic)
        Bst_c = Bst_abc.z
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
        L0[0::2, 1::2] +=   np.diag(Bst_c) + Ms[:, None] * Jbb
        L0[0::2, 0::2] +=                    Ms[:, None] * Jba
        L0[1::2, 0::2] += - np.diag(Bst_c) - Ms[:, None] * Jaa
        L0[1::2, 1::2] +=                  - Ms[:, None] * Jab
        L0[0::2, 0::2] += - Ms[:, None] * constants.mu_0 * Hba
        L0[0::2, 1::2] += - Ms[:, None] * constants.mu_0 * Hbb
        L0[1::2, 0::2] +=   Ms[:, None] * constants.mu_0 * Haa
        L0[1::2, 1::2] +=   Ms[:, None] * constants.mu_0 * Hab
        L0[0::2, :] *= gamma_s[:, None]
        L0[1::2, :] *= gamma_s[:, None]

        L0 *= 1.0 #/ constants.GHz_2pi

        L = np.zeros((2 * N, 2 * N))
        a0 = 1.0 / (1.0 + alpha * alpha)
        a1 = alpha * a0
        a0 = a0[:, None]
        a1 = a1[:, None]
        L[0::2, 0::2] = a0 * L0[0::2, 0::2]  + a1 * L0[1::2, 0::2]
        L[0::2, 1::2] = a0 * L0[0::2, 1::2]  + a1 * L0[1::2, 1::2]
        L[1::2, 0::2] = -a1 * L0[0::2, 0::2] + a0 * L0[1::2, 0::2]
        L[1::2, 1::2] = -a1 * L0[0::2, 1::2] + a0 * L0[1::2, 1::2]
        
        return L

    def solve(self):
        L = self.compute_LLG_operator()
        iomega, mab = linalg.eig(L)
        f = 1j * iomega / constants.GHz_2pi
        ff = list(f)
        ff.sort(key = lambda t: np.abs(t))
        #print ("f = ", ff)
        modes_unstable = [t for t in range(len(f))
                          if f[t].imag > 0]
        modes_pos = [t for t in range(len(f)) if f[t].real > 0]
        modes_pos.sort(key = lambda t: f[t].real)
        modes_unstable.sort(key = lambda t: -f[t].imag)
        f_unst = np.array([f[t] for t in modes_unstable])
        f_pos  = np.array([f[t] for t in modes_pos])
        
        modes_unpacked = []
        for t in modes_unstable:
            modes_unpacked.append(Mode(self, f[t], mab[:, t]))
        for t in modes_pos:
            modes_unpacked.append(Mode(self, f[t], mab[:, t]))

        coords = dict()
        for area in self.areas():
            coords[area.name] = self.area_coords(area)

        area_names = list([t.name for t in self.areas()])

        result = dict()
        result['area_names'] = area_names
        result['modes'] = modes_unpacked
        result['coords'] = coords
        result['coords_all'] = self.box_array.X, self.box_array.Y, self.box_array.Z
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
            #print ("done")
            mode = Mode(self, omega / constants.GHz_2pi, mxz)
            modes.append(mode)

        area_names = list([t.name for t in self.areas()])

        coords = dict()
        for area in self.areas():
            coords[area.name] = self.area_coords(area)
            
        result = dict()
        result['area_names'] = area_names
        result['modes'] = modes
        result['coords'] = coords
        result['coords_all'] = self.box_array.X, self.box_array.Y, self.box_array.Z
        result['dV_all'] = self.box_array.dV()

        return result
        
    def area_coords(self, area):
        return area.meshgrid()
        
    def to_area_coord(self, field, area):
        Nx = len(area.grid.xc)
        Ny = len(area.grid.yc)
        Nz = len(area.grid.zc)
        FIELD = np.zeros ((Nx, Ny, Nz), dtype=field.dtype)
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    if not area.belongs(i, j, k): continue
                    FIELD[i, j, k] = field[area.get_pos(i, j, k)]
        return FIELD
        
if __name__ == '__main__':

    Py_Ms = 800 * constants.kA_m
    Aex_Py = 13 * constants.pJ_m * 2
    Jex_Py = Aex_Py / Py_Ms**2
    alpha_Py = 5e-3
    Py_gamma_s = constants.gamma_s
    
    YIG_Ms = 140 * constants.kA_m
    YIG_alpha = 1 * 0.001
    YIG_gamma_s = constants.gamma_s
    Aex = 2 * 3.5 * constants.pJ_m
    YIG_Jex = Aex / YIG_Ms**2

    YIG = Material("YIG", YIG_Ms, YIG_Jex, YIG_alpha, YIG_gamma_s)
    Py  = Material("Py", Py_Ms, Jex_Py, alpha_Py, Py_gamma_s)

    #Bext = 5 * constants.mT
    #Bext = 20 * constants.mT
    Bext = 70 * constants.mT
    #Bext = 100 * constants.mT
    a = 70 * constants.nm
    #a = 50 * constants.nm
    b = 60 * constants.nm
    #b = 60 * constants.nm
    c = 10 * constants.nm
    #b = 20 * constants.nm
    Nx = 16
    Ny = 14
    Nz =  3

    def test_bare_resonator():
        #resonator = Area ("resonator", Grid(-a/2, a/2, -b/2, b/2, -c/2, c/2,
        #                                    Nx, Ny, Nz), YIG)
        #def mask_func_ellipse(x, y, z):
        #    return ((2*x/a)**2 + (2*y/b)**2 < 1)
        def mask_func_ellipse(x, y, z):
            return ((2*x/a)**2 + (1.9*y/b)**2 < 1)
        def mask_func_rect(x, y, z):
            return True
        resonator = Area ("resonator", Grid(-a/2, a/2, -b/2, b/2, -c/2, c/2,
                                            Nx, Ny, Nz), Py, mask_func_ellipse)
        model = CellArrayModel()
        model.add_area (resonator, 0.0, Bext, 0.0)
        n = model.relax_magnetisation()
        model.setup_dreibein(n)
        result = model.solve()

        
        box = model.box_array

        XR, YR, ZR = result['coords']['resonator']
        X, Y, Z = result['coords_all']
        print ("********")
        #print ("X = ", X)
        #print ("Y = ", Y)
        #print ("Z = ", Z)
        
        def show_mode(mode):
            #pl.figure()
            mx, my, mz = mode.m('resonator')
            ma_all, mb_all = mode.m_ab_all()
            Z_min = np.min(Z)
            Z_max = np.max(Z)
            Y_max = np.max(Y)
            Y_min = np.min(Y)
            #for z in [Z_min, Z_max, 0.5 * (Z_min + Z_max)]:
            for z in [0.5 * (Z_min + Z_max)]:
              i_0 = np.argmin(np.abs(z - Z))
              i_z = [t for t in range(len(Z)) if abs(Z[t] - Z[i_0]) < 1e-6]
              #print ("i_z = ", i_z)
              if len(i_z) < 1: continue
              ma_z = np.array([ma_all[t] for t in i_z])
              mb_z = np.array([mb_all[t] for t in i_z])
              m_scale = max(np.max(np.abs(ma_z)), np.max(np.abs(mb_z)))
              X_z = np.array([X[t] for t in i_z])
              Y_z = np.array([Y[t] for t in i_z])
              Z_z = np.array([Z[t] for t in i_z])

              pl.figure()
              #print ("X_z = ", X_z, "Y_z = ", Y_z)
              #print ("Z_z = ", Z_z)
              pl.tripcolor(X_z, Y_z, np.abs(ma_z),
                           vmin=0.0, vmax=m_scale)
              dx = 1.1 * (np.max(X_z) - np.min(X_z))
              pl.tripcolor(X_z + dx, Y_z, np.abs(mb_z),
                           vmin=0.0, vmax=m_scale)
              pl.gca().set_aspect('equal', 'box')
              pl.colorbar()
              pl.title(r"$m_a$ @ $z = %g$ $f = %g + i %g$" % (z, mode.f.real,
                                                       mode.f.imag))
              #pl.figure()
              #pl.tripcolor(X_z, Y_z, np.abs(mb_z))
              #pl.gca().set_aspect('equal', 'box')
              #pl.colorbar()
              #pl.title(r"$m_b$ @ $z = %g$ $f = %g$" % (z, mode.f))

            for y in [Y_min, 0.5 * (Y_min + Y_max)]:
              i_0 = np.argmin(np.abs(y - Y))
              i_y = [t for t in range(len(Y)) if abs(Y[t] - Y[i_0]) < 1e-6]
              #print ("i_z = ", i_z)
              if len(i_y) < 1: continue
              ma_y = np.array([ma_all[t] for t in i_y])
              mb_y = np.array([mb_all[t] for t in i_y])
              X_y = np.array([X[t] for t in i_y])
              Y_y = np.array([Y[t] for t in i_y])
              Z_y = np.array([Z[t] for t in i_y])
              m_scale = max(np.max(np.abs(ma_y)), np.max(np.abs(mb_y)))

              pl.figure()
              #print ("X_z = ", X_z, "Y_z = ", Y_z)
              #print ("Z_z = ", Z_z)
              pl.tripcolor(X_y, Z_y, np.abs(ma_y), vmin=0.0, vmax=m_scale)
              dz = 1.1 * (np.max(Z_y) - np.min(Z_y))
              pl.tripcolor(X_y, Z_y + dz, np.abs(mb_y), vmin=0.0, vmax=m_scale)
              pl.gca().set_aspect('equal', 'box')
              pl.colorbar()
              pl.title(r"$m_a$ @ $y = %g$ $f = %g + i %g$" % (y,
                                                mode.f.real, mode.f.imag))
              #pl.figure()
              #pl.tripcolor(X_z, Y_z, np.abs(mb_z))
              #pl.gca().set_aspect('equal', 'box')
              #pl.colorbar()
              #pl.title(r"$m_b$ @ $z = %g$ $f = %g$" % (z, mode.f))

            pl.show()

            if False:
                pl.quiver(X, Z,
                      ma_all.real, mb_all.real, pivot='middle')
                pl.gca().set_aspect('equal', 'box')
                pl.title("omega = %g %g" % (mode.f.real, mode.f.imag))
                pl.xlim(-a - 5 * b, a + 5 * b)
                pl.figure()
                mxmax = np.max(np.abs(mx))
                mzmax = np.max(np.abs(mz))
                re = model.integrate(np.abs(ma_all.real)**2
                                 + np.abs(mb_all.real)**2, "resonator")
                im = model.integrate(np.abs(ma_all.imag)**2
                                 + np.abs(mb_all.imag)**2, "resonator")
                print ("ell = ", np.sqrt(np.abs(im/re)))
                ell2  = model.integrate(np.abs(mb_all)**2, 'resonator')
                ell2 /= model.integrate(np.abs(ma_all)**2, 'resonator')
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

            if False:
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

            if False:
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

            if False:
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
            show_mode(mode)
            #pl.show()
            if mode.f.real > 15.0: break
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
        model_bare.add_area(resonator_bare, 0.0, Bext, 0.0)
        
        model = CellArrayModel()
        model.add_area (resonator, 0.0, Bext, 0.0)
        model.add_area (slab,      0.0, Bext, 0.0)
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
        model.add_area (resonator, 0.0, Bext, 0.0)
        model.add_area (slab,      0.0, Bext, 0.0)
        #model.add_area (damp_l,    Bext)
        #model.add_area (damp_r,    Bext)

        model_bare.relax_magnetisation()
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

    test_bare_resonator()
    #test_coupled_resonator()
    #test_response()


    



