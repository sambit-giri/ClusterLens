import numpy as np 
from time import time 
from tqdm import tqdm
from scipy import integrate #import quad, simpson, romb, trapezoid
from scipy.interpolate import splev, splrep

from .constants import *
from .weight_func import WeightFunctions

class AngularCoefficients(WeightFunctions):
    def __init__(self, param, Cosmology, GalaxyBias, Telescope, z_nbins=100, integrator=None, pk_suppression=None):
        '''
        Angular Coefficients computed using the Limber approximation.
        '''
        super().__init__(param, Cosmology, GalaxyBias, Telescope)  # Initialize the base class (InterfaceCCL)
        # self.param = param
        # # self.set_cosmology(param)
        self.integrator = integrator if integrator is not None else param.code.integrator
        self.z_nbins = z_nbins if z_nbins is not None else {'W': param.code.n_integrator, 'C': param.code.n_integrator}
        self.Cosmology.pk_suppression = pk_suppression if pk_suppression is not None else param.cosmo.pk_suppression
        if param.code.verbose:
            if self.Cosmology.pk_suppression is None: print('DMO | No suppression.')
            else: print(self.Cosmology.pk_suppression)

    def C_AB(self, W_A, W_B, Pkl, integrator=None): 
        '''
        General coefficient calculator.
        '''
        param = self.param
        if integrator is None: integrator = self.integrator
        try: cosmo = self.Cosmology.cosmo 
        except: cosmo = self.Cosmology.set_cosmology(param)
        # Ez = lambda z: self.H(z)/param.cosmo.h/100
        c_kmps = param.code.c/km_per_m
        zmin, zmax = param.code.zmin, param.code.zmax
        z_to_cdist = self.Cosmology.z_to_cdist
        integrand = lambda z,l: W_A(z)*W_B(z)/self.Cosmology.H(z)/(z_to_cdist(z))**2*Pkl(z,l)
        if integrator in ['simps', 'simpson', 'romb', 'trapezoid']:
            try: z_nbins = self.z_nbins['C']
            except: z_nbins = self.z_nbins
            zs = np.linspace(param.code.zmin,param.code.zmax,z_nbins)
            ells = np.logspace(np.log10(param.code.lmin),np.log10(param.code.lmax),param.code.Nl)
            itg = integrand(zs,ells[:,None])
            if integrator=='trapezoid': CABij_array = c_kmps*integrate.trapezoid(itg, zs) 
            elif integrator=='romb': CABij_array = c_kmps*integrate.romb(itg, zs) 
            else: CABij_array = c_kmps*integrate.simpson(itg, zs)
            CABij = lambda l: splev(l, splrep(ells,CABij_array))
        else:
            CABij = np.vectorize(lambda l: c_kmps*integrate.quad(integrand, zmin, zmax, args=(l,))[0])
        return CABij

    def dict_to_2Darray(self, C_AB_dict):
        param = self.param 
        C_AB_array = C_AB_dict['ells']
        for i,zi in enumerate(param.telescope.z_edges[:-1]):
            for j,zj in enumerate(param.telescope.z_edges[:i+1]):
                C_AB_array = np.vstack((C_AB_array,C_AB_dict[i,j]))
        return C_AB_array.T
    
    def C_LL(self, ells=None, verbose=True, **kwargs):
        '''
        Cosmic Shear Angular Correlation.
        '''
        param = self.param
        if ells is None:
            ells = np.logspace(np.log10(param.telescope.l_min),np.log10(param.telescope.l_max),param.telescope.l_nbins)
        Pkl_interp = self.Cosmology.create_Pkl_interpolator()

        try: z_nbins = self.z_nbins['W']
        except: z_nbins = self.z_nbins
        WL_dict = self.get_weight_function_dict('L', verbose=verbose, z_nbins=z_nbins)
        zs = WL_dict['z']; #print('z_nbins = {}, {}'.format(z_nbins, zs.shape))

        if verbose: print('Estimating the Cosmic Shear Angular Coefficient...')
        C_LL_3D = {'ells': ells}
        tstart, count = time(), 0
        for i,zi in enumerate(param.telescope.z_edges[:-1]):
            for j,zj in enumerate(param.telescope.z_edges[:i+1]):
                count += 1
                WL_i = lambda z: splev(z, splrep(zs,WL_dict[i])) 
                WL_j = lambda z: splev(z, splrep(zs,WL_dict[j])) 
                C_LL = self.C_AB(WL_i, WL_j, Pkl_interp)
                C_LL_3D[i,j] = C_LL(ells)
                if verbose and i==j: print('{0:2d}| {1:2d}, {2:2d}| time elapsed = {5:.1f} s'.format(count, i, j, zi, zj, time()-tstart))
        if verbose: print('...done')
        return C_LL_3D

    def C_GG(self, ells=None, verbose=True, **kwargs):
        '''
        Galaxy Clustering Angular Correlation.
        '''
        param = self.param
        if ells is None:
            ells = np.logspace(np.log10(param.telescope.l_min),np.log10(param.telescope.l_max),param.telescope.l_nbins)
        Pkl_interp = self.Cosmology.create_Pkl_interpolator()

        try: z_nbins = self.z_nbins['W']
        except: z_nbins = self.z_nbins
        WG_dict = self.get_weight_function_dict('G', verbose=verbose, z_nbins=z_nbins)
        zs = WG_dict['z']; #print('z_nbins = {}, {}'.format(z_nbins, zs.shape))

        if verbose: print('Estimating the Galaxy Clustering Angular Coefficient...')
        C_GG_3D = {'ells': ells}
        tstart, count = time(), 0
        for i,zi in enumerate(param.telescope.z_edges[:-1]):
            for j,zj in enumerate(param.telescope.z_edges[:i+1]):
                count += 1
                WG_i = lambda z: splev(z, splrep(zs,WG_dict[i])) 
                WG_j = lambda z: splev(z, splrep(zs,WG_dict[j])) 
                C_GG = self.C_AB(WG_i, WG_j, Pkl_interp)
                C_GG_3D[i,j] = C_GG(ells)
                if verbose and i==j: print('{0:2d}| {1:2d}, {2:2d}| time elapsed = {5:.1f} s'.format(count, i, j, zi, zj, time()-tstart))
        if verbose: print('...done')
        return C_GG_3D
    
    def C_GL(self, ells=None, verbose=True, **kwargs):
        '''
        Galaxy-Lensing Angular Correlation.
        '''
        param = self.param
        if ells is None:
            ells = np.logspace(np.log10(param.telescope.l_min),np.log10(param.telescope.l_max),param.telescope.l_nbins)
        Pkl_interp = self.Cosmology.create_Pkl_interpolator()

        try: z_nbins = self.z_nbins['W']
        except: z_nbins = self.z_nbins
        WG_dict = self.get_weight_function_dict('G', verbose=verbose, z_nbins=z_nbins)
        WL_dict = self.get_weight_function_dict('L', verbose=verbose, z_nbins=z_nbins)
        # zs = WG_dict['z']; print('z_nbins = {}, {}'.format(z_nbins, zs.shape))
        
        if verbose: print('Estimating the Galaxy-Lensing Angular Coefficient...')
        C_GL_3D = {'ells': ells}
        tstart, count = time(), 0
        for i,zi in enumerate(param.telescope.z_edges[:-1]):
            for j,zj in enumerate(param.telescope.z_edges[:-1]):
                count += 1
                WG_i = lambda z: splev(z, splrep(WG_dict['z'],WG_dict[i])) 
                WL_j = lambda z: splev(z, splrep(WL_dict['z'],WL_dict[j])) 
                C_GL = self.C_AB(WG_i, WL_j, Pkl_interp)
                C_GL_3D[i,j] = C_GL(ells)
                if verbose and i==j: print('{0:2d}| {1:2d}, {2:2d}| time elapsed = {5:.1f} s'.format(count, i, j, zi, zj, time()-tstart))
        if verbose: print('...done')
        return C_GL_3D
    
    def get_weight_function_dict(self, kind, verbose=True, z_nbins=100):
        param = self.param
        t0 = time() 

        if kind.upper() in ['G', 'GC', 'GALAXY']:
            try: 
                W_dict = self.WG_dict
                print('Using the Galaxy Clustering Weight Function computed before.')
            except:
                zs = np.linspace(param.code.zmin,param.code.zmax,z_nbins)
                self.WG_dict = {i: self.W_GalaxyClustering(zs,i,verbose=0) for i,zi in enumerate(param.telescope.z_edges[:-1])}
                self.WG_dict['z'] = zs
                W_dict = self.WG_dict
                if verbose: print('Galaxy Clustering Weight Function for the {} tomographic bins estimated in {:.1f} s.'.format(len(param.telescope.z_edges[:-1]),time()-t0))
                
        if kind.upper() in ['L', 'WL', 'LENSING']:
            try: 
                W_dict = self.WL_dict
                print('Using the Cosmic Shear Weight Function computed before.')
            except:
                zs = np.linspace(param.code.zmin,param.code.zmax,z_nbins)
                self.WL_dict = {i: self.W_CosmicShear(zs,i,verbose=0) for i,zi in enumerate(param.telescope.z_edges[:-1])}
                self.WL_dict['z'] = zs
                W_dict = self.WL_dict
                if verbose: print('Cosmic Shear Weight Function for the {} tomographic bins estimated in {:.1f} s.'.format(len(param.telescope.z_edges[:-1]),time()-t0))
        
        return W_dict
