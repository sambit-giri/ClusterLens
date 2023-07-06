import numpy as np 
from time import time 
from scipy.integrate import simps, quad
from scipy.interpolate import splev, splrep
# from astropy import units

from .constants import *
from .cosmo import Cosmology
from .instrument import Telescope
from .galaxy import GalaxyBias

class WeightFunctions(Cosmology,GalaxyBias,Telescope):
    def __init__(self, param):
        super().__init__(param)  # Initialize the base class (InterfaceCCL)
        self.param = param
        self.set_cosmology(param)

    def GalaxyClustering(self, z, i, verbose=True):
        '''
        Galaxy Clustering Weight Function
        WGi(z) = b(z)*n_i(z)*H(c)/c
        '''
        if verbose: print('Estimating the Galaxy Clustering Weight Function...')
        tstart = time()
        nzi = self.galaxy_density_at_zbin_i()
        bz  = np.vectorize(lambda z: self.bias(z))
        WGi = np.vectorize(lambda z,i: nzi(z,i)*bz(z)*self.H(z)/self.param.code.c*km_per_m)
        if verbose: print('...done in {:.1f} s'.format(time()-tstart))
        return WGi(z,i)
    
    def WeakLensing(self, z, i, verbose=True, z_nbins=30, k=3):
        '''
        Weak Lensing Weight Function
        '''
        if verbose: print('Estimating the Weak Lensing Weight Function...')
        tstart = time()
        param = self.param
        cosmo = self.cosmo
        zmax  = param.code.zmax
        A = lambda z: (3/2)*(param.cosmo.h*100/self.param.code.c*km_per_m)**2*param.cosmo.Om*(1+z)*self.z_to_cdist(z)
        nzi = self.galaxy_density_at_zbin_i()
        z_to_cdist = self.z_to_cdist
        integrand = lambda zp,z: nzi(zp,i)*(1-z_to_cdist(z)/z_to_cdist(zp))
        solve_integral = np.vectorize(lambda z: quad(integrand, z, zmax, args=(z,))[0])
        Wgi = np.vectorize(lambda z: solve_integral(z)*A(z))
        # out = Wgi(z)
        # zz = np.logspace(np.log10(param.code.zmin),np.log10(param.code.zmax),50)
        zz = np.linspace(param.code.zmin,param.code.zmax,z_nbins)
        yy = Wgi(zz)
        out = splev(z, splrep(zz,yy,k=k))
        if verbose: print('...done in {:.1f} s'.format(time()-tstart))
        return out
    
    def IntrinsicAlignment(self, z, i, verbose=True):
        if verbose: print('Estimating the Weak Lensing Weight Function...')
        tstart = time()
        param = self.param
        cosmo = self.cosmo
        A = param.telescope.aia*param.telescope.cia*param.cosmo.Om*(self.H(z)/self.param.code.c*km_per_m)
        WIAi = lambda z: A
        out = WIAi(z)
        if verbose: print('...done in {:.1f} s'.format(time()-tstart))
        return out