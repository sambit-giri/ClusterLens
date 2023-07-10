import numpy as np 
from time import time 
from scipy import integrate #import quad, simpson, romb, trapezoid
from scipy.interpolate import splev, splrep
# from astropy import units

from .constants import *
# from .cosmo import Cosmology
# from .instrument import Telescope
# from .galaxy import GalaxyBias

class WeightFunctions():
    def __init__(self, param, Cosmology, GalaxyBias, Telescope):
        # super().__init__(param)  # Initialize the base class (InterfaceCCL)
        self.param = param
        self.Cosmology  = Cosmology(param)
        self.GalaxyBias = GalaxyBias(param)
        self.Telescope  = Telescope(param)
        # self.Cosmology.set_cosmology(param)

    def W_GalaxyClustering(self, z, i, verbose=True):
        '''
        Galaxy Clustering Weight Function
        WGi(z) = b(z)*n_i(z)*H(c)/c
        '''
        if verbose: print('Estimating the Galaxy Clustering Weight Function...')
        tstart = time()
        try: nzi = self.Telescope.nzi 
        except: nzi = self.Telescope.normalized_galaxy_density()
        bz  = np.vectorize(lambda z: self.GalaxyBias.bias(z))
        WGi = np.vectorize(lambda z,i: nzi(z,i)*bz(z)*self.Cosmology.H(z)/self.param.code.c*km_per_m)
        try: out = WGi(z,i)
        except: out = np.array([WGi(z,j) for j in i]).T
        if verbose: print('...done in {:.1f} s'.format(time()-tstart))
        return out
    
    def W_WeakLensing(self, z, i, verbose=True, z_nbins=30, k=3):
        '''
        Weak Lensing Weight Function
        '''
        if verbose: print('Estimating the Weak Lensing Weight Function...')
        tstart = time()
        param = self.param
        cosmo = self.Cosmology.cosmo
        zmax  = param.code.zmax
        z_to_cdist = self.Cosmology.z_to_cdist
        A = lambda z: (3/2)*(param.cosmo.h*100/self.param.code.c*km_per_m)**2*param.cosmo.Om*(1+z)*z_to_cdist(z)
        try: nzi = self.Telescope.nzi 
        except: nzi = self.Telescope.normalized_galaxy_density()
        integrand  = lambda zp,z: nzi(zp,i)*(1-z_to_cdist(z)/z_to_cdist(zp))
        integrator = param.code.integrator
        if integrator in ['simps', 'simpson', 'romb', 'trapezoid']:
            n_integrator = param.code.n_integrator
            zmax = param.code.zmax
            if integrator=='trapezoid': 
                lensing_efficiency = np.vectorize(lambda z: integrate.trapezoid(integrand(np.linspace(z,zmax,n_integrator)),np.linspace(z,zmax,n_integrator)))
            elif integrator=='romb':
                lensing_efficiency = np.vectorize(lambda z: integrate.romb(integrand(np.linspace(z,zmax,n_integrator)),np.linspace(z,zmax,n_integrator)))
            else:
                lensing_efficiency = np.vectorize(lambda z: integrate.simpson(integrand(np.linspace(z,zmax,n_integrator),z),np.linspace(z,zmax,n_integrator)))
        else:
            lensing_efficiency = np.vectorize(lambda z: integrate.quad(integrand, z, zmax, args=(z,))[0])
        Wgi = np.vectorize(lambda z: lensing_efficiency(z)*A(z))
        try:
            zz = self.Wg_dict['z']
            yy = self.Wg_dict[i]
        except:
            zz = np.linspace(param.code.zmin,param.code.zmax,z_nbins)
            yy = Wgi(zz)
            self.Wg_dict = {'z': zz, i: yy}
        out = splev(z, splrep(zz,yy,k=k))
        if verbose: print('...done in {:.1f} s'.format(time()-tstart))
        return out
    
    def W_IntrinsicAlignment(self, z, i, verbose=True):
        '''
        Intrinsic Alignment Weight Function.
        '''
        if verbose: print('Estimating the Intrinsic Alignment Weight Function...')
        tstart = time()
        param = self.param
        cosmo = self.Cosmology.cosmo
        aia = param.telescope.aia
        cia = param.telescope.cia
        nia = param.telescope.nia
        bia = param.telescope.bia
        try: nzi = self.Telescope.nzi 
        except: nzi = self.Telescope.normalized_galaxy_density()
        L     = lambda z: 1 # Incorrect model for bia!=0
        Lstar = lambda z: 1 # Incorrect model for bia!=0
        A = lambda z: -aia*cia*param.cosmo.Om*(self.Cosmology.H(z)/self.param.code.c*km_per_m)
        B = lambda z: (1+z)**nia*(L(z)/Lstar(z))**bia/self.Cosmology.D(z)
        WIAi = lambda z: A(z)*B(z)*nzi(z,i)
        out = WIAi(z)
        if verbose: print('...done in {:.1f} s'.format(time()-tstart))
        return out
    
    def W_CosmicShear(self, z, i, verbose=True, z_nbins=30, k=3):
        '''
        Cosmic Shear Weight Function.
        '''
        # WGi = lambda z, i: self.GalaxyClustering(z,i,verbose=verbose)
        Wgi  = lambda z, i: self.W_WeakLensing(z,i,verbose=verbose, z_nbins=z_nbins, k=k)
        WIAi = lambda z, i: self.W_IntrinsicAlignment(z,i,verbose=verbose)
        return Wgi(z,i)+WIAi(z,i)
