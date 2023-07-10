import numpy as np
from time import time 
import pickle 
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import simps, quad
from scipy.interpolate import splev, splrep

class Euclid2020:
    def __init__(self, param, verbose=None):
        self.verbose = param.code.verbose if verbose is None else verbose
        self.param = param
        # self.cosmo = self.set_cosmology(param) 

    def analitycal_density_at_z(self, **kwargs):
        '''
        Analitycal density of galaxies at redshift z.
        '''
        kind = kwargs.get('kind', 'exp')
        if kind.lower()=='exp':
            z0 = kwargs.get('z0', 0.9*np.sqrt(2))
            nz = lambda z: (z/z0)**2 * np.exp(-(z/z0)**(3/2))
        else: 
            print('{} kind of density distribution of galaxies is not implemented'.format(kind))
            nz = None 
        self.nz = nz
        return nz 
    
    def prob_galaxy_z_observed_zp(self, **kwargs):
        '''
        The instrumental response giving the probability of redshift z galaxies observed at zp.
        '''
        cb = kwargs.get('cb', 1.0)
        zb = kwargs.get('zb', 0.0)
        sigb = kwargs.get('sigb', 0.05)
        co = kwargs.get('co', 1.0)
        zo = kwargs.get('zo', 0.1)
        sigo = kwargs.get('sigo', 0.05)
        fout = kwargs.get('fout', 0.1) 
        p_ph_i = lambda z, zp, ci, zi, sigi: 1/(np.sqrt(2*np.pi)*sigi*(1+z))*np.exp(-0.5*((z-ci*zp-zi)/(sigi*(1+z)))**2)
        p_ph = lambda z, zp: (1-fout)*p_ph_i(z,zp,cb,zb,sigb) + fout*p_ph_i(z,zp,co,zo,sigo)
        self.p_ph = p_ph
        return p_ph
    
    def galaxy_density_at_zbin_i(self, **kwargs):
        verbose = self.verbose
        try: nz = self.nz 
        except: nz = self.analitycal_density_at_z(**kwargs)
        try: p_ph = self.p_ph
        except: p_ph = self.prob_galaxy_z_observed_zp(**kwargs)
        z_i = self.param.telescope.z_edges
        if verbose:
            print('{} tomographic bins used with the following bin edges:'.format(len(z_i)-1))
            print(['{:.3f}'.format(zi) for zi in z_i])
        integrand = lambda zp, z: nz(z) * p_ph(z, zp)
        numerator_integral = np.vectorize(lambda z,z1,z2: quad(integrand, z1, z2, args=(z,))[0])
        def solve_integral(zmin, zmax, z1, z2):
            inner_integral = lambda z,z1,z2: numerator_integral(z,z1,z2)  # Define the limits of integration
            result, error = quad(numerator_integral, zmin, zmax, args=(z1,z2,)) # Perform the outer integral
            return result
        tstart = time()
        nz_i_norm = {}
        for i,zi in enumerate(z_i[:-1]):
            integral_result = solve_integral(min(z_i), max(z_i), z_i[i], z_i[i+1])
            nz_i_norm[i] = integral_result
        if verbose: print('Normalisation values estimated in {:.1f} s'.format(time()-tstart))
        self.nz_i_norm = nz_i_norm
        nz_i = lambda z,i: numerator_integral(z,z_i[i],z_i[i+1])/nz_i_norm[i]
        self.nz_i = nz_i
        return nz_i

    def normalized_galaxy_density(self, verbose=True, z_nbins=500):
        param = self.param 
        try:
            nzi_dict = self.nzi_dict
        except:
            try: 
                nzi_dict = pickle.load(open(param.telescope.nzi_file, 'rb'))
                if verbose: print('data read from {} file'.format(param.telescope.nzi_file))
                self.nzi_dict = nzi_dict
            except:
                if verbose:
                    if param.telescope.nzi_file is None: print('No file name provided in param.')
                    else: print('Creating the file as it was not found...')
                zs = np.linspace(param.code.zmin,param.code.zmax,z_nbins)
                nz_i = self.galaxy_density_at_zbin_i()
                nzi_dict = {i: nz_i(zs,i) for i,zi in enumerate(param.telescope.z_edges[:-1])}
                nzi_dict['z'] = zs 
                self.nzi_dict = nzi_dict
                if param.telescope.nzi_file is not None:
                    pickle.dump(nzi_dict, open(param.telescope.nzi_file, 'wb'))
                    if verbose: print('...data saved as {}'.format(param.telescope.nzi_file))
        self.nzi = lambda z,i: splev(z, splrep(nzi_dict['z'], nzi_dict[i]))
        return self.nzi

class Telescope(Euclid2020):
    def __init__(self, param, verbose=None):
        super().__init__(param, verbose=verbose)  # Initialize the base class (InterfaceCCL)
        self.param = param