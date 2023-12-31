import numpy as np
from time import time
from scipy.interpolate import RegularGridInterpolator, interp1d, splev, splrep

import pyccl as ccl
# import pyhmcode

class InterfaceCCL:
    def __init__(self, param, verbose=None, ps_suppression=None):
        self.verbose = param.code.verbose if verbose is None else verbose
        self.param = param
        self.cosmo = self.set_cosmology(param) 
        self.ps_suppression = ps_suppression if ps_suppression is not None else param.cosmo.ps_suppression
        # self.prepare_z_cdist_table()
        self.prepare_D_table()

    def set_cosmology(self, param):
        # Define the cosmology
        cosmo = ccl.Cosmology(
                        Omega_c = param.cosmo.Om-param.cosmo.Ob,
                        Omega_b = param.cosmo.Ob,
                        h   = param.cosmo.h,
                        n_s = param.cosmo.ns,
                        sigma8  = param.cosmo.s8,
                        A_s     = param.cosmo.As,
                        Omega_k = param.cosmo.Ok,
                        Omega_g = None,
                        Neff = param.cosmo.Neff,
                        m_nu = param.cosmo.mnu,
                        mass_split = 'normal',
                        w0 = param.cosmo.w0,
                        wa = param.cosmo.wa,
                        T_CMB   = param.cosmo.Tcmb,
                        mu_0    = 0,
                        sigma_0 = 0,
                        transfer_function      = param.cosmo.transfer_function, 
                        matter_power_spectrum  = param.cosmo.matter_power_spectrum, 
                        baryons_power_spectrum = param.cosmo.baryons_power_spectrum, 
                        mass_function          = param.cosmo.mass_function, 
                        halo_concentration     = param.cosmo.halo_concentration,
                        bcm_log10Mc = None,
                        bcm_etab    = None,
                        bcm_ks      = None,
                    )
        return cosmo 
    
    def a_to_z(self, a): return 1/a-1

    def z_to_a(self, z): return 1/(1+z)

    def prepare_D_table(self, k=3, z_nbins=30):
        if self.verbose: print('Creating z vs D(z) table...')
        tstart = time()
        # zz = np.linspace(self.param.code.zmin,self.param.code.zmax,z_nbins)
        # kk = self.param.code.kmin
        try: pk_dict = self.pk_dict
        except: pk_dict = self.power_spectrum() # k=self.param.code.kmin, z=zz
        yy = np.sqrt(pk_dict['pk_lin'][:,0]/pk_dict['pk_lin'][0,0])
        D_table = lambda z: splev(z, splrep(pk_dict['z'],yy,k=k))
        self.D_table = D_table
        if self.verbose: print('...table created in {:.1f} s'.format(time()-tstart))
        return D_table

    def D(self, z): 
        try: return self.D_table(z)
        except: 
            self.prepare_D_table()
            return self.D_table(z)

    def H(self, z):
        try: cosmo = self.cosmo 
        except: cosmo = self.set_cosmology(self.param)
        return cosmo.h_over_h0(self.z_to_a(z))*cosmo['h']*100 #km/s/Mpc
    
    def prepare_z_cdist_table(self):
        try: cosmo = self.cosmo 
        except: cosmo = self.set_cosmology(self.param)
        if self.verbose: print('Creating z vs cdist table...')
        tstart = time()
        param = self.param 
        zs = np.logspace(np.log10(param.code.zmin),np.log10(param.code.zmax),500)
        rs = cosmo.comoving_radial_distance(self.z_to_a(zs))  #Mpc
        fn_log10dcom = interp1d(np.log10(zs), np.log10(rs), fill_value="extrapolate")
        self.fn_log10dcom = fn_log10dcom
        if self.verbose: print('...table created in {:.1f} s'.format(time()-tstart))
        return None

    def z_to_cdist(self, z):
        try: cosmo = self.cosmo 
        except: cosmo = self.set_cosmology(self.param)
        cdist = cosmo.comoving_radial_distance(self.z_to_a(z))  #Mpc
        # cdist = 10**self.fn_log10dcom(np.log10(z))
        return cdist  

    def power_spectrum(self, param=None, k=None, z=None, a=None):
        try:
            pk_dict = self.pk_dict
        except:
            param = self.param
            if z is None: z = np.logspace(np.log10(param.code.zmin),np.log10(param.code.zmax),param.code.Nz)
            if k is None: k = np.logspace(np.log10(param.code.kmin),np.log10(param.code.kmax),param.code.Nk)
            if a is None: a = 1/(1+z)
            try: cosmo = self.set_cosmology(param)
            except: cosmo = self.cosmo
            # Compute the linear matter power spectrum
            pk_l  = ccl.linear_matter_power(cosmo, k, a)
            # Compute the non-linear matter power spectrum
            pk_nl = ccl.nonlin_matter_power(cosmo, k, a)
            ps_suppress = self.ps_suppression
            # if ps_suppress is not None: print(pk_nl.shape,ps_suppress(z,k).shape)
            pk_dict = {
                    'z': z,
                    'k': k,
                    'pk_lin': pk_l,
                    'pk_nonlin': pk_nl if ps_suppress is None else pk_nl*ps_suppress(z,k),
                    }
            self.pk_dict = pk_dict
        return pk_dict

    def ell_to_kl(self, ell, z=None, a=None):
        param = self.param
        if z is None: z = np.logspace(np.log10(param.code.zmin),np.log10(param.code.zmax),param.code.Nz)
        if a is None: a = 1/(1+z)
        try: cosmo = self.set_cosmology(param)
        except: cosmo = self.cosmo
        rz = cosmo.comoving_radial_distance(a)
        kl = (ell+1/2)[:,None]/rz[None,:] if type(ell)==np.ndarray else (ell+1/2)/rz[None,:]
        return kl
    
    def create_Pk_interpolator(self, param=None, method='linear',
                                ks=None, zs=None, a=None):
        if self.verbose: print('Creating the Pk_interpolator...')
        param = self.param
        if zs is None: zs = np.logspace(np.log10(param.code.zmin),np.log10(param.code.zmax),param.code.Nz)
        if ks is None: ks = np.logspace(np.log10(param.code.kmin),np.log10(param.code.kmax),param.code.Nk)
        if a is None: a = 1/(1+zs)
        try: cosmo = self.set_cosmology(param)
        except: cosmo = self.cosmo
        pk   = self.power_spectrum(k=ks, z=zs)
        if self.verbose:
            print('z range:', zs.min(), zs.max())
            print('k range:', ks.min(), ks.max())
        logPk_interp = RegularGridInterpolator((np.log10(zs), np.log10(ks)), 
                                    np.log10(pk['pk_nonlin']), method=method)
        Pk_interp = np.vectorize(lambda z,k: 10**logPk_interp([np.log10(z),np.log10(k)]))
        self.logPk_interp = logPk_interp
        self.Pk_interp = Pk_interp
        if self.verbose: print('...done')
        return Pk_interp 
    
    def create_Pkl_interpolator(self, param=None, method='linear', 
                                ells=None, ks=None, zs=None, a=None):
        if self.verbose: print('Creating the Pk_interpolator...')
        param = self.param
        if zs is None: zs = np.logspace(np.log10(param.code.zmin),np.log10(param.code.zmax),param.code.Nz)
        if ks is None: ks = np.logspace(np.log10(param.code.kmin),np.log10(param.code.kmax),param.code.Nk)
        if a is None: a = 1/(1+zs)
        try: cosmo = self.set_cosmology(param)
        except: cosmo = self.cosmo
        ells = np.logspace(np.log10(param.code.lmin),np.log10(param.code.lmax),param.code.Nl)
        k_ls = self.ell_to_kl(ells, z=zs)
        if self.verbose:
            print('z range:', zs.min(), zs.max())
            print('l range:', ells.min(), ells.max())
            print('k(z,l) :', k_ls.min(), k_ls.max())
        logkl_interp = RegularGridInterpolator((np.log10(zs), np.log10(ells)), 
                                    np.log10(k_ls).T, method=method)
        kl_interp = np.vectorize(lambda z,l: 10**logkl_interp([np.log10(z),np.log10(l)]))
        Pk_interp = self.create_Pk_interpolator(param=param, method=method,
                                ks=ks, zs=zs, a=a)
        Pkl_interp = np.vectorize(lambda z,l: Pk_interp(z, kl_interp(z,l).squeeze()))
        # Pkl_zl_interp = np.vectorize(lambda zl: Pk_interp(zl[0], kl_interp(zl[0],zl[1]).squeeze()))
        self.Pkl_interp = Pkl_interp
        if self.verbose: print('...done')
        return Pkl_interp


class Cosmology(InterfaceCCL):
    def __init__(self, param, verbose=None):
        super().__init__(param, verbose=verbose)  # Initialize the base class (InterfaceCCL)
        self.param = param