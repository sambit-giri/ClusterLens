import numpy as np
from time import time
from scipy.interpolate import splev, splrep

import BCemu 
from .cosmo import *

class BCemuCCL(InterfaceCCL):
    def __init__(self, param, verbose=None):
        super().__init__(param, verbose=verbose)  # Initialize the base class (InterfaceCCL)
        self.param = param

    def Sk(self, z=0, k=None, verbose=True, beyond_emul_kmax=11):
        param = self.param
        if z is None: z = np.logspace(np.log10(param.code.zmin),np.log10(param.code.zmax),param.code.Nz)
        if k is None: 
            k  = np.logspace(np.log10(param.code.kmin),np.log10(param.code.kmax),param.code.Nk)
            kh = k/param.cosmo.h 
        bfcemu = BCemu.BCM_7param(Ob=param.cosmo.Ob, Om=param.cosmo.Om, verbose=verbose)
        bcmdict = param.bcemu.__dict__
        k_eval = np.logspace(np.log10(0.035),np.log10(12.5),50)
        p_eval = bfcemu.get_boost(z, bcmdict, k_eval)
        p_tck  = splrep(np.log10(k_eval*param.cosmo.h), p_eval, k=1)
        if beyond_emul_kmax=='trim_beyond_kmax' or beyond_emul_kmax==1:
            Sk_fn  = np.vectorize(lambda k: splev(np.log10(k),p_tck) if k/param.cosmo.h<=12.51740232 else 1)
        elif beyond_emul_kmax=='trim_above_unity' or beyond_emul_kmax==2:
            Sk_fn  = np.vectorize(lambda k: splev(np.log10(k),p_tck) if splev(np.log10(k),p_tck)<=1 else 1)
        else:
            Sk_fn  = lambda k: splev(np.log10(k),p_tck)
        return {
                'Sk': Sk_fn(k), 'k': k, 
                #'p_eval': p_eval, 'k_eval': k_eval*param.cosmo.h
                }
    
    def power_spectrum(self, param=None, k=None, z=None, a=None):
        return super().power_spectrum(param, k, z, a)