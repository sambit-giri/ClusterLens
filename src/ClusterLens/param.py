"""
External Parameters
"""

import numpy as np
#from numba import jit

class Bunch(object):
    """
    translates dic['name'] into dic.name 
    """

    def __init__(self, data):
        self.__dict__.update(data)

#@jit(nopython=True)
def step_func(x, bin_edges, bin_values):
    bin_centres = bin_edges[1:]/2 + bin_edges[:-1]/2
    func = lambda x: bin_values[np.abs(bin_centres-x).argmin()]
    vfunc = np.vectorize(func)
    return vfunc(x)

def Tutusaus2020_fit(z,A,B,C,D):
    return A + B/(1+np.exp(-(z-D)*C))

def Euclid_par():
    par = {
        "z_mean" : 0.9, 
        "z_edges": np.array([0.001,0.42,0.56,0.68,0.79,0.90,1.02,1.15,1.32,1.58,2.50]),
        "nzi_file": 'Euclid2020_nzi.pkl', # None, #
        "bias": 'IST:F', #{'IST:F': np.array([1.10,1.22,1.27,1.32,1.36,1.40,1.44,1.50,1.57,1.74])}, # {"Tutusaus2020": np.array([1.0,2.5,2.8,1.6])}

        "aia": 1.72,
        "cia": 0.0134,
        "nia": -0.41,
        "bia": 0,

        "l_min": 10,
        "l_max": 5000,
        "l_nbins": 20,
    }
    return Bunch(par)

def code_par():
    par = {
        "verbose": False, 
        "zmin": 0.001,                # min redshift
        "zmax": 4.000,                # max redshift
        "Nz"  : 80,                   # number of z bins
        "kmin": 2e-4,                 # 1/Mpc, min wavenumber
        "kmax": 2e3,                  # 1/Mpc, max wavenumber
        "Nk"  : 100,                  # number of k bins
        "lmin": 1,                    # 1/Mpc, min wavenumber
        "lmax": 5000,                 # 1/Mpc, max wavenumber
        "Nl"  : 200,                  # number of k bins
        "c"   : 3e8,                  # m/s
        "Cl_struct": '2D',
        "integrator": 'quad',         # options: quad, trapezoid, simpson
        "n_integrator": 500,          # number of bins used in e.g. trapezoid and simpson
        }
    return Bunch(par)


def cosmo_par():
    par = {
        "Om" : 0.32, 
        "Ob" : 0.05, 
        "h"  : 0.67,  # (km/s)/Mpc
        "ns" : 0.96, 
        "s8" : None, #0.816, 
        "As" : 2.1e-9, 
        "mnu": 0.06, 
        "w0" : -1, 
        "wa" : 0, 
        "Ode": 0.68, 
        "Ok" : 0,
        "Neff" : 3.046,
        "Tcmb" : 2.725,
        }
    return Bunch(par)

def bcemu_par():
    bcmdict = {'log10Mc': 13.32,
                'mu'     : 0.93,
                'thej'   : 4.235,  
                'gamma'  : 2.25,
                'delta'  : 6.40,
                'eta'    : 0.15,
                'deta'   : 0.14,
                }
    return Bunch(bcmdict)

def par():
    par = Bunch({
        "cosmo": cosmo_par(),
        "code" : code_par(),
        "telescope": Euclid_par(),
        "bcemu": bcemu_par(),
        })
    return par

