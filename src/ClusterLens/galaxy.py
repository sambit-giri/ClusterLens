import numpy as np

class GalaxyBias:
    def __init__(self, param, verbose=None):
        self.verbose = param.code.verbose if verbose is None else verbose
        self.param = param 

    def PiecewiseBias(self, b_values, z_edges=None, z_means=None):
        if z_edges is None: z_edges = self.param.telescope.z_edges
        if z_means is None: z_means = (z_edges[1:]+z_edges[:-1])/2
        bias_fct = np.vectorize(lambda z: b_values[np.abs(z_means-z).argmin()])
        return bias_fct 
    
    def ISTF(self):
        b_values = np.array([1.10,1.22,1.27,1.32,1.36,1.40,1.44,1.50,1.57,1.74])
        bias_fct = self.PiecewiseBias(b_values)
        self.bias_fct = bias_fct 
        return bias_fct 
    
    def Tutusaus2020(self):
        A,B,C,D = np.array([1.0,2.5,2.8,1.6])
        bias_fct = lambda z: A + B/(1+np.exp(-(z-D)*C))
        self.bias_fct = bias_fct 
        return bias_fct 
    
    def SQRT(self):
        bias_fct = lambda z: np.sqrt(1+z)
        self.bias_fct = bias_fct 
        return bias_fct 
    
    def bias(self, z, kind=None):
        if kind is None: kind = self.param.telescope.bias 
        if self.verbose: print('Using {} bias prescription'.format(kind))

        if kind.upper() in ['ISTF', 'IST:F', 'IST']: bias_fct = self.ISTF()
        elif kind.lower() == 'tutusaus2020': bias_fct = self.Tutusaus2020()
        elif kind.lower() == 'sqrt': bias_fct = self.SQRT()
        else:
            print('{} bias prescription is not implemented'.format(kind))
            bias_fct = lambda z: None

        return bias_fct(z)
