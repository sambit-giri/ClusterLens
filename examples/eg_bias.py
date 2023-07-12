import numpy as np 
from time import time 
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm' #'dejavuserif' #'stix' #
import ClusterLens

# Parameters 
param = ClusterLens.par()

# Galaxy bias
t0 = time()
Bias = ClusterLens.GalaxyBias(param)
print('Galaxy Bias class intialisation | runtime: {:.3f} s'. format(time()-t0)); t0 = time()

# zs = np.logspace(np.log10(param.code.zmin),np.log10(param.code.zmax),500)
zs = np.linspace(param.code.zmin,param.code.zmax,500)
t0 = time() 
bias_ISTF = Bias.bias(zs, kind='IST:F')
print('IST:F 2020 bias | runtime: {:.3f} s'. format(time()-t0)); t0 = time()
bias_Tutusaus2020 = Bias.bias(zs, kind='Tutusaus2020')
print('Tutusaus 2020 bias | runtime: {:.3f} s'. format(time()-t0)); t0 = time()

fig, ax = plt.subplots(1,1,figsize=(7,6))
ax.plot(zs, bias_ISTF, label='IST:F+(2020)')
ax.plot(zs, bias_Tutusaus2020, label='Tutusaus+(2020)')
ax.plot(zs, np.sqrt(1+zs), ls=':', label='$\sqrt{1+z}$')
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$b(z)$', fontsize=15)
ax.legend()
plt.tight_layout()
plt.show()