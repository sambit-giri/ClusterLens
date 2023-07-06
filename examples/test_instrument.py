import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm' #'dejavuserif' #'stix' #
import ClusterLens

# Parameters 
param = ClusterLens.par()

# Instrument Response using Euclid 2020 setup parameters
Euclid = ClusterLens.Euclid2020(param)

# zs = np.logspace(np.log10(param.code.zmin),np.log10(param.code.zmax),500)
zs = np.linspace(param.code.zmin,param.code.zmax,500)
nz_i = Euclid.galaxy_density_at_zbin_i()

fig, ax = plt.subplots(1,1,figsize=(7,6))
for i,zi in enumerate(param.telescope.z_edges[:-1]):
    ax.plot(zs, nz_i(zs,i), label='$i={}$'.format(i))
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$n^g_i(z)$', fontsize=15)
ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
ax.legend()
plt.tight_layout()
plt.show()