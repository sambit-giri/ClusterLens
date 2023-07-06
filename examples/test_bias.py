import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm' #'dejavuserif' #'stix' #
import ClusterLens

# Parameters 
param = ClusterLens.par()

# Galaxy bias
Bias = ClusterLens.GalaxyBias(param)

# zs = np.logspace(np.log10(param.code.zmin),np.log10(param.code.zmax),500)
zs = np.linspace(param.code.zmin,param.code.zmax,500)
bias_ISTF = Bias.bias(zs, kind='IST:F')
bias_Tutusaus2020 = Bias.bias(zs, kind='Tutusaus2020')

fig, ax = plt.subplots(1,1,figsize=(7,6))
ax.plot(zs, bias_ISTF, label='IST:F+(2020)')
ax.plot(zs, bias_Tutusaus2020, label='Tutusaus+(2020)')
ax.plot(zs, np.sqrt(1+zs), ls=':', label='$\sqrt{1+z}$')
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$b(z)$', fontsize=15)
ax.legend()
plt.tight_layout()
plt.show()