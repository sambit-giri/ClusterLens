import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm' #'dejavuserif' #'stix' #
import ClusterLens

# Parameters 
param = ClusterLens.par()

# Power spectrum
zs = np.logspace(np.log10(param.code.zmin),np.log10(param.code.zmax),param.code.Nz)
ks = np.logspace(np.log10(param.code.kmin),np.log10(param.code.kmax),param.code.Nk)
cosmo = ClusterLens.BCemuCCL(param) 
zplot = [0.0,0.5,1,1.5,2]
k1 = np.logspace(np.log10(0.025),np.log10(8.25),50)
Sk1 = {zi: cosmo.Sk(z=zi, k=k1, verbose=False) for i,zi in enumerate(zplot)}
k2 = np.logspace(np.log10(0.025),np.log10(80.25),100)
Sk2 = {zi: cosmo.Sk(z=zi, k=k2, verbose=False, beyond_emul_kmax=1) for i,zi in enumerate(zplot)}

# Plot
fig, ax = plt.subplots(1,1,figsize=(7,6))
for i,zi in enumerate(zplot):
    # ax.semilogx(Sk1[zi]['k_eval'], Sk1[zi]['p_eval'], ls='-', lw=4, alpha=0.3)
    ax.semilogx(Sk1[zi]['k'], Sk1[zi]['Sk'], ls='-', lw=4, label='$z={}$'.format(zi))
    ax.semilogx(Sk2[zi]['k'], Sk2[zi]['Sk'], ls='--', c='k')
ax.set_xlabel('$k$ [1/Mpc]', fontsize=15)
ax.set_ylabel('$S_k$', fontsize=15)
ax.axis([0.02,105,0.74,1.04])
ax.legend()
plt.tight_layout()
plt.show()