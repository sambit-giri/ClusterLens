import numpy as np 
from time import time
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm' #'dejavuserif' #'stix' #

import ClusterLens
from ClusterLens import InterfaceCCL, GalaxyBias, Euclid2020

# Parameters 
param = ClusterLens.par()
param.code.lmin = 10
param.code.lmax = 5000
param.code.Nl = 20
param.code.integrator  = "simpson"
param.code.n_integrator = 30
param.telescope.l_min = 10
param.telescope.l_max = 5000
# param.telescope.l_nbins = 20
# print(param.code.__dict__)

# Angular Coefficient
param.telescope.l_nbins = 10
AngCoeff1 = ClusterLens.AngularCoefficients(param, InterfaceCCL, GalaxyBias, Euclid2020)
C_LL_3D_1 = AngCoeff1.C_LL()
C_GG_3D_1 = AngCoeff1.C_GG()
C_GL_3D_1 = AngCoeff1.C_GL()

param.telescope.l_nbins = 30
AngCoeff2 = ClusterLens.AngularCoefficients(param, InterfaceCCL, GalaxyBias, Euclid2020) 
C_LL_3D_2 = AngCoeff2.C_LL()
C_GG_3D_2 = AngCoeff2.C_GG()
#C_GL_3D_2 = AngCoeff2.C_GL()
      
# Plot
fig, axs = plt.subplots(1,3,figsize=(15,5))
ax = axs[0]; ax.set_title('Weak Lensing or Cosmic Shear')
ells_1 = C_LL_3D_1['ells']; print(ells_1.shape)
ells_2 = C_LL_3D_2['ells']; print(ells_2.shape)
for i,zi in enumerate(param.telescope.z_edges[:-1]):
    ax.loglog(ells_1, C_LL_3D_1[i,i]*ells_1*(ells_1+1), c='C{}'.format(i), label='$i={}$'.format(i))
    ax.loglog(ells_2, C_LL_3D_2[i,i]*ells_2*(ells_2+1), c='k', ls='--')
ax.set_xlabel('$l$', fontsize=15)
ax.set_ylabel('$l(l+1)C^{LL}_{ii}$', fontsize=15)
ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
ax.legend()
ax = axs[1]; ax.set_title('Galaxy Clustering')
ells_1 = C_GG_3D_1['ells']; print(ells_1.shape)
ells_2 = C_GG_3D_2['ells']; print(ells_2.shape)
for i,zi in enumerate(param.telescope.z_edges[:-1]):
    ax.loglog(ells_1, C_GG_3D_1[i,i]*ells_1*(ells_1+1), c='C{}'.format(i), label='$i={}$'.format(i))
    ax.loglog(ells_2, C_GG_3D_2[i,i]*ells_2*(ells_2+1), c='k', ls='--')
ax.set_xlabel('$l$', fontsize=15)
ax.set_ylabel('$l(l+1)C^{GG}_{ii}$', fontsize=15)
ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
# ax.legend()
ax = axs[2]; ax.set_title('Galaxy-Lensing')
ells_1 = C_GL_3D_1['ells']; print(ells_1.shape)
# ells_2 = C_GL_3D_2['ells']; print(ells_2.shape)
for i,zi in enumerate(param.telescope.z_edges[:-1]):
    ax.loglog(ells_1, np.abs(C_GL_3D_1[i,i])*ells_1*(ells_1+1), c='C{}'.format(i), label='$i={}$'.format(i))
    # ax.loglog(ells_2, np.abs(C_GL_3D_2[i,i])*ells_2*(ells_2+1), c='k', ls='--')
ax.set_xlabel('$l$', fontsize=15)
ax.set_ylabel('$l(l+1)C^{GL}_{ii}$', fontsize=15)
ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
# ax.legend()
plt.tight_layout()
plt.show()