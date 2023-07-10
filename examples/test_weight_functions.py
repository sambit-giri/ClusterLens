import numpy as np 
from time import time
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm' #'dejavuserif' #'stix' #

import ClusterLens
from ClusterLens import InterfaceCCL, GalaxyBias, Euclid2020

# Parameters 
param = ClusterLens.par()
# param.code.verbose = True
# param.telescope.bias = 'sqrt'
param.code.integrator   = "simpson"
param.code.n_integrator = 50

# Weight funtions
WeightFunc = ClusterLens.WeightFunctions(param, InterfaceCCL, GalaxyBias, Euclid2020) 

# zs = np.logspace(np.log10(param.code.zmin),np.log10(param.code.zmax),500)
zs = np.linspace(param.code.zmin,param.code.zmax,500)
WGi = WeightFunc.W_GalaxyClustering
Wgi = WeightFunc.W_WeakLensing
WIAi = WeightFunc.W_IntrinsicAlignment
WCSi = WeightFunc.W_CosmicShear

t0, t1 = time(), time(); 
WGi_dict = {i: WGi(zs,i,verbose=0) for i,zi in enumerate(param.telescope.z_edges[:-1])}
print('Galaxy Clustering Weight Function for the {} bins estimated in {:.1f} s.'.format(len(param.telescope.z_edges[:-1]),time()-t0)); t0 = time() 
Wgi_dict = {i: Wgi(zs,i,verbose=0) for i,zi in enumerate(param.telescope.z_edges[:-1])}
print('Weak Lensing Weight Function for the {} bins estimated in {:.1f} s.'.format(len(param.telescope.z_edges[:-1]),time()-t0)); t0 = time() 
WIAi_dict = {i: WIAi(zs,i,verbose=0) for i,zi in enumerate(param.telescope.z_edges[:-1])}
print('Intrinsic Alignment Weight Function for the {} bins estimated in {:.1f} s.'.format(len(param.telescope.z_edges[:-1]),time()-t0)); t0 = time() 
WCSi_dict = {i: WCSi(zs,i,verbose=0) for i,zi in enumerate(param.telescope.z_edges[:-1])}
print('Cosmic Shear Weight Function for the {} bins estimated in {:.1f} s.'.format(len(param.telescope.z_edges[:-1]),time()-t0)); t0 = time() 
print('Total runtime: {:.1f} s.'.format(time()-t1))

fig, axs = plt.subplots(2,2,figsize=(9,7)) #plt.subplots(2,2,figsize=(12,9))
ax = axs[0,0]; ax.set_title('Galaxy Clustering Weight Function')
for i,zi in enumerate(param.telescope.z_edges[:-1]):
    ax.plot(zs, WGi_dict[i], c='C{}'.format(i), label='$i={}$'.format(i))
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$W^g_i(z)$', fontsize=15)
ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
ax.legend()
ax = axs[0,1]; ax.set_title('Weak Lensing Weight Function')
for i,zi in enumerate(param.telescope.z_edges[:-1]):
    ax.plot(zs, Wgi_dict[i], c='C{}'.format(i), label='$i={}$'.format(i))
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$W^g_i(z)$', fontsize=15)
ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
# ax.legend()
ax = axs[1,0]; ax.set_title('IA Weight Function')
for i,zi in enumerate(param.telescope.z_edges[:-1]):
    ax.plot(zs, WIAi_dict[i], c='C{}'.format(i), label='$i={}$'.format(i))
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$W^{IA}_i(z)$', fontsize=15)
ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
# ax.legend()
ax = axs[1,1]; ax.set_title('Cosmic Shear Weight Function')
for i,zi in enumerate(param.telescope.z_edges[:-1]):
    ax.plot(zs, WCSi_dict[i], c='C{}'.format(i), label='$i={}$'.format(i))
    ax.plot(zs, WCSi_dict[i]-WIAi_dict[i], c='C{}'.format(i), label='$i={}$'.format(i), ls='-.')
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$W^{L}_i(z)$', fontsize=15)
ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
# ax.legend()
plt.tight_layout()
plt.show()