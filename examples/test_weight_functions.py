import numpy as np 
from time import time
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm' #'dejavuserif' #'stix' #

import ClusterLens
# from ClusterLens import WeightFunctions

# Parameters 
param = ClusterLens.par()
# param.code.verbose = True

# Weight funtions
WeightFunc = ClusterLens.WeightFunctions(param) 

# zs = np.logspace(np.log10(param.code.zmin),np.log10(param.code.zmax),500)
zs = np.linspace(param.code.zmin,param.code.zmax,500)
WGi = WeightFunc.GalaxyClustering
Wgi = WeightFunc.WeakLensing
WIAi = WeightFunc.IntrinsicAlignment

fig, axs = plt.subplots(2,2,figsize=(9,7)) #plt.subplots(2,2,figsize=(12,9))
ax = axs[0,0]; ax.set_title('Galaxy Clustering Weight Function')
t0 = time()
for i,zi in enumerate(param.telescope.z_edges[:-1]):
    ax.plot(zs, WGi(zs,i,verbose=0), label='$i={}$'.format(i))
t1 = time(); print('Galaxy Clustering Weight Function for the {} bins estimated in {:.1f} s.'.format(i+1,t1-t0))
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$W^g_i(z)$', fontsize=15)
ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
ax.legend()
ax = axs[0,1]; ax.set_title('Weak Lensing Weight Function')
for i,zi in enumerate(param.telescope.z_edges[:-1]):
    t3 = time()
    ax.plot(zs, Wgi(zs,i,verbose=0), label='$i={}$'.format(i))
    t4 = time(); print('Weak Lensing Weight Function for bin-{} estimated in {:.1f} s.'.format(i,t4-t3))
t2 = time(); print('Weak Lensing Weight Function for the {} bins estimated in {:.1f} s.'.format(i+1,t2-t1))
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$W^g_i(z)$', fontsize=15)
ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
ax.legend()
ax = axs[1,0]; ax.set_title('Weak Lensing Weight Function')
for i,zi in enumerate(param.telescope.z_edges[:-1]):
    t3 = time()
    ax.plot(zs, WIAi(zs,i,verbose=0), label='$i={}$'.format(i))
    t4 = time(); print('IA Weight Function for bin-{} estimated in {:.1f} s.'.format(i,t4-t3))
t2 = time(); print('IAWeight Function for the {} bins estimated in {:.1f} s.'.format(i+1,t2-t1))
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$W^{IA}_i(z)$', fontsize=15)
ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
ax.legend()
plt.tight_layout()
plt.show()