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
CCL = ClusterLens.Cosmology(param) #ClusterLens.InterfaceCCL(param)
pk  = CCL.power_spectrum(k=ks, z=zs)

# Plot
fig, ax = plt.subplots(1,1,figsize=(7,6))
ax.plot(zs, CCL.D(zs))
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$D(z)$', fontsize=15)
ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
plt.tight_layout()
plt.show()

# Plot
zplot = [0,1,2]
fig, ax = plt.subplots(1,1,figsize=(7,6))
for i,zi in enumerate(zplot):
    ax.plot(pk['k'], pk['pk_lin'][np.abs(pk['z']-zi).argmin(),:], label='$z={}$'.format(zi))
    ax.plot(pk['k'], pk['pk_nonlin'][np.abs(pk['z']-zi).argmin(),:], ls='--', c='k')
ax.loglog(pk['k'][:1], pk['pk_lin'][0,:1], label='linear', ls='--', c='k')
ax.loglog(pk['k'][:1], pk['pk_nonlin'][0,:1], label='non linear', ls='--', c='k')
ax.set_xlabel('$k$ [1/Mpc]', fontsize=15)
ax.set_ylabel('$P_k$', fontsize=15)
ax.legend()
plt.tight_layout()
plt.show()

ells = np.logspace(np.log10(param.code.lmin),np.log10(param.code.lmax),param.code.Nl)
k_ls = CCL.ell_to_kl(ells, z=zs)
print(ks.shape, zs.shape, ells.shape, k_ls.shape)
print('min-max k : {:.2f}, {:.2f}'.format(ks.min(), ks.max()))   
print('min-max kl: {:.2f}, {:.2f}'.format(k_ls.min(), k_ls.max()))
print('min-max log10k : {:.2f}, {:.2f}'.format(np.log10(ks.min()), np.log10(ks.max())))
print('min-max log10kl: {:.2f}, {:.2f}'.format(np.log10(k_ls.min()), np.log10(k_ls.max())))

# fig, ax = plt.subplots(1,1,figsize=(7,6))
# im = ax.pcolor(zs, ells, np.log10(k_ls), 
#                shading='auto')
# ax.set_xlabel('$z$', fontsize=15)
# ax.set_ylabel('$l$', fontsize=15)
# ax.set_title('log$_{10}k_l(z)$', fontsize=15)
# cbar = fig.colorbar(im)
# contour_level = [np.log10(ks.min()), np.log10(ks.max())]
# ax.contour(zs, ells, np.log10(k_ls), levels=contour_level, colors='r')
# plt.tight_layout()
# plt.show()

Pkl_interp = CCL.create_Pkl_interpolator()

# Plot
lplot = [17.59,32.96,61.75,115.67,216.70,405.95,760.5,1424.7,2668.99,5000]
fig, ax = plt.subplots(1,1,figsize=(7,6))
for i,ell in enumerate(lplot):
    ax.loglog(zs, Pkl_interp(zs,ell), label='$l={}$'.format(ell))
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$P(z,k_l)$', fontsize=15)
ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
ax.legend()
plt.tight_layout()
plt.show()
