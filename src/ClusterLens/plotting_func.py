import numpy as np 
import matplotlib.pyplot as plt 


def solve_quadratic(a, b, c):
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        raise ValueError("No real solutions exist for the given value of a")
    n1 = (-b + np.sqrt(discriminant)) / (2*a)
    n2 = (-b - np.sqrt(discriminant)) / (2*a)
    return n1, n2


def plot_Cls_2D(ells, Cls, **kwargs):
    fig = kwargs.get('fig', None)
    n_tomo_bins = int(max(solve_quadratic(1,1,-2*Cls.shape[1])))
    print('Number of tomographic bins in the data: {}'.format(n_tomo_bins))
    figsize = kwargs.get('figsize', (10,9))
    color = kwargs.get('color', kwargs.get('c', 'k'))
    ls = kwargs.get('ls', '-')
    lw = kwargs.get('lw', 2.0)
    legend = kwargs.get('legend', None)

    if fig is None: fig, axs = plt.subplots(n_tomo_bins,n_tomo_bins,figsize=figsize)
    else: axs = np.array(fig.get_axes()).reshape(n_tomo_bins,n_tomo_bins)
    count = 0
    for i in range(n_tomo_bins):
        for j in range(n_tomo_bins):
            ax = axs[i,j]
            if i<j:
                ax.set_axis_off()
            else:
                ax.plot(ells, Cls[:,count]*ells*(ells+1), color=color, ls=ls, lw=lw)
                ax.set_xscale(kwargs.get('xscale','log'))
                ax.set_yscale(kwargs.get('yscale','log'))
                count += 1
    return fig