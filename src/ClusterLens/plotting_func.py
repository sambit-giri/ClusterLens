import numpy as np 
import matplotlib.pyplot as plt 


def solve_quadratic(a, b, c):
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        raise ValueError("No real solutions exist for the given value of a")
    n1 = (-b + np.sqrt(discriminant)) / (2*a)
    n2 = (-b - np.sqrt(discriminant)) / (2*a)
    return n1, n2


def plot_Cls_2D(ells, Cls, plot_type='full', **kwargs):
    fig = kwargs.get('fig', None)
    n_tomo_bins = int(max(solve_quadratic(1,1,-2*Cls.shape[1])))
    print('Number of tomographic bins in the data: {}'.format(n_tomo_bins))
    color = kwargs.get('color', kwargs.get('c', 'k'))
    ls = kwargs.get('ls', '-')
    lw = kwargs.get('lw', 2.0)
    legend = kwargs.get('legend', None)
    model_name = kwargs.get('model_name', None)
    if plot_type=='full':
        figsize = kwargs.get('figsize', (10,9))
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
        fig.text(0.005, 0.5, kwargs.get('ylabel', 'Y-Label'), va='center', rotation='vertical', fontsize=15)
        fig.text(0.5, 0.005, kwargs.get('xlabel', 'X-Label'), ha='center', fontsize=15)
        plt.subplots_adjust(wspace=0.15, hspace=0.15)
    elif plot_type=='ii':
        figsize = kwargs.get('figsize', (11,5))
        if fig is None: fig, axs = plt.subplots(1,2,figsize=figsize)
        else: axs = fig.get_axes()
        count, ax = 0, axs[0]
        for i in range(n_tomo_bins):
            for j in range(n_tomo_bins):
                if i==j:
                    label = kwargs.get('label', False)
                    ax.plot(ells, Cls[:,count]*ells*(ells+1), alpha=1-i*0.8/n_tomo_bins, 
                                    color=color, ls=ls, lw=lw, label='i={}'.format(i) if label else None)
                    ax.set_xscale(kwargs.get('xscale','log'))
                    ax.set_yscale(kwargs.get('yscale','log'))
                if i>=j: count += 1
        ax.set_ylabel(kwargs.get('ylabel', '$l(l+1)C_{ii}$'), fontsize=15)
        ax.set_xlabel(kwargs.get('xlabel', '$l$'), fontsize=15)
        ax.legend(fontsize=15, ncol=2)
        axs[1].plot([0,1], [0,1], color=color, ls=ls, lw=lw, label=model_name)
        axs[1].set_axis_off()
        axs[1].axis([2,3,2,3])
        axs[1].legend(loc='upper left', fontsize=16)
        plt.subplots_adjust(left=0.08, bottom=0.10, right=0.99, top=0.98, wspace=0.05, hspace=0.05)
    return fig