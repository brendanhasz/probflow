"""Plotting utilities.

TODO: more info...

----------

"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


def approx_kde(data, bins=500, bw=0.075):
    """A fast approximation to kernel density estimation."""
    stds = 3 #use a gaussian kernel w/ this many std devs
    counts, be = np.histogram(data, bins=bins)
    db = be[1]-be[0]
    pad = 0.5*bins*bw*stds*db
    pbe = np.arange(db, pad, db)
    x_out = np.concatenate((be[0]-np.flip(pbe),
                           be[0:-1] + np.diff(be),
                           be[-1]+pbe))
    z_pad = np.zeros(pbe.shape[0])
    raw = np.concatenate((z_pad, counts, z_pad))
    k_x = np.linspace(-stds, stds, bins*bw*stds)
    kernel = 1.0/np.sqrt(2.0*np.pi)*np.exp(-np.square(k_x)/2.0)
    y_out = np.convolve(raw, kernel, mode='same')
    return x_out, y_out


def get_next_color(def_color, ix):
    """Get the next color in the color cycle"""
    if def_color is None:
        return COLORS[ix%len(COLORS)]
    elif isinstance(def_color, list):
        return def_color[ix%len(def_color)]
    else:
        return def_color


def get_ix_label(ix, shape):
    """Get a string representation of the current index"""
    dims = np.zeros(len(shape))
    for d in range(len(shape)-1, 0, -1):
        prod = np.prod(shape[:d])
        dims[d] = np.floor(ix/prod)
        ix -= dims[d]*prod
    dims[0] = ix
    if len(shape) == 1:
        return str(dims[0].astype('int32'))
    else:
        return str(list(dims.astype('int32')))


def plot_dist(data, xlabel='', style='fill', bins=20, ci=0.0, bw=0.075, 
              alpha=0.4, color=None):
    """Plot the distribution of samples.

    Parameters
    ----------
    data : |ndarray|
        Samples to plot.  Should be of size (Nsamples,...)
    xlabel : str
        Label for the x axis
    style : str
        Which style of plot to create.  Available types are:

        * ``'fill'`` - filled density plot (the default)
        * ``'line'`` - line density plot
        * ``'hist'`` - histogram

    bins : int or list or |ndarray|
        Number of bins to use for the histogram (if 
        ``kde=False``), or a list or vector of bin edges.
    ci : float between 0 and 1
        Confidence interval to plot.  Default = 0.0 (i.e., not plotted)
    bw : float
        Bandwidth of the kernel density estimate (if using ``style='line'``
        or ``style='fill'``).  Default is 0.075
    alpha : float between 0 and 1
        Transparency of the plot (if ``style``=``'fill'`` or ``'hist'``)
    color : matplotlib color code or list of them
        Color(s) to use to plot the distribution.
        See https://matplotlib.org/tutorials/colors/colors.html
        Default = use the default matplotlib color cycle
    """

    # If 1d make 2d
    if data.ndim == 1:
        data = np.expand_dims(data, 1)

    # Number of datasets
    dims = data.shape[1:]
    Nd = np.prod(dims)

    # Flatten if >1D
    data = np.reshape(data, (data.shape[0], Nd), order='F')

    # Compute confidence intervals
    if ci:
        cis = np.empty((Nd, 2))
        ci0 = 100 * (0.5 - ci/2.0)
        ci1 = 100 * (0.5 + ci/2.0)
        for i in range(Nd):
            cis[i,:] = np.percentile(data[:,i], [ci0, ci1])

    # Plot the data
    for i in range(Nd):
        next_color = get_next_color(color, i)
        lab = get_ix_label(i, dims)
        if style == 'line':
            px, py = approx_kde(data[:,i], bw=bw)
            plt.plot(px, py, color=next_color, label=lab)
            if ci:
                yci = np.interp(cis[i,:], px, py)
                plt.plot([cis[i,0], cis[i,0]], [0, yci[0]], 
                         ':', color=next_color)
                plt.plot([cis[i,1], cis[i,1]], [0, yci[1]], 
                         ':', color=next_color)
        elif style == 'fill':
            px, py = approx_kde(data[:,i], bw=bw)
            plt.fill(px, py, facecolor=next_color, alpha=alpha, label=lab)
            if ci:
                k = (px>cis[i,0]) & (px<cis[i,1])
                kx = px[k]
                ky = py[k]
                plt.fill(np.concatenate(([kx[0]], kx, [kx[-1]])),
                         np.concatenate(([0], ky, [0])),
                         facecolor=next_color, alpha=alpha)
        elif style == 'hist':
            _, be, patches = plt.hist(data[:,i], alpha=alpha,
                                      bins=bins, color=next_color, label=lab)
            if ci:
                k = (data[:,i]>cis[i,0]) & (data[:,i]<cis[i,1])
                plt.hist(data[k,i], alpha=alpha, bins=be, color=next_color)

    # Only show the legend if there are >1 sample set
    if Nd > 1:
        plt.legend()

    # Set x axis label, and no y axis or bounding box needed
    plt.xlabel(xlabel)
    plt.gca().get_yaxis().set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)


def plot_line(xdata, ydata, xlabel='', ylabel='', fmt='-', color=None):
    """Plot lines.

    Parameters
    ----------
    xdata : |ndarray|
        X values of points to plot.  Should be vector of length ``Nsamples``.
    ydata : |ndarray|
        Y vaules of points to plot.  Should be of size ``(Nsamples,...)``.
    xlabel : str
        Label for the x axis. Default is no x axis label.
    ylabel : str
        Label for the y axis.  Default is no y axis label.
    fmt : str or matplotlib linespec
        Line marker to use.  Default = ``'-'`` (a normal line).
    color : matplotlib color code or list of them
        Color(s) to use to plot the distribution.
        See https://matplotlib.org/tutorials/colors/colors.html
        Default = use the default matplotlib color cycle
    """

    # If 1d make 2d
    if ydata.ndim == 1:
        ydata = np.expand_dims(ydata, 1)

    # Check x and y are the same size
    if xdata.shape[0] != ydata.shape[0]:
        raise ValueError('x and y data do not have same length')

    # Number of datasets
    dims = ydata.shape[1:]
    Nd = np.prod(dims)

    # Flatten if >1D
    ydata = np.reshape(ydata, (ydata.shape[0], Nd), order='F')

    # Plot the data
    for i in range(Nd):
        next_color = get_next_color(color, i)
        lab = get_ix_label(i, dims)
        plt.plot(xdata, ydata[:,i], fmt, color=next_color, label=lab)

    # Only show the legend if there are >1 sample set
    if Nd > 1:
        plt.legend()

    # Set axis labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def fill_between(xdata, lb, ub, xlabel='', ylabel='', alpha=0.3, color=None):
    """Fill between lines.

    Parameters
    ----------
    xdata : |ndarray|
        X values of points to plot.  Should be vector of length ``Nsamples``.
    ydata : |ndarray|
        Y vaules of points to plot.  Should be of size ``(Nsamples,...)``.
    xlabel : str
        Label for the x axis. Default is no x axis label.
    ylabel : str
        Label for the y axis.  Default is no y axis label.
    fmt : str or matplotlib linespec
        Line marker to use.  Default = ``'-'`` (a normal line).
    color : matplotlib color code or list of them
        Color(s) to use to plot the distribution.
        See https://matplotlib.org/tutorials/colors/colors.html
        Default = use the default matplotlib color cycle
    """

    # Check shapes
    if not np.all(lb.shape == ub.shape):
        raise ValueError('lb and ub must have same shape')
    if len(xdata) != lb.shape[1]:
        raise ValueError('xdata does not match shape of lb and ub')

    # If 1d make 2d
    if lb.ndim == 1:
        lb = np.expand_dims(lb, 1)
        ub = np.expand_dims(ub, 1)

    # Number of fills and datasets
    dims = lb.shape[2:]
    Nd = np.prod(dims)
    Np = lb.shape[0]

    # Flatten if >1D
    lb = np.reshape(lb, (lb.shape[0], lb.shape[1], Nd), order='F')
    ub = np.reshape(ub, (ub.shape[0], ub.shape[1], Nd), order='F')

    # Plot the data
    for iD in range(Nd): #for each dataset,
        next_color = get_next_color(color, iD)
        lab = get_ix_label(iD, dims)
        for iP in range(Np): #for each polygon,
            plt.fill_between(xdata, lb[iP,:,iD], ub[iP,:,iD],
                             alpha=alpha, facecolor=next_color, 
                             label=lab if iP==0 else None)

    # Only show the legend if there are >1 datasets
    if Nd > 1:
        plt.legend()

    # Set x axis label, and no y axis or bounding box needed
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def centered_text(text):
    """Display text centered in the figure"""
    plt.gca().text(0.5, 0.5, text,
               horizontalalignment='center',
               verticalalignment='center',
               transform=plt.gca().transAxes)
