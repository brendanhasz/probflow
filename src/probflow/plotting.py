"""Plotting utilities.

TODO: more info...

----------

"""


import numpy as np
import matplotlib.pyplot as plt

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
    Nd = np.prod(data.shape[1:])

    # Compute confidence intervals
    if ci:
        cis = np.empty((Nd, 2))
        ci0 = 100 * (0.5 - ci/2.0);
        ci1 = 100 * (0.5 + ci/2.0);
        for i in range(Nd):
            cis[i,:] = np.percentile(data[:,i], [ci0, ci1])

    # Plot the data
    for i in range(Nd):
        next_color = get_next_color(color, i)
        if style == 'line':
            px, py = approx_kde(data[:,i], bw=bw)
            p1 = plt.plot(px, py, color=next_color)
            if ci:
                yci = np.interp(cis[i,:], px, py)
                plt.plot([cis[i,0], cis[i,0]], [0, yci[0]], 
                         ':', color=next_color)
                plt.plot([cis[i,1], cis[i,1]], [0, yci[1]], 
                         ':', color=next_color)
        elif style == 'fill':
            px, py = approx_kde(data[:,i], bw=bw)
            p1 = plt.fill(px, py, facecolor=next_color, alpha=alpha)
            if ci:
                k = (px>cis[i,0]) & (px<cis[i,1])
                kx = px[k]
                ky = py[k]
                plt.fill(np.concatenate(([kx[0]], kx, [kx[-1]])),
                         np.concatenate(([0], ky, [0])),
                         facecolor=next_color, alpha=alpha)
        elif style == 'hist':
            _, be, patches = plt.hist(data[:,i], alpha=alpha, 
                                      bins=bins, color=next_color)
            if ci:
                k = (data[:,i]>cis[i,0]) & (data[:,i]<cis[i,1])
                plt.hist(data[k,i], alpha=alpha, bins=be, color=next_color)

    # TODO: may want to have an option to add legends w/ indexes
    # (for Parameters w/ shape>1 there will be multiple lines in the plots)

    # Set x axis label, and no y axis needed
    plt.xlabel(xlabel)
    plt.gca().get_yaxis().set_visible(False)


def centered_text(text):
    """Display text centered in the figure"""
    plt.gca().text(0.5, 0.5, text,
               horizontalalignment='center',
               verticalalignment='center',
               transform=plt.gca().transAxes)
