"""Plotting utilities.

TODO: more info...

----------

"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def approx_kde(data, bins=500, bw=0.075):
    """A fast approximation to kernel density estimation."""
    stds = 3  # use a gaussian kernel w/ this many std devs
    counts, be = np.histogram(data, bins=bins)
    db = be[1] - be[0]
    pad = 0.5 * bins * bw * stds * db
    pbe = np.arange(db, pad, db)
    x_out = np.concatenate(
        (be[0] - np.flip(pbe), be[0:-1] + np.diff(be), be[-1] + pbe)
    )
    z_pad = np.zeros(pbe.shape[0])
    raw = np.concatenate((z_pad, counts, z_pad))
    k_x = np.linspace(-stds, stds, int(bins * bw * stds))
    kernel = 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-np.square(k_x) / 2.0)
    y_out = np.convolve(raw, kernel, mode="same")
    return x_out, y_out


def get_next_color(def_color, ix):
    """Get the next color in the color cycle"""
    if def_color is None:
        return COLORS[ix % len(COLORS)]
    elif isinstance(def_color, list):
        return def_color[ix % len(def_color)]
    else:
        return def_color


def get_ix_label(ix, shape):
    """Get a string representation of the current index"""
    dims = np.zeros(len(shape))
    for d in range(len(shape) - 1, 0, -1):
        prod = np.prod(shape[:d])
        dims[d] = np.floor(ix / prod)
        ix -= dims[d] * prod
    dims[0] = ix
    if len(shape) == 1:
        return str(dims[0].astype("int32"))
    else:
        return str(list(dims.astype("int32")))


def plot_dist(
    data,
    xlabel="",
    style="fill",
    bins=20,
    ci=0.0,
    bw=0.075,
    alpha=0.4,
    color=None,
    legend=True,
):
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
    legend : bool
        Whether to show legends for plots with >1 distribution
        Default = True
    """

    # Check inputs
    if ci < 0.0 or ci > 1.0:
        raise ValueError("ci must be between 0 and 1")

    # If 1d make 2d
    if data.ndim == 1:
        data = np.expand_dims(data, 1)

    # Number of datasets
    dims = data.shape[1:]
    Nd = np.prod(dims)

    # Flatten if >1D
    data = np.reshape(data, (data.shape[0], Nd), order="F")

    # Compute confidence intervals
    if ci:
        cis = np.empty((Nd, 2))
        ci0 = 100 * (0.5 - ci / 2.0)
        ci1 = 100 * (0.5 + ci / 2.0)
        for i in range(Nd):
            cis[i, :] = np.percentile(data[:, i], [ci0, ci1])

    # Plot the data
    for i in range(Nd):
        next_color = get_next_color(color, i)
        lab = get_ix_label(i, dims)
        if style == "line":
            px, py = approx_kde(data[:, i], bw=bw)
            plt.plot(px, py, color=next_color, label=lab)
            if ci:
                yci = np.interp(cis[i, :], px, py)
                plt.plot(
                    [cis[i, 0], cis[i, 0]], [0, yci[0]], ":", color=next_color
                )
                plt.plot(
                    [cis[i, 1], cis[i, 1]], [0, yci[1]], ":", color=next_color
                )
        elif style == "fill":
            px, py = approx_kde(data[:, i], bw=bw)
            plt.fill(px, py, facecolor=next_color, alpha=alpha, label=lab)
            if ci:
                k = (px > cis[i, 0]) & (px < cis[i, 1])
                kx = px[k]
                ky = py[k]
                plt.fill(
                    np.concatenate(([kx[0]], kx, [kx[-1]])),
                    np.concatenate(([0], ky, [0])),
                    facecolor=next_color,
                    alpha=alpha,
                )
        elif style == "hist":
            _, be, patches = plt.hist(
                data[:, i], alpha=alpha, bins=bins, color=next_color, label=lab
            )
            if ci:
                k = (data[:, i] > cis[i, 0]) & (data[:, i] < cis[i, 1])
                plt.hist(data[k, i], alpha=alpha, bins=be, color=next_color)
        else:
            raise ValueError("style must be 'fill', 'line', or 'hist'")

    # Only show the legend if there are >1 sample set
    if Nd > 1 and legend:
        plt.legend()

    # Set x axis label, and no y axis or bounding box needed
    plt.xlabel(xlabel)
    plt.gca().get_yaxis().set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)


def plot_line(xdata, ydata, xlabel="", ylabel="", fmt="-", color=None):
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
        raise ValueError("x and y data do not have same length")

    # Number of datasets
    dims = ydata.shape[1:]
    Nd = np.prod(dims)

    # Flatten if >1D
    ydata = np.reshape(ydata, (ydata.shape[0], Nd), order="F")

    # Plot the data
    for i in range(Nd):
        next_color = get_next_color(color, i)
        lab = get_ix_label(i, dims)
        plt.plot(xdata, ydata[:, i], fmt, color=next_color, label=lab)

    # Only show the legend if there are >1 sample set
    if Nd > 1:
        plt.legend()

    # Set axis labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def fill_between(xdata, lb, ub, xlabel="", ylabel="", alpha=0.3, color=None):
    """Fill between lines.

    Parameters
    ----------
    xdata : |ndarray|
        X values of points to plot.  Should be vector of length ``Nsamples``.
    lb : |ndarray|
        Lower bound of fill.  Should be of size ``(Nsamples,...)``.
    ub : |ndarray|
        Upper bound of fill.  Should be same size as lb.
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
        raise ValueError("lb and ub must have same shape")
    if len(xdata) != lb.shape[0]:
        raise ValueError("xdata does not match shape of lb and ub")

    # If 1d make 2d
    if lb.ndim == 1:
        lb = np.expand_dims(lb, 1)
        ub = np.expand_dims(ub, 1)

    # Number of fills and datasets
    dims = lb.shape[1:]
    Nd = int(np.prod(dims))

    # Flatten if >1D
    lb = np.reshape(lb, (lb.shape[0], Nd), order="F")
    ub = np.reshape(ub, (ub.shape[0], Nd), order="F")

    # Plot the data
    for iD in range(Nd):  # for each dataset,
        next_color = get_next_color(color, iD)
        lab = get_ix_label(iD, dims)
        plt.fill_between(
            xdata,
            lb[:, iD],
            ub[:, iD],
            alpha=alpha,
            facecolor=next_color,
            label=lab,
        )

    # Only show the legend if there are >1 datasets
    if Nd > 1:
        plt.legend()

    # Set x axis label, and no y axis or bounding box needed
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def centered_text(text):
    """Display text centered in the figure"""
    plt.gca().text(
        0.5,
        0.5,
        text,
        horizontalalignment="center",
        verticalalignment="center",
        transform=plt.gca().transAxes,
    )


def plot_discrete_dist(x):
    """Plot histogram of discrete variable"""
    minx = np.min(x)
    maxx = np.max(x)
    be = np.linspace(minx - 0.5, maxx + 0.5, int(maxx - minx + 2))
    bc = np.linspace(minx, maxx, int(maxx - minx + 1))
    xc, _ = np.histogram(x, be)
    xc = xc / xc.sum()  # normalize
    plt.bar(bc, xc)


def plot_categorical_dist(x):
    """Plot histogram of categorical variable"""
    xc = pd.Series(x.ravel()).value_counts().sort_index()
    xc = xc / xc.sum()  # normalize
    plt.bar(xc.index, xc.values)
    if len(xc.index) < 15:
        plt.xticks(xc.index, [str(e) for e in xc.index])
    else:
        step = int(len(xc.index) / 7)
        plt.xticks(xc.index[::step], [str(e) for e in xc.index[::step]])


def plot_by(
    x, data, bins=30, func="mean", plot=True, bootstrap=100, ci=0.95, **kwargs
):
    """Compute and plot some function func of data as a function of x.

    Parameters
    ----------
    x : |ndarray|
        Coordinates of data to plot
    data : |ndarray|
        Data to plot by bins of x
    bins : int
        Number of bins to bin x into
    func : callable or str
        Function to apply on elements of data in each x bin.  Can be a
        callable or one of the following str:

        * ``'count'``
        * ``'sum'``
        * ``'mean'``
        * ``'median'``

        Default = ``'mean'``

    plot : bool
        Whether to plot ``data`` as a function of ``x``
        Default = False
    bootstrap : None or int > 0
        Number of bootstrap samples to use for estimating the uncertainty of
        the true coverage.
    ci : list of float between 0 and 1
        Bootstrapped confidence interval percentiles of coverage to show.
    **kwargs
        Additional arguments are passed to plt.plot or fill_between

    Returns
    -------
    x_o : |ndarray|
        ``x`` bin centers
    data_o : |ndarray|
        ``func`` applied to ``data`` values in each ``x`` bin
    """

    # Check types
    if not isinstance(bins, int):
        raise TypeError("bins must be an int")
    if bins < 1:
        raise ValueError("bins must be positive")
    if not isinstance(plot, bool):
        raise TypeError("plot must be True or False")
    if bootstrap is not None and not isinstance(bootstrap, int):
        raise TypeError("bootstrap must be None or an int")
    if isinstance(bootstrap, int) and bootstrap < 1:
        raise ValueError("bootstrap must be > 0")
    if ci < 0.0 or ci > 1.0:
        raise ValueError("ci must be between 0 and 1")

    # Determine what function to use
    if callable(func):
        pass
    elif isinstance(func, str):
        if func == "mean":
            func = np.mean
        elif func == "median":
            func = np.median
        elif func == "count":
            func = len
        else:
            raise ValueError("Unknown function name " + func)
    else:
        raise TypeError("func must be a callable or a function name str")

    # Default color
    if "color" in kwargs:
        color = kwargs["color"]
    else:
        color = COLORS[0]

    # Ensure x is at least 2d
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    # 1 Dimensional
    if x.shape[1] == 1:

        # Create bins over x
        edges = np.linspace(min(x), max(x), int(bins)).flatten()
        edges[-1] += 1e-9
        bin_id = np.digitize(x, edges)
        x_o = (edges[:-1] + edges[1:]) / 2.0  # bin centers

        # Bootstrap estimate coverage uncertainty
        if bootstrap is not None:

            # Compute func for data in each bins
            boots = pd.DataFrame(index=range(1, bins))
            for iB in range(bootstrap):
                ix = np.random.choice(range(data.size), size=data.size)
                boots[str(iB)] = (
                    pd.Series(data[ix].flatten())
                    .groupby(bin_id[ix].flatten())
                    .agg(func)
                )

            # Plot coverage confidence intervals
            ci = np.array(ci)
            ci_lb = 100 * (0.5 - ci / 2.0)
            ci_ub = 100 * (0.5 + ci / 2.0)
            boots = boots.values
            prc_lb = np.nanpercentile(boots, ci_lb, axis=1)
            prc_ub = np.nanpercentile(boots, ci_ub, axis=1)
            plt.fill_between(x_o, prc_lb, prc_ub, alpha=0.3, facecolor=color)

        # Compute func for data in each bins
        data_o = pd.DataFrame(index=range(1, bins))
        data_o["data"] = (
            pd.Series(data.flatten()).groupby(bin_id.flatten()).agg(func)
        )
        data_o = data_o["data"]

        # Plot coverage
        plt.plot(x_o, data_o, **kwargs)

        # Return values
        return x_o, data_o.values

    # 2 Dimensional
    elif x.shape[1] == 2:

        pass
        # TODO

    else:
        raise ValueError("x.shape[1] cannot be >2")
