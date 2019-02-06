
import numpy as np
import matplotlib.pyplot as plt
from probflow import *

from probflow.utils.plotting import plot_dist
from probflow.utils.plotting import get_ix_label
from probflow.utils.plotting import fill_between


def test_get_ix_label():
    """Tests utils.plotting.get_ix_label"""
    assert get_ix_label(0, [3, 5, 2]) == '[0, 0, 0]'
    assert get_ix_label(1, [3, 5, 2]) == '[1, 0, 0]'
    assert get_ix_label(3, [3, 5, 2]) == '[0, 1, 0]'
    assert get_ix_label(4, [3, 5, 2]) == '[1, 1, 0]'
    assert get_ix_label(15, [3, 5, 2]) == '[0, 0, 1]'
    assert get_ix_label(16, [3, 5, 2]) == '[1, 0, 1]'
    assert get_ix_label(19, [3, 5, 2]) == '[1, 1, 1]'
    assert get_ix_label(29, [3, 5, 2]) == '[2, 4, 1]'


def test_plot_dist(plot):
    """Tests utils.plotting.plot_dist"""
    data = np.random.randn(100)
    plot_dist(data, xlabel='the x label')
    if plot:
        plt.show()


def test_fill_between_scalar(plot):
    """Tests utils.plotting.fill_between"""

    xdata = np.linspace(0, 1, 10)
    lb = np.empty([3, 10, 1])
    ub = np.empty([3, 10, 1])
    lb[0,:,0] = np.linspace(-5, 0, 10)
    lb[1,:,0] = np.linspace(-3, 0, 10)
    lb[2,:,0] = np.linspace(-1, 0, 10)
    ub[0,:,0] = np.linspace(10, 0, 10)
    ub[1,:,0] = np.linspace(5, 0, 10)
    ub[2,:,0] = np.linspace(1, 0, 10)

    fill_between(xdata, lb, ub)
    if plot:
        plt.show()


def test_fill_between_vector(plot):
    """Tests utils.plotting.fill_between w/ multiple datasets"""

    xdata = np.linspace(0, 1, 10)
    lb = np.empty([2, 10, 3])
    ub = np.empty([2, 10, 3])
    lb[0,:,0] = np.linspace(-1, 0, 10)
    lb[1,:,0] = np.linspace(-0.5, 0, 10)
    ub[0,:,0] = np.linspace(2, 0, 10)
    ub[1,:,0] = np.linspace(1, 0, 10)

    lb[0,:,1] = np.linspace(0, -1, 10) + 3
    lb[1,:,1] = np.linspace(0, -0.5, 10) + 3
    ub[0,:,1] = np.linspace(0, 2, 10) + 3
    ub[1,:,1] = np.linspace(0, 1, 10) + 3

    lb[0,:,2] = np.linspace(-1, 0, 10) + 5
    lb[1,:,2] = np.linspace(-0.5, 0, 10) + 5
    ub[0,:,2] = np.linspace(2, 0, 10) + 5
    ub[1,:,2] = np.linspace(1, 0, 10) + 5

    fill_between(xdata, lb, ub)
    if plot:
        plt.show()


def test_fill_between_matrix(plot):
    """Tests utils.plotting.fill_between w/ 2d datasets"""

    xdata = np.linspace(0, 1, 10)
    lb = np.empty([2, 10, 3, 2])
    ub = np.empty([2, 10, 3, 2])
    lb[0,:,0, 0] = np.linspace(-1, 0, 10)
    lb[1,:,0, 0] = np.linspace(-0.5, 0, 10)
    ub[0,:,0, 0] = np.linspace(2, 0, 10)
    ub[1,:,0, 0] = np.linspace(1, 0, 10)

    lb[0,:,1, 0] = np.linspace(0, -1, 10) + 3
    lb[1,:,1, 0] = np.linspace(0, -0.5, 10) + 3
    ub[0,:,1, 0] = np.linspace(0, 2, 10) + 3
    ub[1,:,1, 0] = np.linspace(0, 1, 10) + 3

    lb[0,:,2, 0] = np.linspace(-1, 0, 10) + 5
    lb[1,:,2, 0] = np.linspace(-0.5, 0, 10) + 5
    ub[0,:,2, 0] = np.linspace(2, 0, 10) + 5
    ub[1,:,2, 0] = np.linspace(1, 0, 10) + 5

    lb[0,:,0, 1] = np.linspace(-1, 0, 10) + 10
    lb[1,:,0, 1] = np.linspace(-0.5, 0, 10) + 10
    ub[0,:,0, 1] = np.linspace(2, 0, 10) + 10
    ub[1,:,0, 1] = np.linspace(1, 0, 10) + 10

    lb[0,:,1, 1] = np.linspace(0, -1, 10) + 3 + 10
    lb[1,:,1, 1] = np.linspace(0, -0.5, 10) + 3 + 10
    ub[0,:,1, 1] = np.linspace(0, 2, 10) + 3 + 10
    ub[1,:,1, 1] = np.linspace(0, 1, 10) + 3 + 10

    lb[0,:,2, 1] = np.linspace(-1, 0, 10) + 5 + 10
    lb[1,:,2, 1] = np.linspace(-0.5, 0, 10) + 5 + 10
    ub[0,:,2, 1] = np.linspace(2, 0, 10) + 5 + 10
    ub[1,:,2, 1] = np.linspace(1, 0, 10) + 5 + 10
    fill_between(xdata, lb, ub)
    if plot:
        plt.show()
