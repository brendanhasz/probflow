
import numpy as np
import matplotlib.pyplot as plt
from probflow import *

from probflow.plotting import plot_dist, get_ix_label

PLOT = False


def test_get_ix_label():
    """Tests plotting.get_ix_label"""
    assert get_ix_label(0, [3, 5, 2]) == '[0, 0, 0]'
    assert get_ix_label(1, [3, 5, 2]) == '[1, 0, 0]'
    assert get_ix_label(3, [3, 5, 2]) == '[0, 1, 0]'
    assert get_ix_label(4, [3, 5, 2]) == '[1, 1, 0]'
    assert get_ix_label(15, [3, 5, 2]) == '[0, 0, 1]'
    assert get_ix_label(16, [3, 5, 2]) == '[1, 0, 1]'
    assert get_ix_label(19, [3, 5, 2]) == '[1, 1, 1]'
    assert get_ix_label(29, [3, 5, 2]) == '[2, 4, 1]'


def test_plot_dist():
    """Tests plotting.plot_dist"""
    data = np.random.randn(100)
    plot_dist(data, xlabel='the x label')
    if PLOT:
        plt.show()

if __name__ == "__main__":
    PLOT = True
    test_get_ix_label()
    test_plot_dist()
