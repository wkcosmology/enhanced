import numpy as np
from enhanced.stats.binned_stats import binned_stat

def test_binned_stats():
    data_xs = np.linspace(0, 1, 11)
    data_ys = np.ones(len(data_xs))
    xs, ys, err = binned_stat(data_xs, data_ys, np.linspace(0, 1, 10))
    assert np.all(ys == 1)

