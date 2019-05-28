import numpy as np
from halotools.mock_observables.pair_counters import pairwise_distance_xy_z
from enhanced.tool.io import sparse_matrix2fits
from enhanced.tool.io import fits2sparse_matrix
import pysnooper
import os

@pysnooper.snoop()
def test_sparsematrix2fits():
    pos_row = np.random.random((1000, 3))
    pos_col = np.random.random((1000, 3))
    rps = pairwise_distance_xy_z(pos_row, pos_col, 0.1, 0.05)
    sparse_matrix2fits(pos_row, pos_col, rps, "test.fits", ["first", "second"])
    out_row, out_col, mat_list = fits2sparse_matrix("test.fits")
    assert np.all(pos_row == out_row)
    assert np.all(pos_col == out_col)
    assert np.all(rps[0].row == mat_list[0].row)
    assert np.all(rps[0].col == mat_list[0].col)
    assert np.all(rps[0].data == mat_list[0].data)
    assert np.all(rps[1].row == mat_list[1].row)
    assert np.all(rps[1].col == mat_list[1].col)
    assert np.all(rps[1].data == mat_list[1].data)
    if os.path.exists("test.fits"):
        os.remove("test.fits")
