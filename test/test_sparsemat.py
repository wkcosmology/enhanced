from enhanced.math.sparse_mat import index_mats_or
from enhanced.math.sparse_mat import counter2d_in_sparse_mat
from enhanced.math.sparse_mat import index_mats_and
from enhanced.math.numpy import structured_array
import numpy as np
from scipy.sparse import coo_matrix

def test_index_mats_or():
    mat1 = coo_matrix(([1, 1, 1], ([0, 1, 2], [0, 1, 2])))
    mat2 = coo_matrix(([1, 1, 1], ([0, 0, 2], [0, 2, 2])))
    res_mat = index_mats_or([mat1, mat2])
    index_type = [("row", "<i8"), ("col", "<i8")]
    row = np.array([0, 1, 2, 0])
    col = np.array([0, 1, 2, 2])
    index_true = structured_array([row, col], index_type)
    index_return = structured_array([res_mat.row, res_mat.col], index_type)
    assert np.all(np.in1d(index_true, index_return))
    assert np.all(np.in1d(index_return, index_true))


def test_index_mats_and():
    mat1 = coo_matrix(([1, 1, 1], ([0, 1, 2], [0, 1, 2])))
    mat2 = coo_matrix(([1, 1, 1], ([0, 0, 2], [0, 2, 2])))
    res_mat = index_mats_and([mat1, mat2])
    index_type = [("row", "<i8"), ("col", "<i8")]
    row = np.array([0, 2])
    col = np.array([0, 2])
    index_true = structured_array([row, col], index_type)
    index_return = structured_array([res_mat.row, res_mat.col], index_type)
    assert np.all(np.in1d(index_true, index_return))
    assert np.all(np.in1d(index_return, index_true))


def test_counter2d():
    coords = [[10, 11], [13, 14], [10, 11], [10, 11], [13, 14], [20, 1]]
    count_mat = counter2d_in_sparse_mat(np.array(coords)).toarray()
    print(count_mat)
    assert count_mat.shape == (21, 15)
    assert count_mat[10, 11] == 3
    assert count_mat[13, 14] == 2
    assert count_mat[20, 1] == 1
