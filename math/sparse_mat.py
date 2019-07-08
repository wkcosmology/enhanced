#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File            : sparse_mat.py
# Author          : Kai Wang <wkcosmology@gmail.com>
# Date            : 06.04.2019
# Last Modified By: Kai Wang <wkcosmology@gmail.com>
import numpy as np
from collections import Counter
from scipy.sparse import coo_matrix
from enhanced.math.numpy import structured_array


def sparse_mat_max(mat, axis):
    """return the maximum value of a sparse matrix along an axis

    Parameters
    ----------
    mat : :class:scipy.sparse.spmatrix
    axis : 0 or 1

    Returns
    -------
    array_like
        the maximum values along an axis

    """
    max_vals = mat.max(axis=axis).toarray()
    return max_vals.reshape(max_vals.size)


def sparse_mat_argmax(mat, axis):
    """calculate the argmax for a sparse matrix

    Parameters
    ----------
    mat : :class:scipy.sparse.spmatrix
    axis : 0 or 1

    Returns
    -------
    array_like
        index of the maximum values along an axis

    """
    argmax_ids = np.array(mat.argmax(axis=axis)).astype(np.int32)
    return argmax_ids.reshape(argmax_ids.size)


def sparse_mat_sum(mat, axis):
    """calculate the sum for a sparse matrix

    Parameters
    ----------
    mat : :class:scipy.sparse.spmatrix
    axis : 0 or 1

    Returns
    -------
    array_like
        summation along an axis

    """
    sum_vals = np.array(mat.sum(axis=axis))
    return sum_vals.reshape(sum_vals.size)


def in2d_index(row_1, col_1, row_2, col_2):
    """return a boolean array of index 1 is in index 2

    Parameters
    ----------
    row_1 : array_like
    col_1 : array_like
    row_2 : array_like
    col_2 : array_like

    Returns
    -------
    array_like
        boolean array, True for index 1 in index 2, False for otherwise

    """
    index_type = [("row", "<i8"), ("col", "<i8")]
    index_tuple1 = structured_array([row_1, col_1], index_type)
    index_tuple2 = structured_array([row_2, col_2], index_type)
    return np.in1d(index_tuple1, index_tuple2)


def second_max(mat_in, axis):
    """return the second maximum value for the sparse matrix

    Parameters
    ----------
    mat_in : :class:scipy.sparse.coo_matrix
    axis : 0 or 1

    Returns
    -------
    array_like
        second maximum value along an axis

    """
    if not isinstance(mat_in, coo_matrix):
        raise Exception("Input matrix must be a scipy.coo_matrix object")
    mat = mat_in.copy()
    if axis == 1:
        row_index = np.arange(mat.shape[0])
        col_index = np.array(mat.argmax(axis=axis)).reshape(mat.shape[0])
    if axis == 0:
        col_index = np.arange(mat.shape[1])
        row_index = np.array(mat.argmax(axis=axis)).reshape(mat.shape[1])
    mask_maximum = in2d_index(mat.row, mat.col, row_index, col_index)
    mat.data[mask_maximum] = -np.inf
    return sparse_mat_max(mat, axis=axis)


def index_mats_or(mats_in):
    """return the sparse matrix with effective elements when any the input array
    is effective in that position

    Parameters
    ----------
    mats_in : array_like
        array_like with each element a sparse matrix

    Returns
    -------
    :class:scipy.sparse.coo_matrix
        sparse matrix with elements 0 or 1 where 1 for elements appear in the
        sparse matrix input
    """
    if len(mats_in) > 1:
        assert all([mat.shape == mats_in[0].shape for mat in mats_in[1:]])
    mats = [mat.copy().tocoo() for mat in mats_in]
    index_type = [("row", "<i8"), ("col", "<i8")]
    index_tuples = []
    for mat in mats:
        index_tuples.append(structured_array([mat.row, mat.col], index_type))
    index_tuples = np.unique(np.concatenate(index_tuples))
    res_mat = coo_matrix((np.ones(len(index_tuples)), (index_tuples["row"], index_tuples["col"])), shape=mats_in[0].shape)
    return res_mat


def index_mats_and(mats_in):
    """return the sparse matrix with effective elements when all the input array
    is effective in that position

    Parameters
    ----------
    mats_in : array_like
        array_like with each element a sparse matrix

    Returns
    -------
    :class:scipy.sparse.coo_matrix
        sparse matrix with elements 0 or 1 where 1 for elements appear in the
        sparse matrix input
    """
    if len(mats_in) > 1:
        assert all([mat.shape == mats_in[0].shape for mat in mats_in[1:]])
    mats = [mat.copy().tocoo() for mat in mats_in]
    index_type = [("row", "<i8"), ("col", "<i8")]
    index_tuples = []
    for mat in mats:
        index_tuples.append(structured_array([mat.row, mat.col], index_type))
    index_tuples = np.concatenate(index_tuples)
    index_tuples.flags.writeable = False
    counts = Counter(index_tuples)
    res_index = np.array([k for k, v in counts.items() if v == len(mats_in)])
    res_mat = coo_matrix((np.ones(len(res_index)), (res_index["row"], res_index["col"])), shape=mats_in[0].shape)
    return res_mat


def mask_sparse_matrix(mat, mask):
    """return a new sparse matrix with elements masked

    Parameters
    ----------
    mat : :class:scipy.sparse.spmatrix
    mask : array_like
        boolean array

    Returns
    -------
    :class:scipy.sparse.coo_matrix
        sparse matrix with masked elements

    """
    assert len(mat.data) == len(mask)
    mat = mat.copy().tocoo()
    res_mat = coo_matrix((mat.data[mask], (mat.row[mask], mat.row[mask])), shape=mat.shape)
    return res_mat
