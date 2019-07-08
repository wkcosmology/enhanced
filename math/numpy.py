#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File            : numpy.py
# Author          : Kai Wang <wkcosmology@gmail.com>
# Date            : 06.12.2019
# Last Modified By: Kai Wang <wkcosmology@gmail.com>

import numpy as np


def structured_array(data, dtype):
    """constructed a structed array from ndarray given dtype

    Parameters
    ----------
    data : array_like
        list of data to be columns of structured array
    dtype : array_like
        dtype for each column

    Returns
    -------
    :class:numpy.array
        numpy structured array with user defined dtype

    """
    assert(len(data) == len(dtype))
    length = len(data[0])
    arr = np.zeros(length, dtype=dtype)
    for i, t in enumerate(dtype):
        arr[t[0]] = data[i]
    return arr
