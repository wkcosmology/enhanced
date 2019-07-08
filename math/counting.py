#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File            : counting.py
# Author          : Kai Wang <wkcosmology@gmail.com>
# Date            : 06.11.2019
# Last Modified By: Kai Wang <wkcosmology@gmail.com>
from collections import Counter
import numpy as np

def counter(arr):
    """counting the number of apperance of each elements in an array

    Parameters
    ----------
    arr : array_like
        the input array
    Returns
    -------
    array_like
        the counts of apperance of each elements
    """
    counter_dict = Counter(arr)
    counts = np.array([counter_dict[i] for i in arr])
    return counts
