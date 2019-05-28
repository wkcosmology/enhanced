#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File            : binned_stats.py
# Author          : Kai Wang <wkcosmology@gmail.com>
# Date            : 05.18.2019
# Last Modified By: Kai Wang <wkcosmology@gmail.com>
import numpy as np
from cached_property import cached_property


def binned_stat(xs, ys, x_edges=None, num_bins=10, ret_std=True):
    """calculate the mean and standard error of ys in the bin of xs

    Parameters
    ----------
    xs : ndarray
        the array as xs
    ys : ndarray
        the array as ys
    x_edges : ndarray, optional
        the edges
    num_bins : int, optional
        number of bins, useful when x_edges is None
    ret_std : boolean, optional
        if return the standard error
    Returns
    -------
    return an array of shape (3, len(x_edges) - 1)
    contains [x coordinate, mean, std]
    """
    if x_edges is None:
        x_edges = np.linspace(np.min(xs), np.max(xs), num_bins + 1)
    assert x_edges[0] < x_edges[1]
    mask = (xs >= x_edges[0]) & (xs < x_edges[-1])
    xs = xs[mask]
    ys = ys[mask]
    x_vals = (x_edges[1:] + x_edges[:-1]) / 2
    bin_ids = np.digitize(xs, x_edges) - 1
    mean = np.array([np.nanmean(ys[bin_ids == i]) for i in range(len(x_vals))])
    if ret_std:
        std = np.array([np.nanstd(ys[bin_ids == i]) for i in range(len(x_vals))])
        return np.array([x_vals, mean, std])
    else:
        return np.array([x_vals, mean])


def binned_percentile(xs, ys, x_edges=None, num_bins=10, percens=None):
    """calculate the percentile of ys in each x's bin

    Parameters
    ----------
    xs : ndarray
        the input xs
    ys : ndarray
        the input ys
    x_edges : ndarray, optional
        edges for x array
    num_bins : int, optional
        number of bins with min and max of xs as edges,
        ignored when x_edges is not NONE
    percens : ndarray, optional
        the percentils calculated,
        default is [25, 50, 75]

    Returns
    -------
    ndarray of percentiles in each x bins,
    shape (len(percens), len(x_edges) - 1)

    """
    if x_edges is None:
        x_edges = np.linspace(np.min(xs), np.max(xs), num_bins + 1)
    if percens is None:
        percens = [25, 50, 75]
    assert x_edges[0] < x_edges[1]
    mask = (xs >= x_edges[0]) & (xs < x_edges[-1])
    xs = xs[mask]
    ys = ys[mask]
    x_vals = (x_edges[1:] + x_edges[:-1]) / 2
    bin_ids = np.digitize(xs, x_edges) - 1
    results = [
        np.nanpercentile(ys[bin_ids == i], percens)
        for i in range(len(x_vals))]
    results = np.array([v if not np.isnan(v) else [np.nan] * len(percens) for v in results]).T
    return np.row_stack((x_vals, results))


class EmpiricalInterp(object):

    """Interpolation empirically from given data"""

    def __init__(self, data_xs, data_ys, data_edges=None):
        """initialize the object

        Parameters
        ----------
        data_xs : ndarray
            xs of data to calibrate the relation
        data_ys : ndarray
            ys of data to calibrate the relation
        data_edges : ndarray, optional
            the edges of the xs

        """
        self._data_xs = data_xs
        self._data_ys = data_ys
        self._data_edges = data_edges

    @cached_property
    def mean(self):
        """return the mean of y values in each bin """
        return binned_stat(self._data_xs, self._data_ys, self._data_edges, ret_std=False)

    @cached_property
    def medium(self):
        """return the medium of y values in each bin """
        return binned_percentile(self._data_xs, self._data_ys, self._data_edges, percens=[50])

    def interp(self, xs, kind="mean"):
        """interp for the new xs

        Parameters
        ----------
        xs : ndarray
            the xs to inpterpolate
        kind : {"mean", "medium"}
            representation of value in each bin

        Returns
        -------
        ndarray
            the interpolated values for xs, np.nan for xs out of range

        """
        if kind == "mean":
            self._x_p, self._y_p = self.mean
        if kind == "medium":
            self._x_p, self._y_p = self.medium
        return np.interp(xs, self._x_p, self._y_p, left=np.nan, right=np.nan)
