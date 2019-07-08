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
    results = [np.nanpercentile(ys[bin_ids == i], percens) for i in range(len(x_vals))]
    results = np.array([v if not np.any(np.isnan(v)) else [np.nan] * len(percens) for v in results]).T
    return np.row_stack((x_vals, results))


class EmpiricalInterp(object):

    """Interpolation empirically from given data"""

    def __init__(self, data_xs, data_ys, data_xs_edges=None):
        """initialize the object

        Parameters
        ----------
        data_xs : ndarray
            xs of data to calibrate the relation
        data_ys : ndarray
            ys of data to calibrate the relation
        data_xs_edges : ndarray, optional
            the edges of the xs

        """
        self._data_xs = data_xs
        self._data_ys = data_ys
        self._data_edges = data_xs_edges

    @cached_property
    def _mean(self):
        """return the mean of y values in each bin """
        return binned_stat(self._data_xs, self._data_ys, self._data_edges, ret_std=False)

    @cached_property
    def _medium(self):
        """return the medium of y values in each bin """
        return binned_percentile(self._data_xs, self._data_ys, self._data_edges, percens=[50])

    def interp(self, xs, kind="mean", left=None, right=None):
        """interp for the new xs

        Parameters
        ----------
        xs : ndarray
            the xs to inpterpolate
        kind : {"mean", "medium"}
            representation of value in each bin
        left : pyobject, optional
            value for xs smaller than the minimal one in calibration
            default the left-most one
        right : pyobject, optional
            value for xs greater than the maximum one in calibration
            default the right-most on

        Returns
        -------
        ndarray
            the interpolated values for xs, np.nan for xs out of range

        """
        if kind == "mean":
            self._x_p, self._y_p = self._mean
        if kind == "medium":
            self._x_p, self._y_p = self._medium
        mask = ~np.isnan(self._y_p)
        self._x_p = self._x_p[mask]
        self._y_p = self._y_p[mask]
        if left is None:
            left = self._y_p[0]
        if right is None:
            right = self._y_p[-1]
        return np.interp(xs, self._x_p, self._y_p, left=left, right=right)


class EmpiricalInterp2d(object):

    """Docstring for EmpiricalInterp2d. """

    def __init__(self, data_xs, data_ys, data_vals, x_edges, y_edges):
        """TODO: to be defined1.

        Parameters
        ----------
        data_xs : TODO
        data_ys : TODO
        data_vals : TODO
        x_edges : TODO
        y_edges : TODO


        """
        self._data_xs = data_xs
        self._data_ys = data_ys
        self._data_vals = data_vals
        self._x_edges = x_edges
        self._y_edges = y_edges
        # clean the data
        mask = (
            (self._data_xs > self._x_edges[0]) &
            (self._data_xs < self._x_edges[-1]) &
            (self._data_ys > self._y_edges[0]) &
            (self._data_ys < self._y_edges[-1]))
        self._data_xs = self._data_xs[mask]
        self._data_ys = self._data_ys[mask]
        self._data_vals = self._data_vals[mask]

    def _get_value(self, func):
        vals = np.zeros((len(self._x_edges) - 1, len(self._y_edges) - 1))
        counts = np.zeros((len(self._x_edges) - 1, len(self._y_edges) - 1))
        bin_id_x = np.digitize(self._data_xs, self._x_edges) - 1
        bin_id_y = np.digitize(self._data_ys, self._y_edges) - 1
        for i in range(len(self._x_edges) - 1):
            for j in range(len(self._y_edges) - 1):
                val_arr = self._data_vals[(bin_id_x == i) & (bin_id_y == j)]
                if len(val_arr) == 0:
                    vals[i, j] = np.nan
                    counts[i, j] = 0
                else:
                    vals[i, j] = func(val_arr)
                    counts[i, j] = np.count_nonzero(~np.isnan(val_arr))
        return vals, counts

    @cached_property
    def _mean(self):
        return self._get_value(np.mean)

    @cached_property
    def _medium(self):
        return self._get_value(np.median)

    def interp(self, xs, ys, kind="mean", min_count=1):
        """TODO: Docstring for interp.

        Parameters
        ----------
        xs : TODO
        ys : TODO
        kind : TODO
        min_counts : TODO

        Returns
        -------
        TODO

        """
        assert len(xs) == len(ys)
        bin_id_x = np.digitize(xs, self._x_edges) - 1
        bin_id_y = np.digitize(ys, self._y_edges) - 1

        if kind == "mean":
            self._vals, self._counts = self._mean
        elif kind == "medium":
            self._vals, self._counts = self._medium
        else:
            raise Exception("kind parameter should be mean or medium")

        vals = np.ones(len(xs)) * np.nan
        for i in range(len(self._x_edges) - 1):
            for j in range(len(self._y_edges) - 1):
                mask = (bin_id_x == i) & (bin_id_y == j)
                vals[mask] = np.where(
                    self._counts[i, j] >= min_count,
                    self._vals[i, j],
                    np.nan)
        return vals
