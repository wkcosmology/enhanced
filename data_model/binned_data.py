#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File            : binned_data.py
# Author          : Kai Wang <wkcosmology@gmail.com>
# Date            : 04.14.2020
# Last Modified By: Kai Wang <wkcosmology@gmail.com>

from itertools import product

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table


class BinnedData(object):

    """
    binned data in D-dimension specified by
    data : (i_1, i_2, i_3, ..., i_D) array
    edges_list :
        the length of elements should be (i_1 + 1, i_2 + 1, i_3 + 1, ..., i_D + 1)
    """

    def __init__(self, data, edges_list):
        """ construct the instance though direct data

        Parameters
        ----------
        data : ndarray
            shape (i_1, i_2, i_3, ..., i_D)
            fill the D dimension
        edges_list : list
            edges for bins,
            the length of elements should be (i_1 + 1, i_2 + 1, i_3 + 1, ..., i_D + 1)
        """
        for i in range(data.ndim):
            assert data.shape[i] == len(edges_list[i]) - 1
        self._data = data
        self._edges_list = edges_list
        self._dim = len(edges_list)

    @classmethod
    def from_file(cls, filename):
        """ construct the instance from a fits file

        Parameters
        ----------
        filename : string
            filename to a fits file

        Returns
        ----------
        BinnedData instance
        """
        binned_data = cls.__new__(cls)
        f = fits.open(filename)
        data = f[1].data
        edges_list = []
        for i in range(len(f) - 2):
            edges_tmp = f[i + 2].data["edges_%02d" % i]
            edges_list.append(edges_tmp)
            assert data.shape[i] == len(edges_tmp) - 1
        binned_data.__init__(data, edges_list)
        return binned_data

    @classmethod
    def from_points(cls, points, edges_list, weights=None):
        """ construct the instance though scatter points
        Parameters
        ----------
        points : ndarray
            coordinates of points, shape(N, D)
            means N points of D dimension
        edges_list : list
            edges for bins,
            the length of elements should be (i_1 + 1, i_2 + 1, i_3 + 1, ..., i_D + 1)
        weights : ndarray
            weights of points, shape(N)

        Returns
        ----------
        BinnedData2D instance
        """
        assert points.shape[1] == len(edges_list)
        binned_data = cls.__new__(cls)
        # evaluate the data
        data = np.histogramdd(points, bins=edges_list, weights=weights)[0]
        # initialize the object with the data
        binned_data.__init__(data, edges_list)
        return binned_data

    @classmethod
    def from_point_vals(
        cls,
        points,
        values,
        edges_list,
        weights=None,
        ufunc=np.mean,
        default_val=np.nan,
        min_sample_size=None,
    ):
        """ construct the instance though scatter points
        Parameters
        ----------
        points : ndarray
            coordinates of points, shape(N, D)
            means N points of D dimension
        values : ndarray
            values for each points, shape(N, 1)
        edges_list : list
            edges for bins,
            the length of elements should be (i_1 + 1, i_2 + 1, i_3 + 1, ..., i_D + 1)
        weights : ndarray
            weights of points, shape(N)
        ufunc : function
            function for aggregation of data in each bin, default is len
            if weights is not None, ufunc mush have a weights parameter,
            excepet len, since we are using histogramdd instead.
        default_val : float or np.nan
            default_val for ufunc applied on empty array
        min_sample_size : int or None
            if int, the bin that has values under this value will be
            assigned as default_val

        Returns
        ----------
        BinnedData2D instance
        """
        assert points.shape[1] == len(edges_list)
        binned_data = cls.__new__(cls)
        # evaluate the data
        data = cls._aggregation_with_ufunc(
            points, values, edges_list, weights, ufunc, default_val, min_sample_size
        )
        # initialize the object with the data
        binned_data.__init__(data, edges_list)
        return binned_data

    @staticmethod
    def _aggregation_with_ufunc(
        points, values, edges_list, weights, ufunc, default_val, min_sample_size
    ):
        """
        aggregating data with ufunc by applying ufunc on points
        in each bin
        """
        edges_tuple_list = np.array(
            [list(zip(edges[:-1], edges[1:])) for edges in edges_list]
        )
        val_list, size_list = [], []
        for edges_tuples in product(*edges_tuple_list):
            mask = np.all(
                [
                    (points[:, i] >= v_l) & (points[:, i] < v_u)
                    for i, (v_l, v_u) in enumerate(edges_tuples)
                ],
                axis=0,
            )
            values_in_bin = values[mask]
            if weights is not None:
                val = ufunc(values_in_bin, weights=weights)
            else:
                val = ufunc(values_in_bin)
            val_list.append(val)
            size_list.append(len(values_in_bin))
        # deal with the empty bins
        val_arr = np.where(np.array(size_list) > 0, np.array(val_list), default_val)
        # deal with bins with size smaller than the min_sample_size
        if min_sample_size is not None:
            val_arr = np.where(
                np.array(size_list) >= min_sample_size, val_arr, default_val
            )
        # reshape to get the data
        data = np.array(val_arr).reshape([len(e) - 1 for e in edges_list])
        return data

    def save(self, filename, comments=None):
        """save the instance in a fits file

        Parameters
        ----------
        filename : string
            file path and name to store the data
        comments : string, optional
            comments for the header of primary HDU
        """
        # primary hdu
        header = fits.Header()
        for i, edges in enumerate(self._edges_list):
            header["SHAPE_%02d" % i] = len(edges) - 1
        if comments is not None:
            header["COMMENTS"] = comments
        primary_hdu = fits.PrimaryHDU(header=header)
        # data
        data_hdu = fits.ImageHDU(data=self._data)
        # edges
        hdu_list = [primary_hdu, data_hdu]
        for i, edges in enumerate(self._edges_list):
            edges_df = pd.DataFrame({"edges_%02d" % i: edges})
            edges_hdu = fits.table_to_hdu(Table.from_pandas(edges_df))
            hdu_list.append(edges_hdu)
        # write to file
        hdul = fits.HDUList(hdu_list)
        hdul.writeto(filename)

    @property
    def data(self):
        """return the data"""
        return self._data

    @property
    def edges_list(self):
        """return the x_edges"""
        return self._edges_list

    @property
    def edges_mid(self):
        """return the medium of x edges"""
        if getattr(self, "_edges_mid_list", None) is None:
            self._edges_mid_list = []
            for edges in self._edges_list:
                self._edges_mid_list.append((edges[:-1] + edges[1:]) / 2)
        return self._edges_mid_list

    def infer(self, points, fill_value=None):
        """infer the value at position given by points

        Parameters
        ----------
        points : ndarray
            D dimensional position, shape(N, D)
        fill_value : number
            fill value for outliers

        Returns
        ----------
        ndarray
        shape(N, ), infered value for points
        """
        if points.shape[1] != self._dim:
            raise Exception(
                "The dimension of points is %d, while the binned data dimension is %d"
                % (points.shape[1], self._dim)
            )
        ids_list = []
        outlier_mask = np.full(len(points), False)
        for i, edges in enumerate(self._edges_list):
            id_tmp = np.digitize(points[:, i], edges) - 1
            outlier_mask = outlier_mask | (id_tmp == len(edges) - 1) | (id_tmp == -1)
            id_tmp[id_tmp == len(edges) - 1] = len(edges) - 2
            id_tmp[id_tmp == -1] = 0
            ids_list.append(id_tmp)
        val_infer = self._data[tuple(ids_list)]
        if fill_value is not None:
            val_infer[outlier_mask] = fill_value
        return val_infer

    def imshow(self, ax, projection, missing_value=np.nan, **kwargs):
        """plot the histogram2d though imshow

        Parameters
        ----------
        ax : Axes object
            to plot the figure
        projection : tuple
            length = 2, the direction to project the data
        **kwargs : dict
            kw args for imshow function

        Returns
        ----------
        image : AxesImage
        """
        data, x_edges, y_edges = self._reduce_data(self._data, projection)
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
        data[np.isnan(data)] = missing_value
        default_config = {"origin": "lower", "aspect": "auto"}
        if len(kwargs) != 0:
            default_config.update(kwargs)
        obj = ax.imshow(data.T, extent=extent, **default_config)
        return obj

    def contour(self, axis, projection, missing_value=np.nan, **kwargs):
        """plot the histogram2d though contour

        Parameters
        ----------
        axis : Axes object
            to plot the figure
        projection : tuple
            length = len(edges_list), the direction to project the data
            "x", "y" : for project direction
            "-1" : summation over this direction
            int : slice in this direction
        **kwargs : dict
            kw args for imshow function

        Returns
        ----------
        image : AxesImage
        """
        data, x_edges, y_edges = self._reduce_data(self._data, projection)
        xs = (x_edges[:-1] + x_edges[1:]) / 2
        ys = (y_edges[:-1] + y_edges[1:]) / 2
        data[np.isnan(data)] = missing_value
        x_grid, y_grid = np.meshgrid(xs, ys, indexing="ij")
        obj = axis.contour(x_grid, y_grid, data, **kwargs)
        return obj

    def _parse_projection(self, projection):
        reverse = projection.index("x") > projection.index("y")
        x_edges = self._edges_list[projection.index("x")]
        y_edges = self._edges_list[projection.index("y")]

        parti = []
        for i, val in enumerate(projection):
            if val == -1 or val == "x" or val == "y":
                parti.append(slice(len(self._edges_list[i])))
            elif (val >= 0) and (val < len(self._edges_list[i])):
                parti.append(slice(val, val + 1))
            else:
                raise Exception(
                    "Input %d-th parameter is not legal: %s" % (i, str(val))
                )
        integ_ids = [i for i, val in enumerate(projection) if val != "x" and val != "y"]
        return reverse, tuple(integ_ids), tuple(parti), (x_edges, y_edges)

    def _reduce_data(self, data, projection):
        reverse, integ_ids, parti, (x_edges, y_edges) = self._parse_projection(
            projection
        )
        data_reduced = np.sum(data[parti], axis=integ_ids)
        if reverse:
            data_reduced = data_reduced.T
        return data_reduced, x_edges, y_edges
