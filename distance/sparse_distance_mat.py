#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File            : sparse_distance_mat.py
# Author          : Kai Wang <wkcosmology@gmail.com>
# Date            : 05.19.2019
# Last Modified By: Kai Wang <wkcosmology@gmail.com>
import numpy as np
from scipy.sparse import coo_matrix
from halotools.mock_observables.pair_counters import pairwise_distance_xy_z
from halotools.mock_observables.pair_counters import pairwise_distance_3d
from enhanced.tool.checker import check_columns_in_dataframe
from grouptools.utilities.halo_model import HaloProfile
from cached_property import cached_property

class SparseDistanceMatrix(object):

    """Creat a sparse distance sparse distance matrix"""

    def __init__(self, row_pos, col_pos, rp_max, pi_max=None, dist_mats=None):
        """initilize the object

        Parameters
        ----------
        row_pos : array_like, shape(N, 3)
            the 3-d coordinates of the row points
        col_pos : array_like, shape(N, 3)
            the 3-d coordinates of the row points
        rp_max : array_like, shape(N) or float
            the maximum value of searching radius in the projected direction
        pi_max : array_like, shape(N) or float, optional
            the maximum value of searching radius in the line-of-sight direction,
            if NONE, then using pairwise_distance_3d function
        dist_mats : list, optional
            list of two distance matrix, one for rps and the other for pis
        """
        self._row_pos = row_pos
        self._col_pos = col_pos
        self._rp_max = rp_max
        self._pi_max = pi_max
        if dist_mats is None:
            if self._pi_max is None:
                pws = pairwise_distance_3d(self._row_pos, self._col_pos, self._rp_max)
            else:
                pws = pairwise_distance_xy_z(self._row_pos, self._col_pos, self._rp_max, self._pi_max)
        else:
            pws = dist_mats
        self._rps = pws[0].tocsr()
        self._pis = pws[1].tocsr()
        self._row = np.sort(self._rps.tocoo().row)
        self._col = self._rps.indices

    @property
    def rps(self):
        """return the projected distance matrix"""
        return self._rps

    @property
    def pis(self):
        """return the line-of-sight distance matrix"""
        return self._pis

    @property
    def row(self):
        """return the row of the sparse matrix in the order of csr """
        return self._row

    @property
    def col(self):
        """return the col of the sparse matix in teh order of csr """
        return self._col

    def update(self, mask):
        """update the sparse matrix using a boolean mask

        Parameters
        ----------
        mask : array_like
            the boolean mask
        """
        self._row = self._row[mask]
        self._col = self._col[mask]
        rps_data = self._rps.data[mask]
        pis_data = self._pis.data[mask]
        shape = self._rps.shape
        self._rps = coo_matrix((rps_data, (self._row, self._col)), shape=shape).tocsr()
        self._pis = coo_matrix((pis_data, (self._row, self._col)), shape=shape).tocsr()

    def copy(self, data):
        """make a copy of the sparse matrix with new data

        Parameters
        ----------
        data : array_like
            the new data

        Returns
        -------
        sparse matrix
            sparse matrix with the same shape of self._rps

        """
        mat = self._rps.copy()
        mat.data = data
        return mat


class HaloGalaxyDistanceMatrix(SparseDistanceMatrix):

    """Construct the distance matrix between the halos and galaxies"""

    def __init__(self, halos, galaxies, photo_z_err, hm="hm", dist_mats=None):
        """initial the object

        Parameters
        ----------
        halos : :class:pandas.DataFrame
            the halo catalogue, columns: {ra, dec, z, co_dist, hm, r_deg}
        galaxies : :class:pandas.DataFrame
            the galaxy catalogue, columns: {ra dec z co_dist z_tag}
        photo_z_err : float
            the standard error of photometric redshift
        hm : str, optional
            the column name of the halo mass
        dist_mats : list, optional
            list of two distance matrix, one for rps and the other for pis
        """
        self._halos = halos.copy()
        self._galaxies = galaxies.copy()
        self._photo_z_err = photo_z_err
        self._hm = hm
        check_columns_in_dataframe(self._halos, "ra dec z co_dist r_deg".split() + [self._hm])
        check_columns_in_dataframe(self._galaxies, "ra dec z co_dist z_tag".split())

        if np.all(self._galaxies["z_tag"] == 1):
            pi_max = 3 * 0.01 * (1 + self._halos["z"].values)
        else:
            pi_max = 3 * self._photo_z_err * (1 + self._halos["z"].values)
        SparseDistanceMatrix.__init__(
            self,
            self._halos["ra dec z".split()].values,
            self._galaxies["ra dec z".split()].values,
            rp_max=self._halos["r_deg"].values,
            pi_max=pi_max,
            dist_mats=dist_mats)

    def _calculate_sigma_zs(self):
        """Calculate the standard error of the redshift """
        disper_vs = HaloProfile().hm2sigma_v(
            self._halos[self._hm].values, self._halos["z"].values)
        if np.any(self._galaxies["z_tag"].values == 0) and self._photo_z_err is None:
            raise Exception("There are photometric galaxies, should give photometric errror")
        z_tag_data = self._galaxies["z_tag"].values[self.col] == 1
        z_data = self._galaxies["z"].values[self.col]
        disper_zs_data = disper_vs[self.row] * (1 + z_data) / 3E5
        photo_zs_data = self._photo_z_err * (1 + z_data)
        sigma_zs = np.where(z_tag_data, disper_zs_data, photo_zs_data)
        return sigma_zs

    @cached_property
    def physical_projected_distance(self):
        """return the sparse matrix of projected distance between galaxies and halos"""
        co_dist_data = self._halos["co_dist"].values[self.row]
        rps_rad_data = np.deg2rad(self.rps.data)
        rps_dist_mat = self.copy(co_dist_data * rps_rad_data)
        return rps_dist_mat

    @cached_property
    def nfw_profile(self):
        """calculate the NFW profile value """
        halo_prof = HaloProfile()
        self._sigma_zs = self._calculate_sigma_zs()
        prof_val = halo_prof.halo_prof(
            self.physical_projected_distance.data,
            self.pis.data,
            self._halos[self._hm].values[self.row],
            self._halos["r200"].values[self.row],
            self._halos["z"].values[self.row],
            self._sigma_zs)
        return self.copy(prof_val)
