#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File            : io.py
# Author          : Kai Wang <wkcosmology@gmail.com>
# Date            : 05.22.2019
# Last Modified By: Kai Wang <wkcosmology@gmail.com>

from astropy.io import fits
from astropy.table import Table
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def data_frame2fits(filename, df, columns=None, indexs=None, units=None, comments=None):
    """write the data in pandas.DataFrame to a fits file

    Parameters
    ----------
    filename : str
        the output file name
    df : :class:pandas.DataFrame
        the dataframe contains the data
    columns : list, optional
        store the data into len(list) HDU blocks
    indexs : list, optional
        store the data into len(index) HDU blocks
    units : list, optional
        contains the unit for each column
    comments : dict, optional
        comments for the primaryHDU
    """
    if columns is None:
        columns = [list(df.columns)]
    if indexs is None:
        indexs = [np.arange(len(df))]
    assert not ((len(columns) > 1) and (len(index) > 1))
    if units is None or isinstance(units, str):
        units = [units] * max(len(columns), len(indexs))
    assert isinstance(units, list)

    hdr = fits.Header()
    if comments is not None:
        for k, v in comments.items():
            hdr[k] = v
    primary_hdu = fits.PrimaryHDU(header=hdr)
    hdu_list = [primary_hdu]

    if len(columns) > 1:
        for cs, us in zip(columns, units):
            hdu_list.append(_table2hdu_units(Table.from_pandas(df.iloc[indexs[0]][cs]), cs, us))
    else:
        for index, us in zip(indexs, units):
            hdu_list.append(_table2hdu_units(Table.from_pandas(df.iloc[index][columns[0]]), columns[0], us))

    hdul = fits.HDUList(hdu_list)
    hdul.writeto(filename)


def _table2hdu_units(table, colnames=None, units=None):
    """convert a astropy.table.Table object to astropy.io.fits.TableHDU

    Parameters
    ----------
    table : :class:pandas.DataFrame
        the dataframe contains the data
    units : list, optional
        the list of string contains the name of columns
    units : list, optional
        the list of string contains the unit of columns

    Returns
    -------
    :class:astropy.io.fits.PrimaryHDU
    """
    hdu = fits.table_to_hdu(table)
    if units is not None and colnames is not None:
        assert isinstance(units, list)
        for c, u in zip(colnames, units):
            fits.ColDefs(hdu).change_unit(c, u)
    return hdu


def fits2data_frame(filename, hdu=None):
    """read the pandas data frame from a fits file

    Parameters
    ----------
    filename : str
        the filepath and name of the fits file
    hdu : int
        index of HDU to read

    Returns
    -------
    :class: pandas.DataFrame
        the pandas dataframe in the fits file

    """
    f = fits.open(filename)
    if hdu is None and len(f) == 2:
        return Table.read(filename).to_pandas()
    elif hdu is None and len(f) >= 2:
        return [Table.read(filename, hdu=i).to_pandas() for i in range(1, len(f))]
    elif isinstance(hdu, int):
        return Table.read(filename, hdu=hdu).to_pandas()
    else:
        return [Table.read(filename, hdu=i).to_pandas() for i in hdu]


def sparse_matrix2fits(row_pos, col_pos, mat_list, filename, comments=None):
    """store the sparse matrix into a fits file

    Parameters
    ----------
    mat_list : list
        list of :class:scipy.sparse.coo_matrix
    filename : str
        filename for the fits file
    comments : list, optional
        list of comments for each sparse matrix in the above parameter

    """
    header = fits.Header()
    header["NUM_MAT"] = len(mat_list)
    for i in range(len(mat_list)):
        header["ROW-{0:d}".format(i + 1)] = mat_list[i].shape[0]
        header["COl-{0:d}".format(i + 1)] = mat_list[i].shape[1]
        header["LEN-{0:d}".format(i + 1)] = len(mat_list[i].data)
        if comments is not None:
            header["COMM-{0:d}".format(i + 1)] = comments[i]
    primary_hdu = fits.PrimaryHDU(header=header)
    hdu_list = [primary_hdu]

    df_row_pos = pd.DataFrame(data=row_pos, columns="x y z".split())
    df_col_pos = pd.DataFrame(data=col_pos, columns="x y z".split())
    hdu_row = fits.table_to_hdu(Table.from_pandas(df_row_pos))
    hdu_col = fits.table_to_hdu(Table.from_pandas(df_col_pos))
    hdu_list.append(hdu_row)
    hdu_list.append(hdu_col)

    for mat in mat_list:
        mat = mat.tocoo()
        df = pd.DataFrame(
            data=np.column_stack((mat.row, mat.col, mat.data)),
            columns="row col data".split())
        hdu = fits.table_to_hdu(Table.from_pandas(df))
        hdu_list.append(hdu)

    hdul = fits.HDUList(hdu_list)
    hdul.writeto(filename)


def fits2sparse_matrix(filename):
    """read the sparse matrix list from file

    Parameters
    ----------
    filename : str
        input filename

    Returns
    -------
    list:
        row, col, (rps, pis)
    """
    dats = fits.open(filename)
    assert len(dats) == 5
    row_pos = np.array(dats[1].data)
    col_pos = np.array(dats[2].data)
    row_pos_arr = np.array([row_pos["x"], row_pos["y"], row_pos["z"]])
    col_pos_arr = np.array([col_pos["x"], col_pos["y"], col_pos["z"]])
    mat_list = []
    for i in range(3, 5):
        dat = np.array(dats[i].data)
        shape = (dats[0].header["ROW-" + str(i - 2)], dats[0].header["COL-" + str(i - 2)])
        mat = coo_matrix((dat["data"], (dat["row"], dat["col"])), shape=shape)
        mat_list.append(mat)
    return row_pos_arr.T, col_pos_arr.T, mat_list
