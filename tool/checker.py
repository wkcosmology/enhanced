#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File            : data_check.py
# Author          : Kai Wang <wkcosmology@gmail.com>
# Date            : 05.20.2019
# Last Modified By: Kai Wang <wkcosmology@gmail.com>

def check_columns_in_table(table, columns):
    """check if the table contains the given columns

    Parameters
    ----------
    table : astropy.table.Table
    columns : list
        list of string, which are columns the table should contains

    Raise
    ---------
    AttributeError
        Table does not contain the column {column_name}
    """
    for col in columns:
        if col not in table.colnames:
            raise AttributeError("Table does not contain the column: {0:s}".format(col))


def check_columns_in_dataframe(dataframe, columns):
    """check if the DataFrame contains the given columns

    Parameters
    ----------
    dataframe : pandas.DataFrame
        the dataframe to check
    columns : list
        list of string, which are columns the table should contains

    Raise
    ---------
    AttributeError
        DataFrame does not contain the column {column_name}
    """
    for col in columns:
        if col not in dataframe.columns:
            raise AttributeError("DataFrame does not contain the column: {0:s}".format(col))


def check_type(arg, arg_type):
    """check the type of arg

    Parameters
    ----------
    arg :
        an object
    arg_type : type
        a type

    Raises
    -------
    TypeError
        This argement is not and instance of {type}

    """
    if not isinstance(arg, arg_type):
        raise TypeError("This arguement is not an instance of {0:s}".format(arg_type.__name__))
