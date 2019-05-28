#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File            : functions.py
# Author          : Kai Wang <wkcosmology@gmail.com>
# Date            : 05.18.2019
# Last Modified By: Kai Wang <wkcosmology@gmail.com>
import numpy as np


def double_linear(xs, m, n, c, yc, amp):
    """double_linear

    Parameters
    ----------
    x : ndarray
        the input xs
    m : float
        slope for x < c
    n : float
        slope for x >= c
    c : float
        turn-over point
    yc : float
        y value at x = c
    amp : float
        global amplitude

    Returns
    -------
    the y values corresponding to the input xs
    shape (len(xs))
    """
    ys = np.empty(len(xs))
    ys[xs < c] = (yc + m * xs - m * c)[xs < c]
    ys[xs >= c] = (yc + n * xs - n * c)[xs >= c]
    return amp * ys


def schechter(xs, amp, x_c, alpha):
    """schechter

    Parameters
    ----------
    xs : ndarray
        the input xs
    amp : float
        the amplitude
    x_c : float
        turn-over point
    alpha : float
        the slope of the power law part

    Returns
    -------
    the y values corresponding to the input xs
    shape (len(xs))
    """
    return amp * (xs / x_c)**alpha * np.exp(-xs / x_c) / x_c


def schechter_log(logx, amp, logx_c, alpha):
    """schechter_log

    Parameters
    ----------
    logx : ndarray
        the input log(x)
    amp : float
        the amplitude
    logx_c : float
        the log of the turn over point
    alpha : float
        the slope of the power law part

    Returns
    -------
    the y values corresponding to the input xs
    shape (len(logx))
    """
    return (np.log(10) * amp *
            np.exp(-10**(logx - logx_c)) *
            (10**((alpha + 1) * (logx - logx_c))))
