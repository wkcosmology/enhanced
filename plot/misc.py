#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File            : misc.py
# Author          : Kai Wang <wkcosmology@gmail.com>
# Date            : 01.14.2020
# Last Modified By: Kai Wang <wkcosmology@gmail.com>

import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def add_text(ax, x, y, s, fontdict=None, **kwargs):
    """add the text to axis using relative position

    Parameters
    ----------
    ax : ax
    x : float
        0 <= x <= 1, x coordinate
    y : float
        0 <= x <= 1, y coordinate
    s : str
        text content
    **kwargs : Text properties
        Other miscellaneous text parameters.
    fontdict : dict, optional
        A dictionary to override the default text properties.
        If fontdict is None, the defaults are determined by your rc parameters.

    Returns
    -------
    text : Text
        The created Text instance
    """
    font_default = {
        "family": "sans-serif",
        "style": "normal",
        "fontsize": 22}
    if fontdict is not None:
        font_default.update(fontdict)
    ax.text(x, y, s, fontdict=font_default, transform=ax.transAxes)


def set_locator(ax, x_interval=None, y_interval=None, major_tick_params=None, minor_tick_params=None):
    if ax.get_xscale() == 'log':
        x_locator = ticker.LogLocator
    else:
        x_locator = ticker.MultipleLocator
    if ax.get_yscale() == 'log':
        y_locator = ticker.LogLocator
    else:
        y_locator = ticker.MultipleLocator
    if x_interval is not None:
        ax.xaxis.set_major_locator(x_locator(x_interval[0]))
        if ax.get_xscale() != 'log':
            ax.xaxis.set_minor_locator(x_locator(x_interval[1]))
    if y_interval is not None:
        ax.yaxis.set_major_locator(y_locator(y_interval[0]))
        if ax.get_xscale() != 'log':
            ax.yaxis.set_minor_locator(y_locator(y_interval[1]))
    major_tick_params_ = {
        "width": 1.2,
        "length": 5}
    minor_tick_params_ = {
        "width": 1.2,
        "length": 3}
    if major_tick_params is not None:
        major_tick_params_.update(major_tick_params)
    if minor_tick_params is not None:
        minor_tick_params_.update(minor_tick_params)
    ax.tick_params(which="major", **major_tick_params_)
    ax.tick_params(which="minor", **minor_tick_params_)

def out_colorbar(mappable, fig, cax, bbox_to_anchor, use_gridspec=True, **kwargs):
    axins = inset_axes(
        cax,
        width="100%",
        height="100%",
        loc='lower left',
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=cax.transAxes,
        borderpad=0)
    fig.colorbar(mappable, cax=axins)
