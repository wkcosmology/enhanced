#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File            : scatter.py
# Author          : Kai Wang <wkcosmology@gmail.com>
# Date            : 05.30.2019
# Last Modified By: Kai Wang <wkcosmology@gmail.com>

import numpy as np
from matplotlib import gridspec
from enhanced.stats.binned_stats import binned_percentile


class Plot2DScatter:
    """Plot2DScatter"""
    def __init__(self, fig, x_edges, y_edges, x_label=None, y_label=None):
        """__init__

        Parameters
        ----------
        fig : :class:matplot.pyplot.figure
            a figure object
        x_edges : array_like
            the bin edges for the x_data
        y_edges : array_like
            the bin edges for the y_data
        x_label : str, optional
            x_label
        y_label : str, optional
            y_label
        """
        self._x_edges = x_edges
        self._y_edges = y_edges
        gs = gridspec.GridSpec(4, 4, wspace=0.01, hspace=0.01)
        self.ax1 = fig.add_subplot(gs[1:4, 0:3])
        self.ax2 = fig.add_subplot(gs[0, 0:3], sharex=self.ax1)
        self.ax3 = fig.add_subplot(gs[1:4, 3], sharey=self.ax1)
        self.ax2.tick_params(axis='x', which='both', labelsize=0)
        self.ax3.tick_params(axis='y', which='both', labelsize=0)
        self.ax3.tick_params(axis='x', which='both', labelrotation=-90)
        if x_label is not None:
            self.ax1.set_xlabel(x_label, fontsize='x-large')
        if y_label is not None:
            self.ax1.set_ylabel(y_label, fontsize='x-large')

        delta_x = 0.1 * (self._x_edges[-1] - self._x_edges[1])
        delta_y = 0.1 * (self._y_edges[-1] - self._y_edges[1])
        x_lim = self._x_edges[1] - 2 * delta_x, self._x_edges[-1] + delta_x
        y_lim = self._y_edges[1] - 2 * delta_y, self._y_edges[-1] + delta_y
        self.ax2.set_xlim(x_lim)
        self.ax3.set_ylim(y_lim)

    def add_component(self, xs, ys, color=None, marker=None, size=None):
        """add_component

        Parameters
        ----------
        xs : array_like
            the x data
        ys : array_like
            the y data
        color : str
            name of colors
        marker : str
            name of markers
        size : float
            size of the marker
        """
        self.ax1.scatter(xs, ys, c=color, linewidth=0, marker=marker, s=size)
        self.ax2.hist(xs, bins=self._x_edges, histtype="step", color=color, linewidth=2)
        self.ax3.hist(ys, bins=self._y_edges, histtype="step", orientation="horizontal", color=color, linewidth=2)


def plot_2d_percentile(ax, xs, ys, x_edges, percens, color=None):
    """plot_2d_percentile

    Parameters
    ----------
    ax : :class: matplotlib.axes.Axes
        an axes object
    xs : array_like
        input x values
    ys : array_like
        input y values
    x_edges : array_like
        bin edges of x values
    percens : array_like
        the percentile to calculate
    color : str, optional
        the color of the lines
    """
    if color is None:
        color = 'k'
    res = binned_percentile(xs, ys, x_edges=x_edges, percens=percens)
    for p, v in zip(percens, res[1:]):
        width = (1 - np.abs(p / 100 - 0.5)) / 2 * 0.9
        ax.errorbar(
            res[0], v, xerr=width * np.diff(x_edges),
            fmt='.', capsize=0, c=color, lw=2,
            label=str(p))
    ax.vlines(res[0], res[1], res[-1], lw=2)
    ax.legend(loc='best')
