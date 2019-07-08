#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File            : scatter.py
# Author          : Kai Wang <wkcosmology@gmail.com>
# Date            : 05.30.2019
# Last Modified By: Kai Wang <wkcosmology@gmail.com>

import numpy as np
import matplotlib.pyplot as plt
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


class Plot2DScatterWithBinnedHist:
    """This class is to visualize the 2-dimension scatter plot.

    This will divide a fig into 2x2 subplot.
    (0, 1) will be scatter plot for xs and ys
    (0, 0) will be len(x_edges) - 1 histgrams for y data is each x bin
    (1, 1) will be len(y_edges) - 1 histgrams for x data is each y bin
    (1, 0) will be empty
    """
    def __init__(self, fig, x_edges, y_edges, hist_x_edges, hist_y_edges, x_label=None, y_label=None):
        """__init__

        Parameters
        ----------
        fig : :class:matplot.pyplot.figure
            a figure object
        x_edges : array_like
            the bin edges for the x_data
        y_edges : array_like
            the bin edges for the y_data
        hist_x_edges : array_like
            the x axis edges for ploting the histgram
        hist_y_edges : array_like
            the y axis edges for ploting the histgram
        x_label : str, optional
            x_label
        y_label : str, optional
            y_label
        """
        self._x_edges = x_edges
        self._y_edges = y_edges
        self._hist_x_edges = hist_x_edges
        self._hist_y_edges = hist_y_edges
        self.row = (len(y_edges) - 1) * 2
        self.col = (len(x_edges) - 1) * 2
        gs = gridspec.GridSpec(self.row, self.col, wspace=0.01, hspace=0.01)
        self.ax1 = fig.add_subplot(gs[int(self.row / 2):, 0:int(self.col / 2)])
        self.ax1.set_xlim(self._x_edges[0], self._x_edges[-1])
        self.ax1.set_ylim(self._y_edges[0], self._y_edges[-1])

        self.ax2, self.ax3 = [], []
        for i in range(int(self.col / 2)):
            self.ax2.append(fig.add_subplot(gs[0:int(self.row / 2), i]))
            self.ax2[i].tick_params(axis='both', which='both', labelsize=0, length=0)
        for i in range(int(self.row / 2)):
            self.ax3.append(fig.add_subplot(gs[self.row - i - 1, int(self.col / 2):]))
            self.ax3[i].tick_params(axis='both', which='both', labelsize=0, length=0)
        if x_label is not None:
            self.ax1.set_xlabel(x_label, fontsize='x-large')
        if y_label is not None:
            self.ax1.set_ylabel(y_label, fontsize='x-large')

    def add_component(self, xs, ys, color=None, marker=None, size=None, label=None, density=False):
        """add_component

        Parameters
        ----------
        xs : array_like
            the x data
        ys : array_like
            the y data
        color : str, optional
            name of colors
        marker : str, optional
            name of markers
        size : float, optional
            size of the marker
        label : str, optional
            label of this component
        """
        self.ax1.scatter(xs, ys, c=color, linewidth=0, marker=marker, s=size, label=label)
        for i in range(int(self.col / 2)):
            mask = (xs > self._x_edges[i]) & (xs < self._x_edges[i + 1])
            self.ax2[i].hist(
                ys[mask],
                bins=self._hist_y_edges,
                histtype="step",
                orientation="horizontal",
                color=color,
                linewidth=2,
                density=density)
        for i in range(int(self.row / 2)):
            mask = (ys > self._y_edges[i]) & (ys < self._y_edges[i + 1])
            self.ax3[i].hist(
                xs[mask],
                bins=self._hist_x_edges,
                histtype="step",
                color=color,
                linewidth=2,
                density=density)
        self.ax1.legend(loc=(1.25, 1.25), fontsize='x-large')


def plot_2d_relative_fraction(
        fig, data1_xs, data1_ys, data2_xs, data2_ys,
        x_edges, y_edges, x_label=None, y_label=None,
        data1_label=None, data2_label=None):
    """TODO: Docstring for Plot2DScatterFractionalImg.

    Parameters
    ----------
    fig : TODO
    xs : TODO
    ys : TODO
    x_edges : TODO
    y_edges : TODO

    Returns
    -------
    TODO

    """
    count1 = np.zeros((len(x_edges) - 1, len(y_edges) - 1))
    count2 = count1.copy()
    tot_count = count1.copy()
    fraction1 = count1.copy()
    fraction2 = count1.copy()

    for i in range(len(x_edges) - 1):
        for j in range(len(y_edges) - 1):
            count1[i, j] = np.count_nonzero(
                (data1_xs < x_edges[i + 1]) &
                (data1_xs > x_edges[i]) &
                (data1_ys < y_edges[j + 1]) &
                (data1_ys > y_edges[j]))
            count2[i, j] = np.count_nonzero(
                (data2_xs < x_edges[i + 1]) &
                (data2_xs > x_edges[i]) &
                (data2_ys < y_edges[j + 1]) &
                (data2_ys > y_edges[j]))
    tot_count = (count1 + count2)
    fraction1 = np.where(tot_count != 0, count1 / tot_count, np.nan).T
    fraction2 = np.where(tot_count != 0, count2 / tot_count, np.nan).T

    gs = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    x_grid, y_grid = np.meshgrid(x_edges, y_edges)
    bar1 = ax1.pcolor(x_grid, y_grid, fraction1)
    plt.colorbar(bar1)

    ax2 = fig.add_subplot(gs[0, 1])
    bar2 = ax2.pcolor(x_grid, y_grid, fraction2)
    plt.colorbar(bar2)

    if x_label is not None:
        ax1.set_xlabel(x_label, fontsize='x-large')
        ax2.set_xlabel(x_label, fontsize='x-large')
    if y_label is not None:
        ax1.set_ylabel(y_label, fontsize='x-large')
        ax2.set_ylabel(y_label, fontsize='x-large')
    if data1_label is not None and data2_label is not None:
        ax1.set_title("{0:s} / ({0:s} + {1:s})".format(data1_label, data2_label))
        ax2.set_title("{1:s} / ({0:s} + {1:s})".format(data1_label, data2_label))
