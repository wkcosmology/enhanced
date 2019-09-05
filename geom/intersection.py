#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File            : intersection.py
# Author          : Kai Wang <wkcosmology@gmail.com>
# Date            : 09.02.2019
# Last Modified By: Kai Wang <wkcosmology@gmail.com>
import numpy as np


def insec_circles(x_main, y_main, r_main, x_ref, y_ref, r_ref):
    """calculate the intersection area between two circles

    Parameters
    ----------
    x1 : TODO
    y1 : TODO
    r1 : TODO
    x2 : TODO
    y2 : TODO
    r2 : TODO

    Returns
    -------
    TODO

    """
    x_ref = np.atleast_1d(x_ref)
    y_ref = np.atleast_1d(y_ref)
    r_ref = np.atleast_1d(r_ref)
    x_main = np.broadcast_to(x_main, len(x_ref))
    y_main = np.broadcast_to(y_main, len(x_ref))
    r_main = np.broadcast_to(r_main, len(x_ref))
    assert np.all(r_main >= 0)
    assert np.all(r_ref >= 0)
    insec_areas = np.zeros(len(x_main))
    d = np.hypot(x_main - x_ref, y_main - y_ref)
    insec_areas[r_main >= r_ref + d] = (np.pi * r_ref * r_ref)[r_main >= r_ref + d]
    insec_areas[r_ref >= r_main + d] = (np.pi * r_main * r_main)[r_ref >= r_main + d]
    nontrivial_mask = ~(
        (r_main == 0) |
        (r_ref == 0) |
        (r_main >= r_ref + d) |
        (r_ref >= r_main + d) |
        (d >= r_main + r_ref))
    ang1 = np.arccos((r_main * r_main + d * d - r_ref * r_ref) / (2 * r_main * d)) * 2
    ang2 = np.arccos((r_ref * r_ref + d * d - r_main * r_main) / (2 * r_ref * d)) * 2
    # nontrivial_area = r_main * r_main * ang1 + r_main * r_ref * ang2 - np.sin(ang1) * r_main * d
    area1 = 0.5 * ang1 * r_main * r_main - 0.5 * r_main * r_main * np.sin(ang1)
    area2 = 0.5 * ang2 * r_ref * r_ref - 0.5 * r_ref * r_ref * np.sin(ang2)
    nontrivial_area = area1 + area2
    insec_areas = np.where(
        nontrivial_mask,
        nontrivial_area,
        insec_areas)
    return insec_areas
