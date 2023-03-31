# The GPLv3 License (GPLv3)
#
# Copyright (c) 2023 Simone Pirota
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
This module implements all the physical elements of our simulation,
such as the different pulse shapes, the resulting interaction coefficients
(g_u(t), g_v(t) and so on) and the interaction Hamiltonian as presented
by Kiilerich and Molmer in Input-Output Theory with Quantum Pulses.
"""

import math
import numpy as np
import qutip as qt

#########################
#  Gaussian shape pulse #
#########################


def gaussian_sqrt(_t, _mu=0., _sigma=1.):
    """
    Returns the Gaussian function
    """
    x = float(_t - _mu) / _sigma
    return math.sqrt(math.exp(-x * x / 2.) / math.sqrt(2 * math.pi) / _sigma)


def g_u_gaussian(_t, _args):
    """
    Gaussian time dependent coupling with the input cavity
    """
    mu = _args['mu']
    sigma = _args['sigma']
    if _t > mu + 4 * sigma:
        return 0
    else:
        x = (_t - mu) / (math.sqrt(2) * sigma)
        denominator = math.sqrt(1 - 0.5 * (1 + math.erf(x)))
        return gaussian_sqrt(_t, mu, sigma) / denominator


############################
#  Exponential shape pulse #
############################
"""
The exponential pulse shape is the optimal shape to
describe stimulated emission
"""


def g_u_exponential(_t, _args):
    """
    Exponential time dependent coupling with the input cavity
    """
    _GAMMA = _args['GAMMA']
    return math.sqrt(_GAMMA) * np.heaviside(_t, 1)


def g_v_exponential(_t, _args):
    """
    Exponential time dependent coupling with the output cavity
    """
    _GAMMA = _args['GAMMA']
    if _t > 0:
        return np.heaviside(_t, 1) * math.sqrt(_GAMMA /
                                               (math.exp(_GAMMA * _t) - 1))
    else:
        return 0


def g_u_v_exponential(_t, _args):
    """
    Exponential time dependent coupling between the input and output modes
    """
    return g_u_exponential(_t, _args) * g_v_exponential(_t, _args)
