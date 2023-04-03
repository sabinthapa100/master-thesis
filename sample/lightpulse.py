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
import utils
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


######################
# Dynamics Operators #
######################


def only_input_inter_H(*_operators, _gamma=1.):
    """
    Calculates the interaction Hamiltonian, with only the input pulse
    """
    if not _operators:
        raise TypeError("Requires at least one input argument")
    if len(_operators) == 1 and isinstance(_operators[0], (list, np.ndarray)):
        # this is the case when only_input_inter_H is called on the form:
        # only_input_inter_H([q1, q2, q3, ...])
        oper_list = _operators[0]
    else:
        # this is the case when only_input_inter_H is called on the form:
        # only_input_inter_H(q1, q2, q3, ...)
        oper_list = _operators
    if len(oper_list) != 2:
        raise TypeError(
            "Requires exactly two operators, one for the input cavity and the other for the system"
        )
    if not all([isinstance(oper, qt.Qobj) for oper in oper_list]):
        # raise error if one of the inputs is not a quantum object
        raise TypeError("One of inputs is not a quantum object")

    return math.sqrt(_gamma) * oper_list[0].dag() * oper_list[1]


def only_output_inter_H(*_operators, _gamma=1.):
    """
    Calculates the interaction Hamiltonian, with only the output pulse
    """
    if not _operators:
        raise TypeError("Requires at least one input argument")
    if len(_operators) == 1 and isinstance(_operators[0], (list, np.ndarray)):
        # this is the case when only_input_inter_H is called on the form:
        # only_output_inter_H([q1, q2, q3, ...])
        oper_list = _operators[0]
    else:
        # this is the case when only_input_inter_H is called on the form:
        # only_output_inter_H(q1, q2, q3, ...)
        oper_list = _operators
    if len(oper_list) != 2:
        raise TypeError(
            "Requires exactly two operators, one for the system and the other for the output cavity"
        )
    if not all([isinstance(oper, qt.Qobj) for oper in oper_list]):
        # raise error if one of the inputs is not a quantum object
        raise TypeError("One of inputs is not a quantum object")

    return math.sqrt(np.conj(_gamma)) * oper_list[0].dag() * oper_list[1]


def input_output_inter_H(*_operators, _gamma=1.):
    """
    Calculates the interaction Hamiltonian between the two pulses
    """
    if not _operators:
        raise TypeError("Requires at least one input argument")
    if len(_operators) == 1 and isinstance(_operators[0], (list, np.ndarray)):
        # this is the case when input_output_inter_H is called on the form:
        # input_output_inter_H([q1, q2, q3, ...])
        oper_list = _operators[0]
    else:
        # this is the case when input_output_inter_H is called on the form:
        # input_output_inter_H(q1, q2, q3, ...)
        oper_list = _operators
    if len(oper_list) != 2:
        raise TypeError(
            "Requires exactly two operators, one for the input cavity and the other for the output cavity"
        )
    if not all([isinstance(oper, qt.Qobj) for oper in oper_list]):
        # raise error if one of the inputs is not a quantum object
        raise TypeError("One of inputs is not a quantum object")

    return oper_list[0].dag() * oper_list[1]


def conj_only_input_inter_H(*_operators, _gamma=1.):
    """
    Calculates the conjucate of the interaction Hamiltonian, with only the input pulse
    They are separate because in principle the have two different time dependencies
    """
    if not _operators:
        raise TypeError("Requires at least one input argument")
    if len(_operators) == 1 and isinstance(_operators[0], (list, np.ndarray)):
        # this is the case when conj_only_input_inter_H is called on the form:
        # conj_only_input_inter_H([q1, q2, q3, ...])
        oper_list = _operators[0]
    else:
        # this is the case when conj_only_input_inter_H is called on the form:
        # conj_only_input_inter_H(q1, q2, q3, ...)
        oper_list = _operators
    if len(oper_list) != 2:
        raise TypeError(
            "Requires exactly two operators, one for the input cavity and the other for the system"
        )
    if not all([isinstance(oper, qt.Qobj) for oper in oper_list]):
        # raise error if one of the inputs is not a quantum object
        raise TypeError("One of inputs is not a quantum object")

    return math.sqrt(np.conj(_gamma)) * oper_list[0] * oper_list[1].dag()


def conj_only_output_inter_H(*_operators, _gamma=1.):
    """
    Calculates the conjucate of the interaction Hamiltonian, with only the output pulse
    They are separate because in principle the have two different time dependencies
    """
    if not _operators:
        raise TypeError("Requires at least one input argument")
    if len(_operators) == 1 and isinstance(_operators[0], (list, np.ndarray)):
        # this is the case when conj_only_output_inter_H is called on the form:
        # conj_only_output_inter_H([q1, q2, q3, ...])
        oper_list = _operators[0]
    else:
        # this is the case when conj_only_output_inter_H is called on the form:
        # conj_only_output_inter_H(q1, q2, q3, ...)
        oper_list = _operators
    if len(oper_list) != 2:
        raise TypeError(
            "Requires exactly two operators, one for the system and the other for the output cavity"
        )
    if not all([isinstance(oper, qt.Qobj) for oper in oper_list]):
        # raise error if one of the inputs is not a quantum object
        raise TypeError("One of inputs is not a quantum object")

    return math.sqrt(_gamma) * oper_list[0] * oper_list[1].dag()


def conj_input_output_inter_H(*_operators, _gamma=1.):
    """
    Calculates the conjucate interaction Hamiltonian between the two pulses
    """
    if not _operators:
        raise TypeError("Requires at least one input argument")
    if len(_operators) == 1 and isinstance(_operators[0], (list, np.ndarray)):
        # this is the case when conj_input_output_inter_H is called on the form:
        # conj_input_output_inter_H ([q1, q2, q3, ...])
        oper_list = _operators[0]
    else:
        # this is the case when conj_input_output_inter_H  is called on the form:
        # conj_input_output_inter_H (q1, q2, q3, ...)
        oper_list = _operators
    if len(oper_list) != 2:
        raise TypeError(
            "Requires exactly two operators, one for the input cavity and the other for the output cavity"
        )
    if not all([isinstance(oper, qt.Qobj) for oper in oper_list]):
        # raise error if one of the inputs is not a quantum object
        raise TypeError("One of inputs is not a quantum object")

    return oper_list[0] * oper_list[1].dag()


#################
# Special cases #
#################

####################
# Gaussian dynamic #
####################


def gaussian_total_H_t(*_operators, _gamma=1., _args={'mu': 1., 'sigma': 1.}):
    """
    Returns the total interaction Hamiltonian in the case of Gaussian pulses
    """
    return qt.QobjEvo(
        [[only_input_inter_H(_operators, _gamma), g_u_gaussian],
         [conj_only_input_inter_H(_operators, _gamma), g_u_gaussian]],
        args=_args)


def gaussian_total_damping_oper_t(*_operators,
                                  _gamma=1.,
                                  _args={
                                      'mu': 1.,
                                      'sigma': 1.
                                  }):
    """
    Return the damping operator L_0(t) in the case of Gaussian pulses
    """
    if not _operators:
        raise TypeError("Requires at least one input argument")
    if len(_operators) == 1 and isinstance(_operators[0], (list, np.ndarray)):
        # this is the case when gaussian_total_damping_oper_t is called on the form:
        # gaussian_total_damping_oper_t ([q1, q2, q3, ...])
        oper_list = _operators[0]
    else:
        # this is the case when gaussian_total_damping_oper_t  is called on the form:
        # gaussian_total_damping_oper_t (q1, q2, q3, ...)
        oper_list = _operators
    if len(oper_list) != 2:
        raise TypeError(
            "Requires exactly two operators, one for the input cavity and the other for the output cavity"
        )
    if not all([isinstance(oper, qt.Qobj) for oper in oper_list]):
        # raise error if one of the inputs is not a quantum object
        raise TypeError("One of inputs is not a quantum object")
    return qt.QobjEvo([
        utils.damping_oper(oper_list[1], _gamma), [oper_list[0], g_u_gaussian]
    ],
                      args=_args)

#######################
# Exponential dynamic #
#######################