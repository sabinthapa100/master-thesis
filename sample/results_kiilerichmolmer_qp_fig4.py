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

import numpy as np
import qutip as qt
import math


def g_u_stimulated_emission(_t, _args):
    _GAMMA = _args['GAMMA']
    return math.sqrt(_GAMMA) * np.heaviside(_t, 0)


def g_v_stimulated_emission(_t, _args):
    _GAMMA = _args['GAMMA']
    # print(_t)
    # print(np.heaviside(_t, 1))
    # print(math.exp(_GAMMA * _t))
    # print(math.exp(_GAMMA * _t) - 1)
    return np.heaviside(_t, 0) * math.sqrt(_GAMMA /
                                           (math.exp(_GAMMA * _t) - 1))


def g_u_v_stimulated_emission(_t, _args):
    return g_u_stimulated_emission(_t, _args) * g_v_stimulated_emission(
        _t, _args)


def oscillator_H(_oper, _w_0=1.):
    """
    Returns the hamiltonian of an harmonic oscillator 
    """
    return _w_0 * _oper.dag() * _oper


def only_input_inter_H(_operAu, _operC, _gamma=1.):
    """
    Calculates the interaction Hamilotonian, with only the input pulse
    """
    return 1j * math.sqrt(_gamma) / 2. * _operAu.dag() * _operC


def only_output_inter_H(_operAv, _operC, _gamma=1.):
    """
    Calculates the interaction Hamilotonian, with only the output pulse
    """
    return 1j * math.sqrt(np.conj(_gamma)) / 2. * _operC.dag() * _operAv


def input_output_inter_H(_operAu, _operAv, _gamma=1.):
    """
    Calculates the interaction Hamilotonian between the two pulses
    """
    return 1j * 0.5 * _operAu.dag() * _operAv


def conj_only_input_inter_H(_operAu, _operC, _gamma=1.):
    """
    Calculates the conjucate of the interaction Hamilotonian, with only the input pulse
    They are separate because in principle the have two different time dependencies
    """
    return -1j * math.sqrt(np.conj(_gamma)) / 2. * _operAu * _operC.dag()


def conj_only_output_inter_H(_operAv, _operC, _gamma=1.):
    """
    Calculates the conjucate of the interaction Hamilotonian, with only the output pulse
    They are separate because in principle the have two different time dependencies
    """
    return -1j * math.sqrt(_gamma) / 2. * _operC * _operAv.dag()


def conj_input_output_inter_H(_operAu, _operAv, _gamma=1.):
    """
    Calculates the conjucate interaction Hamilotonian between the two pulses
    """
    return -1j * 0.5 * _operAu * _operAv.dag()


def total_H_t(_operAu, _operAv, _operC, _w_0=1., _gamma=1.):
    """
    Returns the QobjEvo with the right time dependencies
    """
    return qt.QobjEvo([
        oscillator_H(_operC, _w_0),
        [only_input_inter_H(_operAu, _operC, _gamma), g_u_stimulated_emission],
        [
            only_output_inter_H(_operAv, _operC, _gamma),
            g_v_stimulated_emission
        ],
        [
            input_output_inter_H(_operAv, _operAu, _gamma),
            g_u_v_stimulated_emission
        ],
        [
            conj_only_input_inter_H(_operAu, _operC, _gamma),
            g_u_stimulated_emission
        ],
        [
            conj_only_output_inter_H(_operAv, _operC, _gamma),
            g_v_stimulated_emission
        ],
        [
            conj_input_output_inter_H(_operAv, _operAu, _gamma),
            g_u_v_stimulated_emission
        ]
    ],
                      args={'GAMMA': _gamma / 0.36})


def damping_oper(_operC, _gamma=1.):
    """
    Time independet damping operator
    """
    return np.sqrt(_gamma) * _operC


def total_damping_oper_t(_operAu, _operAv, _operC, _gamma=1.):
    """
    Returns the complete damping operator, with the time dependent part
    """
    return qt.QobjEvo([
        damping_oper(_operC, _gamma), [_operAu, g_u_stimulated_emission],
        [_operAv, g_v_stimulated_emission]
    ],
                      args={'GAMMA': _gamma / 0.36})


def scattering_stimulated_emission():
    """
    Runs the simulation for the stimulated emission
    """
    # print("pippo")
    # Constants of the simulation
    W_0 = 1.
    gamma = 1.
    N_DIMS = 2
    # Operators
    operAu = qt.tensor(qt.destroy(N_DIMS), qt.qeye(N_DIMS), qt.qeye(N_DIMS))
    operAv = qt.tensor(qt.qeye(N_DIMS), qt.qeye(N_DIMS), qt.destroy(N_DIMS))
    operC = qt.tensor(qt.qeye(N_DIMS), qt.destroy(N_DIMS), qt.qeye(N_DIMS))
    # Initial state
    rho0 = qt.tensor(qt.fock_dm(N_DIMS, 1), qt.fock_dm(N_DIMS, 1),
                     qt.fock_dm(N_DIMS, 0))
    # times to integrate the me at
    tlist = np.linspace(0, 4, 10000)
    # Run the simulation
    result = qt.mesolve(
        total_H_t(operAu, operAv, operC, W_0, gamma),
        rho0,
        tlist,
        qt.lindblad_dissipator(
            total_damping_oper_t(operAu, operAv, operC, gamma)),
        [operAu.dag() * operAu,
         operC.dag() * operC,
         operAv.dag() * operAv],
        args={'GAMMA': gamma / 0.36})

    # Write the expectation values to file
    with open(
            './outputs/results_kiilerichmolmer_qp_fig4/audagau_expect_values.dat',
            'w') as f:
        for i in range(len(result.expect[0])):
            f.write(str(result.expect[0][i]) + '\n')

    with open(
            './outputs/results_kiilerichmolmer_qp_fig4/cdagc_expect_values.dat',
            'w') as f:
        for i in range(len(result.expect[1])):
            f.write(str(result.expect[1][i]) + '\n')

    with open(
            './outputs/results_kiilerichmolmer_qp_fig4/avdagav_expect_values.dat',
            'w') as f:
        for i in range(len(result.expect[2])):
            f.write(str(result.expect[2][i]) + '\n')


def main():
    scattering_stimulated_emission()


if __name__ == "__main__":
    main()
