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
    return math.sqrt(_GAMMA) * np.heaviside(_t, 1)


def g_v_stimulated_emission(_t, _args):
    _GAMMA = _args['GAMMA']
    if _t > 0:
        return np.heaviside(_t, 0) * math.sqrt(_GAMMA /
                                               (math.exp(_GAMMA * _t) - 1))
    else:
        return 0


def g_u_v_stimulated_emission(_t, _args):
    return g_u_stimulated_emission(_t, _args) * g_v_stimulated_emission(
        _t, _args)


def qubit_H(_w_0=1.):
    """
    This function is ill-defined, TODO: implement it better!!
    Returns the hamiltonian of a two level atom, with two ancillary system of
    dimension 2 and 3
    """
    return _w_0 * 0.5 * (qt.tensor(qt.qeye(2), qt.sigmaz(), qt.qeye(3)) + 1)


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


def total_H_t(_operAu,
              _operAv,
              _operC,
              _w_0=1.,
              _gamma=1.,
              _args={
                  'GAMMA': 1,
              }):
    """
    Returns the QobjEvo with the right time dependencies
    """
    return qt.QobjEvo([
        qubit_H(_w_0),
        [only_input_inter_H(_operAu, _operC, _gamma), g_u_stimulated_emission],
        [
            only_output_inter_H(_operAv, _operC, _gamma),
            g_v_stimulated_emission
        ],
        [
            input_output_inter_H(_operAu, _operAv, _gamma),
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
            conj_input_output_inter_H(_operAu, _operAv, _gamma),
            g_u_v_stimulated_emission
        ]
    ],
                      args=_args)


def damping_oper(_operC, _gamma=1.):
    """
    Time independet damping operator
    """
    return np.sqrt(_gamma) * _operC


def total_damping_oper_t(_operAu,
                         _operAv,
                         _operC,
                         _gamma=1.,
                         _args={
                             'GAMMA': 1,
                         }):
    """
    Returns the complete damping operator, with the time dependent part
    """
    return qt.QobjEvo([
        damping_oper(_operC, _gamma), [_operAu, g_u_stimulated_emission],
        [_operAv, g_v_stimulated_emission]
    ],
                      args=_args)


def scattering_stimulated_emission():
    """
    Runs the simulation for the stimulated emission
    """
    # Constants of the simulation

    # The Halmitonian in the master equation is in the interaction representation w.r.t H_S (system hamiltonian)
    # this means that H_S plays no role in the dymanic, and the easiest way to not have its contribution, is to
    # take W_0 = 0 BEWARE: this has no real physical meaning, it is just a shortcut to not rewrite several lines of code
    W_0 = 0.

    gamma = 1.
    GAMMA = float(gamma) / 0.36
    ARGS = {'GAMMA': GAMMA}
    N_U = 2
    N_S = 2
    N_V = 3
    # Operators
    operAu = qt.tensor(qt.destroy(N_U), qt.qeye(N_S), qt.qeye(N_V))
    operAv = qt.tensor(qt.qeye(N_U), qt.qeye(N_S), qt.destroy(N_V))
    operC = qt.tensor(qt.qeye(N_U), qt.sigmam(), qt.qeye(N_V))
    # Initial state
    rho0 = qt.tensor(qt.basis(N_U, 1), qt.basis(N_S, 0), qt.basis(N_V, 0))
    # times to integrate the me at
    tlist = np.linspace(0, 4, 10000)
    # Run the simulation
    result = qt.mesolve(
        total_H_t(operAu, operAv, operC, W_0, gamma, ARGS), rho0, tlist,
        qt.lindblad_dissipator(
            total_damping_oper_t(operAu, operAv, operC, gamma, ARGS)))

    # Write the expectation values to file
    with open('./outputs/results_kiilerichmolmer_qp_fig4/input_one_photon.dat',
              'w') as f:
        input_pulse_state = qt.basis(N_U, 1)
        for i in range(len(result.states)):
            prob = np.abs(
                result.states[i].ptrace(0).overlap(input_pulse_state))**2
            f.write(str(prob) + '\n')

    with open('./outputs/results_kiilerichmolmer_qp_fig4/excited_atom.dat',
              'w') as f:
        excited_atom_state = qt.basis(N_S, 0)
        for i in range(len(result.states)):
            prob = np.abs(
                result.states[i].ptrace(1).overlap(excited_atom_state))**2
            f.write(str(prob) + '\n')

    with open(
            './outputs/results_kiilerichmolmer_qp_fig4/output_one_photon.dat',
            'w') as f:
        output_pulse_state_one = qt.basis(N_V, 1)
        for i in range(len(result.states)):
            prob = np.abs(
                result.states[i].ptrace(2).overlap(output_pulse_state_one))**2
            f.write(str(prob) + '\n')

    with open(
            './outputs/results_kiilerichmolmer_qp_fig4/output_two_photon.dat',
            'w') as f:
        output_pulse_state_two = qt.basis(N_V, 2)
        for i in range(len(result.states)):
            prob = np.abs(
                result.states[i].ptrace(2).overlap(output_pulse_state_two))**2
            f.write(str(prob) + '\n')


def main():
    scattering_stimulated_emission()


if __name__ == "__main__":
    main()
