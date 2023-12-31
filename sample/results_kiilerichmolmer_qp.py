# The GPLv3 License (GPLv3)

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
import lightpulse as lp


def scattering_gaussian_mode():
    """
    Runs the simulation for the scattering of a Gaussian pulse with single photon on an empy cavity
    """
    # Constants of the simulation

    GAMMA = 1.
    N_DIMS = 2
    MU = 4
    SIGMA = 1
    ARGS = {'mu': MU, 'sigma': SIGMA}
    # Operators
    operA = qt.tensor(qt.destroy(N_DIMS), qt.qeye(N_DIMS))
    operC = qt.tensor(qt.qeye(N_DIMS), qt.destroy(N_DIMS))
    # Initial state, sigle photon in the pulse
    rho0 = qt.tensor(qt.fock_dm(N_DIMS, 1), qt.fock_dm(N_DIMS, 0))
    # Define the times to integrate at
    tlist = np.linspace(0, 12, 10000)
    # Run the simulation
    result = qt.mesolve(
        lp.gaussian_total_H_t([operA, operC], _gamma=GAMMA,
                              _args=ARGS), rho0, tlist,
        qt.lindblad_dissipator(
            lp.gaussian_total_damping_oper_t(operA,
                                             operC,
                                             _gamma=GAMMA,
                                             _args=ARGS)),
        [operA.dag() * operA, operC.dag() * operC])

    # Write the expectation values to file
    with open('./outputs/results_kiilerichmolmer_qp/adaga_expect_values.dat',
              'w') as f:
        for i in range(len(result.expect[0])):
            f.write(str(result.expect[0][i]) + '\n')

    with open('./outputs/results_kiilerichmolmer_qp/cdagc_expect_values.dat',
              'w') as f:
        for i in range(len(result.expect[1])):
            f.write(str(result.expect[1][i]) + '\n')


def scattering_stimulated_emission():
    """
    Runs the simulation for the stimulated emission
    """
    # Constants of the simulation
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
        lp.exponential_total_H_t(operAu,
                                 operC,
                                 operAv,
                                 _gamma=gamma,
                                 _args=ARGS), rho0, tlist,
        qt.lindblad_dissipator(
            lp.exponential_total_damping_oper_t([operAu, operC, operAv],
                                                _gamma=gamma,
                                                _args=ARGS)))

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
    scattering_gaussian_mode()
    scattering_stimulated_emission()


if __name__ == "__main__":
    main()
