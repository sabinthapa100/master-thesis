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

import utils
import numpy as np
import qutip as qt


def compute_energy_ergotropy(_H_evolve, _H_B, _rho0, _times, _c_ops, _w_zero,
                             _filename):
    """ 
    Function that, given an Hamiltonian `_H_evolve` to evolve `_rho0`, outputs to `_filename` 
    energy and ergotropy of the evolution 
    """
    # Resolve the master equation with qutip mesolve
    result = qt.mesolve(_H_evolve, _rho0, _times, _c_ops)
    # Get the partial trace of the state to obtain rho_B
    rho_B = [result.states[i].ptrace(1) for i in range(len(result.states))]

    # write to file the energy results
    with open(
            './outputs/results_energytransferforqb/energy_' + _filename +
            '.dat', 'w') as f:
        for i in range(len(rho_B)):
            f.write(str(qt.expect(_H_B, rho_B[i]) / _w_zero) + '\n')

    # write to file the ergotropy result
    with open(
            './outputs/results_energytransferforqb/ergotropy_' + _filename +
            '.dat', 'w') as f:
        for i in range(len(rho_B)):
            f.write(str(utils.ergotropy(_H_B, rho_B[i]) / _w_zero) + '\n')


def delta_HA_t0(_F, _oper):
    """
    Returns the following qobj, which represents the coherent driving of system A
    """
    return _F * (_oper.dag() + _oper)


def H_AB(_g, _operA, _operB):
    """
    Return the following qobj, which represents the coupling between system A and B 
    """
    return _g * (_operA * _operB.dag() + _operA.dag() * _operB)


def D_A(_N_B, _gamma, _oper):
    """
    Return the following qobj, which represents the dissipation of energy into the environment
    """
    return _gamma * (_N_B + 1) * qt.lindblad_dissipator(_oper)


def D_AD(_N_B, _gamma, _oper):
    """
    Return the following qobj, which represents the dissipation of energy into the environment
    """
    return _gamma * _N_B * qt.lindblad_dissipator(_oper.dag())


def two_harmonic_oscillators():
    # Two harmonic oscillators case
    # Define constants of the simulation
    W_0 = 5.
    G = 0.2 * W_0
    N_DIMS = 15
    # Define the Hamiltonian of system B
    H_B = W_0 * qt.create(N_DIMS) * qt.destroy(N_DIMS)
    # Define the operators that acts on A and B
    operA = qt.tensor(qt.destroy(N_DIMS), qt.qeye(N_DIMS))
    operB = qt.tensor(qt.qeye(N_DIMS), qt.destroy(N_DIMS))
    # Define the initial state
    rho0 = qt.tensor(qt.fock_dm(N_DIMS, 0), qt.fock_dm(N_DIMS, 0))
    # Define the times to integrate at, remember that we need at least 10^-3 precision
    tlist = np.linspace(0, G * 100, 10000)

    F = [0.1 * W_0, 0, 0.1 * W_0]
    N_B = [1, 1, 0]
    GAMMA = [0.05 * W_0, W_0]
    FILENAME = ["both_underdamped", "only_T_underdamped", "only_F_underdamped"]

    for i in range(3):
        compute_energy_ergotropy(
            H_AB(G, operA, operB) + delta_HA_t0(F[i], operA), H_B, rho0, tlist,
            [D_A(N_B[i], GAMMA[0], operA),
             D_AD(N_B[i], GAMMA[0], operA)], W_0, FILENAME[i])

    FILENAME = ["both_overdamped", "only_T_overdamped", "only_F_overdamped"]

    for i in range(3):
        compute_energy_ergotropy(
            H_AB(G, operA, operB) + delta_HA_t0(F[i], operA), H_B, rho0, tlist,
            [D_A(N_B[i], GAMMA[1], operA),
             D_AD(N_B[i], GAMMA[1], operA)], W_0, FILENAME[i])


def two_qubits():
    # Define constants of the simulation
    W_0 = 5.
    G = 0.2 * W_0
    # Define the Hamiltonian of system B
    H_B = W_0 * 0.5 * (qt.sigmaz() + 1)
    # Define the operators that acts on A and B
    sigma_am = qt.tensor(qt.sigmam(), qt.qeye(2))
    sigma_bm = qt.tensor(qt.qeye(2), qt.sigmam())
    # Define the initial state
    rho0 = qt.tensor(qt.fock_dm(2, 1), qt.fock_dm(2, 1))
    # Define the times to integrate at, remember that we need at least 10^-3 precision
    tlist = np.linspace(0, G * 100, 10000)

    F = [0.05 * W_0, 0.2 * W_0, W_0]
    N_B = 0.
    GAMMA = [0.05 * W_0, W_0]
    FILENAME = ["F_005_underdamped", "F_02_underdamped", "F_1_underdamped"]

    for i in range(3):
        compute_energy_ergotropy(
            H_AB(G, sigma_am, sigma_bm) + delta_HA_t0(F[i], sigma_am), H_B,
            rho0, tlist,
            [D_A(N_B, GAMMA[0], sigma_am),
             D_AD(N_B, GAMMA[0], sigma_am)], W_0, FILENAME[i])

    FILENAME = ["F_005_overdamped", "F_02_overdamped", "F_1_overdamped"]

    for i in range(3):
        compute_energy_ergotropy(
            H_AB(G, sigma_am, sigma_bm) + delta_HA_t0(F[i], sigma_am), H_B,
            rho0, tlist,
            [D_A(N_B, GAMMA[1], sigma_am),
             D_AD(N_B, GAMMA[1], sigma_am)], W_0, FILENAME[i])

    N_B = [0.1, 0.5, 1]
    FILENAME = ["N_B_01_overdamped", "N_B_05_overdamped", "N_B_1_overdamped"]

    for i in range(3):
        compute_energy_ergotropy(
            H_AB(G, sigma_am, sigma_bm) + delta_HA_t0(G, sigma_am), H_B, rho0,
            tlist, [
                D_A(N_B[i], GAMMA[1], sigma_am),
                D_AD(N_B[i], GAMMA[1], sigma_am)
            ], W_0, FILENAME[i])


def main():
    two_harmonic_oscillators()
    two_qubits()


if __name__ == "__main__":
    main()
