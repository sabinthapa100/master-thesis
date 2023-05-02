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

import lightpulse as lp
import utils
import qutip as qt
import numpy as np
import os


def delta_HA_t0(_F, _oper):
    """
    Returns the following qobj,
    which represents the coherent driving of system A
    """
    return _F * (_oper.dag() + _oper)


def H_AB(_g, _operA, _operB):
    """
    Return the following qobj,
    which represents the coupling between system A and B
    """
    return _g * (_operA * _operB.dag() + _operA.dag() * _operB)


def D_A(_N_B, _gamma, _oper):
    """
    Return the following qobj,
    which represents the dissipation of energy into the environment
    """
    return _gamma * (_N_B + 1) * qt.lindblad_dissipator(_oper)


def D_AD(_N_B, _gamma, _oper):
    """
    Return the following qobj,
    which represents the dissipation of energy into the environment
    """
    return _gamma * _N_B * qt.lindblad_dissipator(_oper.dag())


def two_qubits(sigma_start,
               sigma_stop,
               sigma_step,
               output_path,
               w_0=1.,
               gaussian_gamma=1.,
               alpha_coherent=1.,
               precision=1e-3):

    # Define simulation constants
    g = 0.2 * w_0
    # Hamiltonian of the system B
    H_B = utils.qubit_H(w_0)
    # Define the operators that acts on A and B
    sigma_am = qt.tensor(qt.sigmam(), qt.qeye(2))
    sigma_bm = qt.tensor(qt.qeye(2), qt.sigmam())
    # Define the initial state
    rho0 = qt.tensor(qt.basis(2, 1), qt.basis(2, 1))

    for sigma in utils.range_decimal(sigma_start,
                                     sigma_stop,
                                     sigma_step,
                                     stop_inclusive=True):

        t_min = -4 * float(sigma)
        t_max = 4 * float(sigma)
        dt = precision / gaussian_gamma
        steps = int((t_max - t_min) / dt)
        tlist = np.linspace(t_min, t_max, steps)

        u_max = max(
            [lp.gaussian_sqrt(tlist[i], 0, sigma) for i in range(tlist)])

        F = 1j * np.sqrt(gaussian_gamma) * u_max * alpha_coherent

        N_B = 0.
        F_GAMMA = [0.05 * w_0, w_0]

        complete_output_path = output_path + '/sigma_' + str(sigma)
        if not os.path.exists(complete_output_path):
            os.makedirs(complete_output_path)
        filename = ["underdamped", "overdamped"]
        tlist = np.linspace(0, g * 100, 10000)

        for gamma, file in zip(F_GAMMA, filename):
            result = qt.mesolve(
                H_AB(g, sigma_am, sigma_bm) + delta_HA_t0(F, sigma_am), rho0,
                tlist, [D_A(N_B, gamma, sigma_am),
                        D_AD(N_B, gamma, sigma_am)])

            rho_B = [
                result.states[i].ptrace(1) for i in range(len(result.states))
            ]

            with (open(complete_output_path + '/erg_' + file + '.dat', 'w') as f1,
                  open(complete_output_path + '/ene_' + file + '.dat', 'w') as f2):
                for i in range(len(rho_B)):
                    f1.write(
                        str(tlist[i]) + ' ' +
                        str(utils.ergotropy(H_B, rho_B[i])) + '\n')
                    f2.write(
                        str(tlist[i]) + ' ' +
                        str(utils.energy(H_B, rho_B[i])) + '\n')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Simulate the Farina setup')
    parser.add_argument(
        'number_of_photons',
        type=int,
        nargs='?',
        default=1,
        help='mean value of photons in the pulse initial state')
    parser.add_argument('sigma_start',
                        type=float,
                        nargs='?',
                        default=0.1,
                        help='starting value of sigma')
    parser.add_argument('sigma_stop',
                        type=float,
                        nargs='?',
                        default=10,
                        help='stopping value of sigma')
    parser.add_argument('sigma_step',
                        type=float,
                        nargs='?',
                        default=0.1,
                        help='stepping value of sigma')
    parser.add_argument(
        'precision',
        type=float,
        nargs='?',
        default=1e-3,
        help='precision of the calculation, cannot be more than 1e-3')
    args = parser.parse_args()
    output_path = "/home/pirota/master-thesis/sample/outputs/farina/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    two_qubits(args.sigma_start, args.sigma_stop, args.sigma_step, output_path,
               w_0=5., gaussian_gamma=1.,
               alpha_coherent=np.sqrt(args.number_of_photons),
               precision=args.precision)
