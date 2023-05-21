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
import lightpulse as lp
import qutip as qt
import numpy as np
import os


def semiclassical_sim(sigma_start,
                      sigma_stop,
                      sigma_step,
                      alpha=1.,
                      gamma=1.,
                      w_0=1.,
                      mu=0,
                      precision=1e-3):
    # Costants of the simulation
    w_0 *= gamma
    sigma_start *= gamma
    sigma_stop *= gamma
    sigma_step *= gamma
    # Initial state of the system
    init_state = qt.basis(2, 1)
    # Operator of the system
    sys_oper = qt.sigmam()
    # Hamiltonian of the system
    H_S = utils.qubit_H(w_0)

    output_path = "/home/pirota/master-thesis/sample/outputs/gaussian/fixed_sigma/"
    output_path += "coherent_semiclassical/" + str(int(alpha * alpha))
    for sigma in utils.range_decimal(sigma_start,
                                     sigma_stop,
                                     sigma_step,
                                     stop_inclusive=True):

        # Calculate time interval for the integration
        t_min = -4 * float(sigma)
        t_max = 4 * float(sigma)
        dt = precision / gamma
        steps = int((t_max - t_min) / dt)
        tlist = np.linspace(t_min, t_max, steps)

        # Define the correct args given the value of sigma
        args = {'mu': mu, 'sigma': float(sigma)}

        sigma_dir = '/sigma_' + str(sigma) + '/precision_' + str(precision)
        # Calculate all of the states
        result = qt.mesolve(
            lp.coherent_semiclassical_hamiltonian(sys_oper, lp.gaussian_sqrt,
                                                  args, alpha, gamma),
            init_state, tlist,
            qt.lindblad_dissipator(utils.damping_oper([sys_oper], gamma)))
        # Get only the states for the system
        rho_sys_final = [state.ptrace(1) for state in result.states]

        complete_output_path = output_path + sigma_dir
        if not os.path.exists(complete_output_path):
            os.makedirs(complete_output_path)
        out_erg_file = complete_output_path + '/ergotropy.dat'
        out_ene_file = complete_output_path + '/energy.dat'
        out_pur_file = complete_output_path + '/purity.dat'

        with (open(out_erg_file, 'w') as f1,
              open(out_ene_file, 'w') as f2,
              open(out_pur_file, 'w') as f3):
            for t, state in zip(tlist, rho_sys_final):
                f1.write(
                    str(t) + ' ' + str(utils.ergotropy(H_S, state)) + '\n')
                f2.write(
                    str(t) + ' ' + str(utils.energy(H_S, state)) + '\n')
                f3.write(
                    str(t) + ' ' + str(state.purity()) + '\n')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=
        'Simulate a pulse interacting with a qubit in a coherent state.')
    parser.add_argument('num_of_photons',
                        type=float,
                        nargs='?',
                        default=1,
                        help='need to get alpha')
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
    parser.add_argument('precision',
                        type=float,
                        nargs='?',
                        default=1e-3,
                        help='precision of the calculation')
    args = parser.parse_args()
    semiclassical_sim(sigma_start=args.sigma_start,
                      sigma_stop=args.sigma_stop,
                      sigma_step=args.sigma_step,
                      alpha=np.sqrt(args.num_of_photons),
                      precision=args.precision)
