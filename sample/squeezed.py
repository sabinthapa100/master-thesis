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


def gaussian_fixed_sigma(init_state,
                         sigma_start,
                         sigma_stop,
                         sigma_step,
                         output_path,
                         gamma=1.,
                         w_0=1.,
                         mu=0,
                         precision=1e-3):
    # Constants of the simulation
    w_0 *= gamma
    sigma_start *= gamma
    sigma_stop *= gamma
    sigma_step *= gamma
    N_U = init_state.dims[0][0]
    N_S = init_state.dims[0][1]

    # Operators
    operAu = qt.tensor(qt.destroy(N_U), qt.qeye(N_S))
    operC = qt.tensor(qt.qeye(N_U), qt.sigmam())

    # Hamiltonian of the system
    H_S = utils.qubit_H(w_0)

    # Output states
    # psi_0 = qt.tensor(qt.ket2dm(qt.basis(N_U, 0)), qt.qeye(N_S))
    # psi_2 = qt.tensor(qt.ket2dm(qt.basis(N_U, 2)), qt.qeye(N_S))

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

        # Calculate all of the states
        result = qt.mesolve(
            lp.gaussian_total_H_t([operAu, operC], _gamma=gamma, _args=args),
            init_state, tlist,
            qt.lindblad_dissipator(
                lp.gaussian_total_damping_oper_t(operAu,
                                                 operC,
                                                 _gamma=gamma,
                                                 _args=args)))
        # Get only the states for the system
        rho_S = [state.ptrace(1) for state in result.states]

        # Save the output
        output_dir = output_path + '/sigma_' + str(sigma) + '/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # np.savetxt(output_dir + 'psi_0.dat',
        #            result.expect[0])
        # np.savetxt(output_dir + '/psi_2.dat',
        #            result.expect[1])
        out_erg_file = output_dir + 'ergotropy.dat'
        out_ene_file = output_dir + 'energy.dat'
        out_pur_file = output_dir + 'purity.dat'

        with (open(out_erg_file, 'w') as f1,
              open(out_ene_file, 'w') as f2,
              open(out_pur_file, 'w') as f3):
            for i in range(len(rho_S)):
                erg = utils.ergotropy(H_S, rho_S[i])
                f1.write(
                    str(tlist[i]) + ' ' + str(erg) + '\n')
                f2.write(
                    str(tlist[i]) + ' ' + str(utils.energy(H_S, rho_S[i])) + '\n')
                f3.write(
                    str(tlist[i]) + ' ' + str(rho_S[i].purity()) + '\n')


def main(mean_num_photons,
         sigma_start,
         sigma_stop,
         sigma_step,
         precision=1e-3):

    # gaussian_population(rho0)
    N_S = 2

    subdir = 'squeezed_' + str(int(mean_num_photons))
    # Calculate N_U such that the state is normalized
    r = np.sqrt(np.arcsinh(mean_num_photons))
    ch_r = np.cosh(r)
    th_r = np.tanh(r)
    N_U = 1
    while abs((1 / ch_r) *
              sum([(th_r * 0.5)**(2 * m) * np.math.factorial(2 * m) /
                   (np.math.factorial(m) * np.math.factorial(m))
                   for m in range(0, N_U)]) - 1) > precision:
        N_U += 1
    # print(N_U)
    rho0 = qt.tensor(
        qt.squeeze(2 * N_U, r) * qt.basis(2 * N_U, 0), qt.basis(N_S, 1))
    # check if I create the input state correctly
    # check_num_op = qt.tensor(qt.destroy(2 * N_U), qt.qeye(N_S))
    # print(qt.expect(check_num_op.dag()*check_num_op, rho0))

    output_path = "/home/pirota/master-thesis/sample/outputs/gaussian/"
    output_path += 'fixed_sigma/' + subdir + '/precision_' + str(precision)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    gaussian_fixed_sigma(init_state=rho0,
                         sigma_start=sigma_start,
                         sigma_stop=sigma_stop,
                         sigma_step=sigma_step,
                         output_path=output_path,
                         precision=precision)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Simulate a gaussian pulse interacting with a qubit.')
    parser.add_argument(
        'number_of_photons',
        type=float,
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
    main(mean_num_photons=args.number_of_photons,
         sigma_start=args.sigma_start,
         sigma_stop=args.sigma_stop,
         sigma_step=args.sigma_step,
         precision=args.precision)
