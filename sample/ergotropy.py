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

import utils
import lightpulse as lp
import qutip as qt
import numpy as np
# import time
import os


def gaussian_population(init_state,
                        gamma=1.,
                        w_0=1.,
                        args={
                            'mu': 0,
                            'sigma': 1
                        },
                        precision=0.001):
    """
    Calculate the populations of the pulse Fock state, and of the ground and
    exicted state of the qubit as the interaction progresses
    """
    # Constants
    w_0 *= gamma
    sigma = args['sigma']
    N_U = init_state.dims[0][0]
    N_S = init_state.dims[0][1]
    # Operators
    operAu = qt.tensor(qt.destroy(N_U), qt.qeye(N_S))
    operC = qt.tensor(qt.qeye(N_U), qt.sigmam())

    # Calculate time interval for the integration
    t_min = -4 * float(sigma)
    t_max = 4 * float(sigma)
    dt = precision / gamma
    steps = int((t_max - t_min) / dt)
    tlist = np.linspace(t_min, t_max, steps)

    # Output states
    input_pulse_state = qt.tensor(qt.fock_dm(N_U, 1), qt.qeye(N_S))
    gs_atom_state = qt.tensor(qt.qeye(N_U), qt.ket2dm(qt.basis(N_S, 1)))
    excited_atom_state = qt.tensor(qt.qeye(N_U), qt.ket2dm(qt.basis(N_S, 0)))

    # Run the simulation
    result = qt.mesolve(
        lp.gaussian_total_H_t([operAu, operC], _gamma=gamma,
                              _args=args), init_state, tlist,
        qt.lindblad_dissipator(
            lp.gaussian_total_damping_oper_t(operAu,
                                             operC,
                                             _gamma=gamma,
                                             _args=args)),
        [input_pulse_state, gs_atom_state, excited_atom_state])

    np.savetxt('./outputs/ergotropy/input_one_photon_' + str(sigma) + '.dat',
               result.expect[0])
    np.savetxt('./outputs/ergotropy/gs_atom_' + str(sigma) + '.dat',
               result.expect[1])
    np.savetxt('./outputs/ergotropy/excited_atom_' + str(sigma) + '.dat',
               result.expect[2])


def gaussian_ergotropy(init_state,
                       sigma_start,
                       sigma_stop,
                       output_path,
                       gamma=1.,
                       w_0=1.,
                       mu=0,
                       precision=0.001):
    """
    The goal is to calculate the ergotropy as a function of sigma,
    given an interaction with a Gaussian pulse.
    """
    # Constants of the simulation
    w_0 *= gamma
    sigma_start *= gamma
    sigma_stop *= gamma
    N_U = init_state.dims[0][0]
    N_S = init_state.dims[0][1]

    # Operators
    operAu = qt.tensor(qt.destroy(N_U), qt.qeye(N_S))
    operC = qt.tensor(qt.qeye(N_U), qt.sigmam())
    # Hamiltonian of the system
    H_S = utils.qubit_H(w_0)

    # Output lists
    # max_erg_S, max_ene_S, max_pow_S = [], [], []
    out_erg_file = output_path + '/ergotropy_' + str(sigma_start) + '_' + str(
        sigma_stop) + '.dat'
    out_ene_file = output_path + '/energy_' + str(sigma_start) + '_' + str(
        sigma_stop) + '.dat'
    out_pow_file = output_path + '/power_' + str(sigma_start) + '_' + str(
        sigma_stop) + '.dat'
    for file in [out_erg_file, out_ene_file, out_pow_file]:
        if os.path.exists(file):
            os.remove(file)

    # Run the simulation
    for sigma in utils.range_decimal(sigma_start,
                                     sigma_stop,
                                     0.1,
                                     stop_inclusive=True):

        # print(float(sigma))
        # Calculate time interval for the integration
        t_min = -4 * float(sigma)
        t_max = 4 * float(sigma)
        dt = 0.001 / gamma
        steps = int((t_max - t_min) / dt)
        tlist = np.linspace(t_min, t_max, steps)

        # Define the correct args given the value of sigma
        args = {'mu': mu, 'sigma': float(sigma)}

        # Calculate all of the states
        # t0 = time.time()
        result = qt.mesolve(
            lp.gaussian_total_H_t([operAu, operC], _gamma=gamma, _args=args),
            init_state, tlist,
            qt.lindblad_dissipator(
                lp.gaussian_total_damping_oper_t(operAu,
                                                 operC,
                                                 _gamma=gamma,
                                                 _args=args)))
        # t1 = time.time()
        # print("mesolve running time: ", t1 - t0)
        # Get only the states for the system
        rho_S = [result.states[i].ptrace(1) for i in range(len(result.states))]

        # Calcate ergotropy, energy and power and append to output lists
        erg_S, ene_S, pow_S = [], [], []
        # t0 = time.time()
        for i in range(len(rho_S)):
            erg_S.append(utils.ergotropy(H_S, rho_S[i]))
            ene_S.append(utils.energy(H_S, rho_S[i]))
            pow_S.append(utils.power(erg_S[i], tlist[i], erg_S[0], tlist[0]))
        # t1 = time.time()
        # print("calculating quantities running time: ", t1 - t0)
        # Save the results to file
        with open(out_erg_file, 'a') as f:
            f.write(str(max(erg_S)) + '\n')
        with open(out_ene_file, 'a') as f:
            f.write(str(max(ene_S)) + '\n')
        with open(out_pow_file, 'a') as f:
            f.write(str(max(pow_S)) + '\n')
    # np.savetxt(
    #     output_path + '/ergotropy_' + str(sigma_start) + '_' +
    #     str(sigma_stop) + '.dat', np.array(max_erg_S))
    # np.savetxt(
    #     output_path + '/energy_' + str(sigma_start) + '_' + str(sigma_stop) +
    #     '.dat', np.array(max_ene_S))
    # np.savetxt(
    #     output_path + '/power_' + str(sigma_start) + '_' + str(sigma_stop) +
    #     '.dat', np.array(max_pow_S))


def main(pulse_state, mean_num_photons, sigma_start, sigma_stop):
    # gaussian_population(rho0)
    output_path = "/home/pirota/master-thesis/sample/outputs/ergotropy/max_gaussian/"
    N_S = 2
    if pulse_state == 'fock':
        subdir = pulse_state + '_' + str(mean_num_photons)
        complete_out_path = output_path + subdir
        if not os.path.exists(complete_out_path):
            os.makedirs(complete_out_path)
        N_U = mean_num_photons + 1
        rho0 = qt.tensor(qt.basis(N_U, 1), qt.basis(N_S, 1))
        gaussian_ergotropy(init_state=rho0,
                           sigma_start=sigma_start,
                           sigma_stop=sigma_stop,
                           output_path=complete_out_path)
    elif pulse_state == 'squeezed':
        pass
    elif pulse_state == 'coherent':
        pass
    else:
        raise TypeError(
            "`pulse_state` must either be fock, squeezed or coherent")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Simulate a gaussian pulse interacting with a qubit.')
    parser.add_argument('pulse_state',
                        choices=['fock', 'squeezed', 'coherent'],
                        help='initial state of the pulse')
    parser.add_argument(
        'number_of_photons',
        type=int,
        help='mean value of photons in the pulse initial state')
    parser.add_argument('sigma_start',
                        type=float,
                        help='starting value of sigma')
    parser.add_argument('sigma_stop',
                        type=float,
                        help='stopping value of sigma')
    args = parser.parse_args()
    main(pulse_state=args.pulse_state,
         mean_num_photons=args.number_of_photons,
         sigma_start=args.sigma_start,
         sigma_stop=args.sigma_stop)
