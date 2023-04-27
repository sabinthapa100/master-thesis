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

    np.savetxt(
        './outputs/gaussian/population/input_one_photon_' + str(sigma) +
        '.dat', result.expect[0])
    np.savetxt('./outputs/gaussian/population/gs_atom_' + str(sigma) + '.dat',
               result.expect[1])
    np.savetxt(
        './outputs/gaussian/population/excited_atom_' + str(sigma) + '.dat',
        result.expect[2])


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

    for sigma in utils.range_decimal(sigma_start,
                                     sigma_stop,
                                     sigma_step,
                                     stop_inclusive=True):

        # print(float(sigma))
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
        rho_S = [result.states[i].ptrace(1) for i in range(len(result.states))]
        complete_output_path = output_path + '/sigma_' + str(sigma)
        if not os.path.exists(complete_output_path):
            os.makedirs(complete_output_path)
        out_erg_file = complete_output_path + '/ergotropy_' + str(sigma) + '.dat'
        out_ene_file = complete_output_path + '/energy_' + str(sigma) + '.dat'
        # out_pow_file = complete_output_path + '/power_' + str(sigma) + '.dat'
        out_pur_file = complete_output_path + '/purity_' + str(sigma) + '.dat'

        with (open(out_erg_file, 'w') as f1,
              open(out_ene_file, 'w') as f2,
              # open(out_pow_file, 'w') as f3,
              open(out_pur_file, 'w') as f4):
            # erg0 = utils.ergotropy(H_S, rho_S[0])
            for i in range(len(rho_S)):
                erg = utils.ergotropy(H_S, rho_S[i])
                f1.write(
                    str(tlist[i]) + ' ' + str(erg) + '\n')
                f2.write(
                    str(tlist[i]) + ' ' + str(utils.energy(H_S, rho_S[i])) + '\n')
                # f3.write(
                #     str(tlist[i]) + ' ' + str(utils.power(erg, tlist[i],
                #                                           erg0, tlist[0])) + '\n')
                f4.write(
                    str(tlist[i]) + ' ' + str(rho_S[i].purity()) + '\n')


def gaussian_max(init_state,
                 sigma_start,
                 sigma_stop,
                 sigma_step,
                 output_path,
                 gamma=1.,
                 w_0=1.,
                 mu=0,
                 precision=1e-3):
    """
    The goal is to calculate the maximum of energy, ergotropy and
    power as a function of sigma, given an interaction with a Gaussian pulse.
    """
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
                                     sigma_step,
                                     stop_inclusive=True):

        # print(float(sigma))
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
        rho_S = [result.states[i].ptrace(1) for i in range(len(result.states))]

        # Calcate ergotropy, energy and power and append to output lists
        erg_S, ene_S, pow_S = [], [], []
        for i in range(len(rho_S)):
            erg_S.append(utils.ergotropy(H_S, rho_S[i]))
            ene_S.append(utils.energy(H_S, rho_S[i]))
            pow_S.append(utils.power(erg_S[i], tlist[i], erg_S[0], tlist[0]))
        # Save the results to file
        with open(out_erg_file, 'a') as f:
            f.write(str(sigma) + ' ' + str(max(erg_S)) + '\n')
        with open(out_ene_file, 'a') as f:
            f.write(str(sigma) + ' ' + str(max(ene_S)) + '\n')
        with open(out_pow_file, 'a') as f:
            f.write(str(sigma) + ' ' + str(max(pow_S)) + '\n')


def main(pulse_state,
         mean_num_photons,
         sigma_start,
         sigma_stop,
         sigma_step,
         precision=1e-3,
         max_flag=True):
    # gaussian_population(rho0)
    N_S = 2

    # This is under the assumption that if I want a small sigma_step,
    # I'm looking at a small interval of sigma, so the simulation is rather
    # short, and I can afford to have a smaller dt for the integration
    # of the master equations
    # if sigma_step < 0.1:
    #     precision *= sigma_step

    if pulse_state == 'fock':
        subdir = pulse_state + '_' + str(mean_num_photons)
        N_U = mean_num_photons + 1
        rho0 = qt.tensor(qt.basis(N_U, mean_num_photons), qt.basis(N_S, 1))
    elif pulse_state == 'squeezed':
        subdir = pulse_state + '_' + str(mean_num_photons)
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
        rho0 = qt.tensor(
            qt.squeeze(N_U, r) * qt.basis(N_U, 0), qt.basis(N_S, 1))
    elif pulse_state == 'coherent':
        subdir = pulse_state + '_' + str(mean_num_photons)
        # Calculate N_U such that the state is normalized
        alpha = np.sqrt(mean_num_photons)
        factor = np.exp(-(alpha * alpha))
        N_U = 1
        while abs(factor * sum(
            [alpha**(2 * m) / np.math.factorial(m)
             for m in range(0, N_U)]) - 1) > precision:
            N_U += 1
        rho0 = qt.tensor(qt.coherent(N_U, alpha), qt.basis(N_S, 1))
    else:
        raise TypeError(
            "`pulse_state` must either be fock, squeezed or coherent")

    output_path = "/home/pirota/master-thesis/sample/outputs/gaussian/"
    if max_flag:
        output_path += 'max/' + subdir + '/precision_' + str(precision)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        gaussian_max(init_state=rho0,
                     sigma_start=sigma_start,
                     sigma_stop=sigma_stop,
                     sigma_step=sigma_step,
                     output_path=output_path,
                     precision=precision)
    else:
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
    parser.add_argument('pulse_state',
                        choices=['fock', 'squeezed', 'coherent'],
                        help='initial state of the pulse')
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
    parser.add_argument(
        '--max',
        action=argparse.BooleanOptionalAction,
        help='if True calculate the maximum as a function of sigma,'
        ' if False outputs the quantities for fixed sigma')
    args = parser.parse_args()
    main(pulse_state=args.pulse_state,
         mean_num_photons=args.number_of_photons,
         sigma_start=args.sigma_start,
         sigma_stop=args.sigma_stop,
         sigma_step=args.sigma_step,
         precision=args.precision,
         max_flag=args.max)
