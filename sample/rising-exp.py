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
import os


def rising_exp_population(init_state,
                          t_start,
                          t_stop,
                          output_path,
                          gamma=1.,
                          w_0=1.,
                          precision=1e-3):
    """
    Calculate the populations of the pulse Fock state, and of the ground and
    exicted state of the qubit as the interaction progresses
    """
    # Constants
    w_0 *= gamma
    t_start *= gamma
    t_stop *= gamma
    N_U = init_state.dims[0][0]
    N_S = init_state.dims[0][1]
    # Operators
    operAu = qt.tensor(qt.destroy(N_U), qt.qeye(N_S))
    operC = qt.tensor(qt.qeye(N_U), qt.sigmam())

    # Calculate time interval for the integration
    dt = precision / gamma
    steps = int((t_stop - t_start) / dt)
    tlist = np.linspace(t_start, t_stop, steps)

    # Output states
    input_pulse_state = qt.tensor(qt.fock_dm(N_U, 1), qt.qeye(N_S))
    gs_atom_state = qt.tensor(qt.qeye(N_U), qt.ket2dm(qt.basis(N_S, 1)))
    excited_atom_state = qt.tensor(qt.qeye(N_U), qt.ket2dm(qt.basis(N_S, 0)))

    args = {
        'GAMMA': gamma,
        't0': 0,
    }

    # opts = qt.Options(nsteps=10000)
    # Run the simulation
    result = qt.mesolve(
        lp.rising_exp_total_H_t([operAu, operC], _gamma=gamma,
                                _args=args), init_state, tlist,
        qt.lindblad_dissipator(
            lp.rising_exp_total_damping_oper_t(operAu,
                                               operC,
                                               _gamma=gamma,
                                               _args=args)),
        [input_pulse_state, gs_atom_state, excited_atom_state])#, options=opts)

    np.savetxt(output_path + '/input_one_photon.dat', result.expect[0])
    np.savetxt(output_path + '/gs_atom.dat', result.expect[1])
    np.savetxt(output_path + '/excited_atom.dat', result.expect[2])


def rising_exp_sim(init_state,
                   t_start,
                   t_stop,
                   output_path,
                   gamma=1.,
                   w_0=1.,
                   precision=1e-3):
    # Constants of the simulation
    w_0 *= gamma
    t_start *= gamma
    t_stop *= gamma
    N_U = init_state.dims[0][0]
    N_S = init_state.dims[0][1]

    # Operators
    operAu = qt.tensor(qt.destroy(N_U), qt.qeye(N_S))
    operC = qt.tensor(qt.qeye(N_U), qt.sigmam())
    # Hamiltonian of the system
    H_S = utils.qubit_H(w_0)

    out_erg_file = output_path + '/ergotropy.dat'
    out_ene_file = output_path + '/energy.dat'
    out_pur_file = output_path + '/purity.dat'
    for file in [out_erg_file, out_ene_file, out_pur_file]:
        if os.path.exists(file):
            os.remove(file)

    # Run the simulation
    # Calculate time interval for the integration
    dt = precision / gamma
    steps = int((t_stop - t_start) / dt)
    tlist = np.linspace(t_start, t_stop, steps)

    # Define the correct args
    args = {
        'GAMMA': gamma,
        't0': 0,
    }

    # opts = qt.Options(nsteps=10000)
    # Calculate all of the states
    result = qt.mesolve(
        lp.rising_exp_total_H_t([operAu, operC], _gamma=gamma, _args=args),
        init_state, tlist,
        qt.lindblad_dissipator(
            lp.rising_exp_total_damping_oper_t(operAu,
                                               operC,
                                               _gamma=gamma,
                                               _args=args)))#, options=opts)
    # Get only the states for the system
    rho_S = [result.states[i].ptrace(1) for i in range(len(result.states))]

    # Write results to files
    with (open(out_erg_file, 'w') as f1,
          open(out_ene_file, 'w') as f2,
          open(out_pur_file, 'w') as f3):
        for t, rho in zip(tlist, rho_S):
            f1.write(
                str(t) + ' ' + str(utils.ergotropy(H_S, rho)) + '\n')
            f2.write(
                str(t) + ' ' + str(utils.energy(H_S, rho)) + '\n')
            f3.write(
                str(t) + ' ' + str(rho.purity()) + '\n')


def main(pulse_state,
         mean_num_photons,
         t_start,
         t_stop,
         precision=1e-3,
         pop_flag=False):
    N_S = 2

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

    output_path = "/home/pirota/master-thesis/sample/outputs/rising_exp/"
    if pop_flag:
        output_path += 'pop/' + subdir + '/precision_' + str(precision)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        rising_exp_population(init_state=rho0,
                              t_start=t_start,
                              t_stop=t_stop,
                              output_path=output_path,
                              precision=precision)
    else:
        output_path += 'time/' + subdir + '/precision_' + str(precision)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        rising_exp_sim(init_state=rho0,
                       t_start=t_start,
                       t_stop=t_stop,
                       output_path=output_path,
                       precision=precision)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Simulate a rising_exp pulse interacting with a qubit.')
    parser.add_argument('pulse_state',
                        choices=['fock', 'squeezed', 'coherent'],
                        help='initial state of the pulse')
    parser.add_argument(
        'number_of_photons',
        type=int,
        nargs='?',
        default=1,
        help='mean value of photons in the pulse initial state')
    parser.add_argument('t_start',
                        type=float,
                        nargs='?',
                        default=-100,
                        help='starting value of t')
    parser.add_argument('t_stop',
                        type=float,
                        nargs='?',
                        default=0,
                        help='stopping value of t')
    parser.add_argument(
        'precision',
        type=float,
        nargs='?',
        default=1e-3,
        help='precision of the calculation, cannot be more than 1e-3')
    parser.add_argument('--pop',
                        action=argparse.BooleanOptionalAction,
                        help='if set calculates the atom levels populations')
    args = parser.parse_args()
    main(pulse_state=args.pulse_state,
         mean_num_photons=args.number_of_photons,
         t_start=args.t_start,
         t_stop=args.t_stop,
         precision=args.precision,
         pop_flag=args.pop)
