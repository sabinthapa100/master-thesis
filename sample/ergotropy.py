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


def gaussian_ergotropy():
    """
    The goal is to calculate the ergotropy as a function of sigma,
    given an interaction with a Gaussian pulse.
    """
    # Constants of the simulation
    GAMMA = 1.
    W_0 = 1 * GAMMA
    N_U = 2
    N_S = 2
    MU = 0
    # SIGMA = 0.1 * GAMMA
    # ARGS = {'mu': MU, 'sigma': SIGMA}

    # Operators
    operAu = qt.tensor(qt.destroy(N_U), qt.qeye(N_S))
    operC = qt.tensor(qt.qeye(N_U), qt.sigmam())
    # Initial state, sigle photon in the pulse
    rho0 = qt.tensor(qt.basis(N_U, 1), qt.basis(N_S, 1))
    H_S = utils.qubit_H(W_0)
    # Define the times to integrate at

    # Run the simulation
    # for SIGMA in utils.range_decimal(0.1 * GAMMA, 10 * GAMMA, 0.1, stop_inclusive=True):
    SIGMA = 1.
    ARGS = {'mu': MU, 'sigma': float(SIGMA)}

    t_max = 4*float(SIGMA)
    t_min = -4*float(SIGMA)
    dt = 0.001/GAMMA
    steps = int((t_max - t_min) / dt)
    tlist = np.linspace(t_min, t_max, steps)

    result = qt.mesolve(
        lp.gaussian_total_H_t([operAu, operC], _gamma=GAMMA, _args=ARGS),
        rho0, tlist,
        qt.lindblad_dissipator(
            lp.gaussian_total_damping_oper_t(operAu,
                                             operC,
                                             _gamma=GAMMA,
                                             _args=ARGS)))
    rho_A = [result.states[i].ptrace(0) for i in range(len(result.states))]
    rho_B = [result.states[i].ptrace(1) for i in range(len(result.states))]
    with (
          open('./outputs/ergotropy/input_one_photon_' + str(SIGMA) +
               '.dat', 'w') as f1,
          open('./outputs/ergotropy/excited_atom_' + str(SIGMA) +
               '.dat', 'w') as f2,
          open('./outputs/ergotropy/gs_atom_' + str(SIGMA) +
               '.dat', 'w') as f3
         ):
        input_pulse_state = qt.basis(N_U, 1)
        excited_atom_state = qt.basis(N_S, 0)
        gs_atom_state = qt.basis(N_S, 1)
        for i in range(len(result.states)):
            prob = rho_A[i].overlap(input_pulse_state)
            f1.write(str(prob) + '\n')
            prob = rho_B[i].overlap(excited_atom_state)
            f2.write(str(prob) + '\n')
            prob = rho_B[i].overlap(gs_atom_state)
            f3.write(str(prob) + '\n')

        # with open(
        #         './outputs/ergotropy/gaussian_ergotropy_' + str(SIGMA) +
        #         '.dat', 'w') as f:
        #     for i in range(len(rho_B)):
        #         f.write(str(utils.ergotropy(H_S, rho_B[i])) + '\n')
        # with (
        #         open('./outputs/ergotropy/max_gaussian_ergotropy.dat', 'a') as f1,
        #         open('./outputs/ergotropy/max_gaussian_energy.dat', 'a') as f2,
        #         # open('./outputs/ergotropy/max_gaussian_purity.dat', 'a') as f3,
        #         open('./outputs/ergotropy/max_gaussian_power.dat', 'a') as f4
        #         ):
        #     erg_B = [utils.ergotropy(H_S, rho_B[i]) for i in range(len(rho_B))]
        #     max_erg_B = max(erg_B)
        #     f1.write(str(SIGMA) + ' ' + str(max_erg_B) + '\n')
        #     ene_B = [utils.energy(H_S, rho_B[i]) for i in range(len(rho_B))]
        #     max_ene_B = max(ene_B)
        #     f2.write(str(SIGMA) + ' ' + str(max_ene_B) + '\n')
        #     # purity_B = [rho_B[i].purity() for i in range(len(rho_B))]
        #     # max_purity_B = max(purity_B)
        #     # f3.write(str(SIGMA) + ' ' + str(max_purity_B) + '\n')
        #     pow_B = [utils.power(erg_B[i], tlist[i]) for i in range(len(rho_B))]
        #     max_pow_B = max(pow_B)
        #     f4.write(str(SIGMA) + ' ' + str(max_pow_B) + '\n')

    # SIGMA = 1.
    # ARGS = {'mu': MU, 'sigma': float(SIGMA)}
    # result = qt.mesolve(
    #     lp.gaussian_total_H_t([operAu, operC], _gamma=GAMMA, _args=ARGS), rho0,
    #     tlist,
    #     qt.lindblad_dissipator(
    #         lp.gaussian_total_damping_oper_t(operAu,
    #                                          operC,
    #                                          _gamma=GAMMA,
    #                                          _args=ARGS)))
    # rho_B = [result.states[i].ptrace(1) for i in range(len(result.states))]
    # with (
    #         open('./outputs/ergotropy/gaussian_purity_1.0.dat', 'w') as f1,
    #         open('./outputs/ergotropy/gaussian_ergotropy_1.0.dat', 'w') as f2,
    #         open('./outputs/ergotropy/gaussian_energy_1.0.dat', 'w') as f3,
    #         ):
    #     for i in range(len(rho_B)):
    #         f1.write(str(tlist[i]) + ' ' + str(rho_B[i].purity()) + '\n')
    #         f2.write(str(tlist[i]) + ' ' + str(utils.ergotropy(H_S, rho_B[i])) + '\n')
    #         f3.write(str(tlist[i]) + ' ' + str(utils.energy(H_S, rho_B[i])) + '\n')


def main():
    gaussian_ergotropy()


if __name__ == "__main__":
    main()
