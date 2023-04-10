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
import decimal


# copied from stackoverflow
# https://stackoverflow.com/questions/39714647/can-you-use-float-numbers-in-this-for-loop
def range_decimal(start, stop, step, stop_inclusive=False):
    """ The Python range() function, using decimals.  A decimal loop_value generator.

    Note: The decimal math (addition) defines the rounding.

    If the stop is None, then:
        stop = start
        start = 0 (zero)

    If the step is 0 (zero) or None, then:
        if (stop < start) then step = -1 (minus one)
        if (stop >= start) then step = 1 (one)

    Example:
        for index in range_decimal(0, 1.0, '.1', stop_inclusive=True):
            print(index)

    :param start: The loop start value
    :param stop: The loop stop value
    :param step: The loop step value
    :param stop_inclusive: Include the stop value in the loop's yield generator: False = excluded ; True = included
    :return: The loop generator's yield increment value (decimal)
    """
    try:
        # Input argument(s) error check
        zero = decimal.Decimal('0')

        if start is None:
            start = zero

        if not isinstance(start, decimal.Decimal):
            start = decimal.Decimal(f'{start}')

        if stop is None:
            stop = start
            start = zero

        if not isinstance(stop, decimal.Decimal):
            stop = decimal.Decimal(f'{stop}')

        if step is None:
            step = decimal.Decimal('-1' if stop < start else '1')

        if not isinstance(step, decimal.Decimal):
            step = decimal.Decimal(f'{step}')

        if step == zero:
            step = decimal.Decimal('-1' if stop < start else '1')

        # Check for valid loop conditions
        if start == stop or (start < stop
                             and step < zero) or (start > stop
                                                  and step > zero):
            return  # Not valid: no loop

        # Case: increment step ( > 0 )
        if step > zero:
            while start < stop:  # Yield the decimal loop points (stop value excluded)
                yield start
                start += step

        # Case: decrement step ( < 0 )
        else:
            while start > stop:  # Yield the decimal loop points (stop value excluded)
                yield start
                start += step

        # Yield the stop value (inclusive)
        if stop_inclusive:
            yield stop

    except (ValueError, decimal.DecimalException) as ex:
        raise ValueError(f'{__name__}.range_decimal() error: {ex}')


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
    MU = 4
    # SIGMA = 0.1 * GAMMA
    # ARGS = {'mu': MU, 'sigma': SIGMA}

    # Operators
    operAu = qt.tensor(qt.destroy(N_U), qt.qeye(N_S))
    operC = qt.tensor(qt.qeye(N_U), qt.sigmam())
    # Initial state, sigle photon in the pulse
    rho0 = qt.tensor(qt.basis(N_U, 1), qt.basis(N_S, 1))
    H_S = utils.qubit_H(W_0)
    # Define the times to integrate at
    tlist = np.linspace(0, 60, 10000)

    # Run the simulation
    # for SIGMA in range_decimal(0.1 * GAMMA,
    #                            10 * GAMMA,
    #                            0.1,
    #                            stop_inclusive=True):
    #     ARGS = {'mu': MU, 'sigma': float(SIGMA)}
    #
    #     result = qt.mesolve(
    #         lp.gaussian_total_H_t([operAu, operC], _gamma=GAMMA, _args=ARGS),
    #         rho0, tlist,
    #         qt.lindblad_dissipator(
    #             lp.gaussian_total_damping_oper_t(operAu,
    #                                              operC,
    #                                              _gamma=GAMMA,
    #                                              _args=ARGS)))
    #     # rho_A = [result.states[i].ptrace(0) for i in range(len(result.states))]
    #     rho_B = [result.states[i].ptrace(1) for i in range(len(result.states))]
    #     # with (
    #     #       open('./outputs/ergotropy/input_one_photon_' + str(SIGMA) +
    #     #            '.dat', 'w') as f1,
    #     #       open('./outputs/ergotropy/excited_atom_' + str(SIGMA) +
    #     #            '.dat', 'w') as f2,
    #     #       open('./outputs/ergotropy/gs_atom_' + str(SIGMA) +
    #     #            '.dat', 'w') as f3
    #     #      ):
    #     #     input_pulse_state = qt.basis(N_U, 1)
    #     #     excited_atom_state = qt.basis(N_S, 0)
    #     #     gs_atom_state = qt.basis(N_S, 1)
    #     #     for i in range(len(result.states)):
    #     #         prob = np.abs(rho_A[i].overlap(input_pulse_state))**2
    #     #         f1.write(str(prob) + '\n')
    #     #         prob = np.abs(rho_B[i].overlap(excited_atom_state))**2
    #     #         f2.write(str(prob) + '\n')
    #     #         prob = np.abs(rho_B[i].overlap(gs_atom_state))**2
    #     #         f3.write(str(prob) + '\n')
    #
    #     # with open(
    #     #         './outputs/ergotropy/gaussian_ergotropy_' + str(SIGMA) +
    #     #         '.dat', 'w') as f:
    #     #     for i in range(len(rho_B)):
    #     #         f.write(str(utils.ergotropy(H_S, rho_B[i])) + '\n')
    #     with (
    #             open('./outputs/ergotropy/max_gaussian_ergotropy.dat', 'a') as f1,
    #             open('./outputs/ergotropy/max_gaussian_energy.dat', 'a') as f2,
    #             open('./outputs/ergotropy/max_gaussian_purity.dat', 'a') as f3,
    #             open('./outputs/ergotropy/max_gaussian_power.dat', 'a') as f4
    #             ):
    #         erg_B = [utils.ergotropy(H_S, rho_B[i]) for i in range(len(rho_B))]
    #         max_erg_B = max(erg_B)
    #         f1.write(str(SIGMA) + ' ' + str(max_erg_B) + '\n')
    #         ene_B = [utils.energy(H_S, rho_B[i]) for i in range(len(rho_B))]
    #         max_ene_B = max(ene_B)
    #         f2.write(str(SIGMA) + ' ' + str(max_ene_B) + '\n')
    #         purity_B = [rho_B[i].purity() for i in range(len(rho_B))]
    #         max_purity_B = max(purity_B)
    #         f3.write(str(SIGMA) + ' ' + str(max_purity_B) + '\n')
    #         # pow_B = [erg_B[i]/float(tlist[i]) for i in range(len(rho_B))]
    #         # max_pow_B = max(pow_B)
    #         # f4.write(str(SIGMA) + ' ' + str(max_pow_B) + '\n')

    SIGMA = 1.
    ARGS = {'mu': MU, 'sigma': float(SIGMA)}
    result = qt.mesolve(
        lp.gaussian_total_H_t([operAu, operC], _gamma=GAMMA, _args=ARGS), rho0,
        tlist,
        qt.lindblad_dissipator(
            lp.gaussian_total_damping_oper_t(operAu,
                                             operC,
                                             _gamma=GAMMA,
                                             _args=ARGS)))
    rho_B = [result.states[i].ptrace(1) for i in range(len(result.states))]
    with open("./outputs/ergotropy/gaussian_purity.dat", 'w') as f:
        for i in range(len(rho_B)):
            f.write(str(tlist[i]) + ' ' + str(rho_B[i].purity()) + '\n')


def main():
    gaussian_ergotropy()


if __name__ == "__main__":
    main()
