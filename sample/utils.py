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

import math
import qutip as qt
import decimal
import numpy as np
# import time


def energy(_H, _rho):
    """
    Wrapper around qt.expect()
    """
    return qt.expect(_H, _rho)


def ergotropy(_H, _rho):
    """
    Function that calculates the ergotropy of state `_rho` on an Hamiltonian `_H`
    """

    # First, we calculate the energy expectation value
    # t0 = time.time()
    _energy = energy(_H, _rho)
    # t1 = time.time()
    # print("ergotropy: energy calc time: ", t1-t0)
    # Then, compute the eigenvalues and sort them in the wanted order
    # t0 = time.time()
    _rho_eigenvalues = _rho.eigenenergies(sort='high')
    _H_eigenvalues = _H.eigenenergies(sort='low')
    # t1 = time.time()
    # print("ergotropy: eigenenergies calc time: ", t1-t0)
    # and then compute the second term of the expression
    # t0 = time.time()
    _energy_passive = np.sum(_rho_eigenvalues * _H_eigenvalues)
    # t1 = time.time()
    # print("ergotropy: emergy_passive calc time: ", t1-t0)
    return _energy - _energy_passive


def oscillator_H():
    pass


def qubit_H(_W_0=1.):
    return 0.5 * _W_0 * (qt.sigmaz() + 1.)


def damping_oper(*_operators, _gamma=1.):
    """
    Returns the damping operator sqrt(_gamma)*oper
    """
    if not _operators:
        raise TypeError("Requires at least one input argument")
    if len(_operators) == 1 and isinstance(_operators[0], qt.Qobj):
        # this is the case when damping_oper is called on the form:
        # damping_oper([q1, q2, q3, ...])
        return math.sqrt(_gamma) * _operators[0]
    else:
        # this is the case when damping_oper  is called on the form:
        # damping_oper (q1, q2, q3, ...)
        oper_list = _operators
    if len(oper_list) != 1:
        raise TypeError("Requires exactly one operator")
    if not all([isinstance(oper, qt.Qobj) for oper in oper_list]):
        # raise error if one of the inputs is not a quantum object
        raise TypeError("The input is not a quantum object")
    return math.sqrt(_gamma) * oper_list[0]


def power(_ef, _tf, _ei=0., _ti=0.):
    """
    Return the ratio (_ef-_ei)/(_tf-_ti), which is the power
    """
    de = float(_ef - _ei)
    dt = float(_tf - _ti)
    if de == 0:
        return 0
    else:
        return de / dt


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
