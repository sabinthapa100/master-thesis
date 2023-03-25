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

import qutip as qt


def ergotropy(_H, _rho):
    """ 
    Function that calculates the ergotropy of state `_rho` on an Hamiltonian `_H`
    """

    # First, we calculate the energy expectation value
    _energy = qt.expect(_H, _rho)
    # Then, compute the eigenvalues and sort them in the wanted order
    _rho_eigenvalues = _rho.eigenenergies(sort='high')
    _H_eigenvalues = _H.eigenenergies(sort='low')

    # and then compute the second term of the expression
    _energy_passive = 0.
    for i in range(len(_rho_eigenvalues)):
        _energy_passive += _rho_eigenvalues[i] * _H_eigenvalues[i]

    return _energy - _energy_passive
