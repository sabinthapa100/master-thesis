The GPLv3 License (GPLv3)

Copyright (c) 2023 Simone Pirota

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

import myfunctions as mf 
import numpy as np 
import cmath

def compute_energy_ergotropy(_H_evolve, _H_B, _rho0, _times, _c_ops, _w_zero, _filename):
    """ 
    Function that, given an Hamiltonian `_H_evolve` to evolve `_rho0`, outputs to `_filename` 
    energy and ergotropy of the evolution 
    """
    # Resolve the master equation with qutip mesolve
    result = qt.mesolve(_H_evolve, _rho0, _times, _c_ops)
    # Get the partial trace of the state to obtain rho_B
    rho_B = [ result.states[i].ptrace(1) for i in range(len(result.states)) ]

    # write to file the energy results
    with open('energy_' + _filename, 'w') as f:
        for i in range( len(rho_B) ):
            f.write( qt.expect(_H_B, rho_B[i]) / w_zero )

    # write to file the ergotropy result
    with open('ergotropy_' + _filename, 'w') as f:
        for i in range( len(rho_B) ):
            f.write( mf.ergotropy(_H_B, rho_B[i]) / w_zero )

def delta_HA_t0(_F, _oper):
    """
    Returns the following qobj, which represents the coherent driving of system A
    """
    return _F * (_oper.dag() + _oper)

def H_AB(_g, _operA, _operB):
    """
    Return the following qobj, which represents the coupling between system A and B 
    """
    return _g * (_operA * _operB.dag() + _operA.dag() * _operB)

def D_A(_N_B, _gamma, _oper):
    """
    Return the following qobj, which represents the dissipation of energy into the environment
    """
    return _gamma * (_N_B + 1) * qt.lindblad_dissipator(_oper)

def D_AD(_N_B, _gamma, _oper):
    """
    Return the following qobj, which represents the dissipation of energy into the environment
    """
    return _gamma * _N_B * qt.lindblad_dissipator(_oper.dag())
