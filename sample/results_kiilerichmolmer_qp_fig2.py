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

import numpy as np
import qutip as qt
import math
import scipy as sp


def gaussian_sqrt(_t, _mu=0., _sigma=1.):
    """
    Returns the gaussian function, which is our pulse shape
    """
    x = float(_t - _mu) / _sigma
    return math.sqrt(math.exp(-x * x / 2.) / math.sqrt(2 * np.pi) / _sigma)


def g_u_gaussian(_t, _args):
    """
    Time dependent coupling with the cavity
    """
    mu = _args['mu']
    sigma = _args['sigma']
    x = (_t - mu) / (math.sqrt(2) * sigma)
    denominator = np.sqrt(1 - 0.5 * (1 + sp.special.erf(x)))
    return gaussian_sqrt(_t, mu, sigma) / denominator


def oscillator_H(_oper, _w_0=1.):
    """
    Returns the hamiltonian of an harmonic oscillator 
    """
    return _w_0 * _oper.dag() * _oper


def only_input_inter_H(_operA, _operC, _gamma=1.):
    """
    Calculates the interaction Hamilotonian, with only the input pulse
    """
    return 1j * math.sqrt(_gamma) / 2. * _operA.dag() * _operC


def conj_only_input_inter_H(_operA, _operC, _gamma=1.):
    """
    Calculates the conjucate of the interaction Hamilotonian, with only the input pulse
    They are separate because in principle the have two different time dependencies
    """
    return -1j * math.sqrt(np.conj(_gamma)) / 2. * _operA * _operC.dag()


def total_H_t(_operA, _operC, _w_0=1., _gamma=1., _args={'mu': 1, 'sigma': 1}):
    """
    Returns the QobjEvo with the right time dependencies
    """
    return qt.QobjEvo([
        oscillator_H(_operC, _w_0),
        [only_input_inter_H(_operA, _operC, _gamma), g_u_gaussian],
        [conj_only_input_inter_H(_operA, _operC, _gamma), g_u_gaussian]
    ],
                      args=_args)


def damping_oper(_operC, _gamma=1.):
    """
    Time independet damping operator
    """
    return np.sqrt(_gamma) * _operC


def total_damping_oper_t(_operA,
                         _operC,
                         _gamma=1.,
                         _args={
                             'mu': 1,
                             'sigma': 1
                         }):
    """
    Returns the complete damping operator, with the time dependent part
    """
    return qt.QobjEvo([damping_oper(_operC, _gamma), [_operA, g_u_gaussian]],
                      args=_args)


def scattering_gaussian_mode():
    """
    Runs the simulation for the scattering of a Gaussian pulse with single photon on an empy cavity
    """
    # Constants of the simulation

    # The Halmitonian in the master equation is in the interaction representation w.r.t H_S (system hamiltonian)
    # this means that H_S plays no role in the dymanic, and the easiest way to not have its contribution, is to
    # take W_0 = 0 BEWARE: this has no real physical meaning, it is just a shortcut to not rewrite several lines of code
    W_0 = 0.

    GAMMA = 1.
    N_DIMS = 2
    MU = 4
    SIGMA = 1
    ARGS = {'mu': MU, 'sigma': SIGMA}
    # Operators
    operA = qt.tensor(qt.destroy(N_DIMS), qt.qeye(N_DIMS))
    operC = qt.tensor(qt.qeye(N_DIMS), qt.destroy(N_DIMS))
    # Initial state, sigle photon in the pulse
    rho0 = qt.tensor(qt.fock_dm(N_DIMS, 1), qt.fock_dm(N_DIMS, 0))
    # Define the times to integrate at
    tlist = np.linspace(0, MU + 4 * SIGMA, 10000)
    # Run the simulation
    result = qt.mesolve(
        total_H_t(operA, operC, W_0, GAMMA, ARGS), rho0, tlist,
        qt.lindblad_dissipator(total_damping_oper_t(operA, operC, GAMMA,
                                                    ARGS)),
        [operA.dag() * operA, operC.dag() * operC])

    # Write the expectation values to file
    with open('./outputs/results_kiilerichmolmer_qp/adaga_expect_values.dat',
              'w') as f:
        for i in range(len(result.expect[0])):
            f.write(str(result.expect[0][i]) + '\n')

    with open('./outputs/results_kiilerichmolmer_qp/cdagc_expect_values.dat',
              'w') as f:
        for i in range(len(result.expect[1])):
            f.write(str(result.expect[1][i]) + '\n')


def main():
    scattering_gaussian_mode()


if __name__ == "__main__":
    main()
