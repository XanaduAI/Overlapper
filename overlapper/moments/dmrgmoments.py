# Overlapper: Prepare and evaluate wavefunctions for quantum
# algorithms using computational chemistry techniques
# Copyright 2024 Xanadu Quantum Technologies Inc.
#
# Author: Stepan Fomichev <stepan.fomichev@xanadu.ai>
#
# This file is part of Overlapper.
#
# Overlapper is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version.
#
# Overlapper is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Overlapper. If not, see <https://www.gnu.org/licenses/>.

import numpy as np

def dmrg_moments(driver, mps, mpo, bra_bond_dims, kmax, verbose=0):
    r"""Use Block2's DMRGDriver to compute moments of a given state.
    Unlike using FCI to calculate moments, DMRG is capable of 
    operating in a much larger Hilbert spaces. However, it comes at 
    the cost of the moments being approximate -- becoming more precise 
    the larger the bond dimension used to evaluate them.

    To compute a moment <mps|H^n|mps>, the driver's `multiply()` method 
    is first used to calculate |ket1> = H|mps>, then the first moment is obtained 
    by calculating the expectation value `driver.expectation(ket1, mpo, mps)`. 
    Each subsequent moment is obtained from the previous one recursively by 
    continually applying H to the state.

    Args:
        driver (object): Block2's DMRGDriver object.
        mps (object): Block2's MPS object representing the state whose Hamiltonian moments are to be calculated.
        mpo (object): Block2's MPO object for the Hamiltonian.
        bra_bond_dims (list(int)): List of bond dimensions to be used to store the states H|mps>, H^2|mps>, ..., H^kmax|mps>. Should be increasing progressively with each application of H in order to be accurate.
        kmax (int): Maximum moment to calculate.
        verbose (int): Integer specifying the verbosity level.

    Returns:
        moments (array[float]): List of moments of the Hamiltonian MPO relative to the state MPS.
    """

    if bra_bond_dims[0] < mps.info.bond_dim:
        raise ValueError("Bond dimensions of states H^k|psi> must be greater than bond dim of input |psi>.")

    list_of_kets = []
    list_of_kets.append(mps.deep_copy("0ket"))
    moments = np.zeros(kmax)
    impo = driver.get_identity_mpo()
    for ii, k in enumerate(list(range(kmax))):
        list_of_kets.append( list_of_kets[ii].deep_copy(f"{ii+1}ket") )
        current_bd = list_of_kets[ii].info.bond_dim
        norm = driver.multiply(list_of_kets[ii+1], mpo, list_of_kets[ii], bond_dims=[current_bd],
                               bra_bond_dims=[bra_bond_dims[ii]], n_sweeps=100, tol=1E-8, iprint=verbose)
        moments[ii] = driver.expectation(list_of_kets[0], impo, list_of_kets[ii+1])

    return moments