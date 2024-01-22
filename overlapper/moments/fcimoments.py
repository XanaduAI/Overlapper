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

from functools import reduce
import numpy as np

from pyscf import ao2mo
from pyscf.fci.cistring import gen_linkstr_index_trilidx

from pyscf.fci.direct_uhf import absorb_h1e as absorb_h1e_uhf
from pyscf.fci.direct_uhf import contract_2e as contract_2e_uhf
from pyscf.fci.direct_spin1 import absorb_h1e as absorb_h1e_rhf
from pyscf.fci.direct_spin1 import contract_2e as contract_2e_rhf

from overlapper.utils.utils import get_mol_attrs, fcidict_to_fcivec

def _rfci_hamiltonian(hf):
    r"""Builder of the FCI Hamiltonian, specifically for the RHF case. 
    UHF builder is available separately as they require drastically 
    different management of the MO coefficients.

    Args:
        hf (object): PySCF restricted Hartree-Fock Solver object.

    Returns:
        link_index_a, link_index_b (tuple): Objects that help contract 
            the many-body wavefunction and Hamiltonian.
        h2e (array): A complete Hamiltonian matrix for the system. 
    """
    norb, nelec_a, nelec_b = get_mol_attrs(hf.mol)

    h1e = hf.mo_coeff.T.dot(hf.get_hcore()).dot(hf.mo_coeff)
    eri = ao2mo.kernel(hf.mol, hf.mo_coeff, aosym='s1', intor='int2e')

    link_index_a = gen_linkstr_index_trilidx(range(norb), nelec_a)
    link_index_b = gen_linkstr_index_trilidx(range(norb), nelec_b)

    h2e = absorb_h1e_rhf(h1e, eri, norb, (nelec_a, nelec_b), 0.5)

    return (link_index_a, link_index_b), h2e


def _contract_rfci_hamiltonian(mol, fcivec, link_index, h2e):
    r"""Take an FCI-format wavefunction (2D-array) and contract 
    it against the FCI Hamiltonian (restricted HF version).

    Args:
        mol (object): PySCF's Molecule object.
        fcivec (array): FCI wavefunction in PySCF's 2D-array format.
        link_index (object): Objects that help contract 
            the many-body wavefunction and Hamiltonian
        h2e (array): A complete Hamiltonian matrix for the system. 

    Returns:
        ci (array): The wavefunction post-contraction 
            (i.e. post-multiplication by the Hamiltonian). 
    """
    norb, nelec_a, nelec_b = get_mol_attrs(mol)
    ci = contract_2e_rhf(h2e, fcivec, norb, (nelec_a, nelec_b), link_index)
    return ci


def rfci_moments(hf, kmax, fcidict):
    r"""Return the moments of the FCI wavefunction (restricted HF case)
    by computing them directly through contracting the FCI wavefunction
    with the FCI Hamiltonian. The higher-order moments are obtained 
    recursively from the lower-order moments by successive contractions
    of the wavefunction with the Hamiltonian. 

    Args:
        hf (object): PySCF restricted Hartree-Fock Solver object.
        kmax (int): Largest moment to be calculated. 
        fcidict (dict): A wavefunction in Overlapper's sparse 
            dictionary format.

    Returns:
        moments (array): Array of the computed Hamiltonian moments of the 
            provided wavefunction `fcidict`. 
    """

    moments = np.zeros((kmax))

    fcivec = fcidict_to_fcivec(fcidict, hf.mol)

    # construct the FCI Hamiltonian
    link_index, h2e = _rfci_hamiltonian(hf)

    # compute the moments
    for ii in range(kmax):
        if ii == 0:
            ci1 = _contract_rfci_hamiltonian(hf.mol, fcivec, link_index, h2e)
            moments[ii] = np.dot(fcivec.reshape(-1), ci1.reshape(-1))
        else:
            ci1 = _contract_rfci_hamiltonian(hf.mol, ci1, link_index, h2e)
            moments[ii] = np.dot(fcivec.reshape(-1), ci1.reshape(-1))

    return moments


def _ufci_hamiltonian(hf):
    r"""Builder of the FCI Hamiltonian, specifically for the UHF case. 
    RHF builder is available separately as they require drastically 
    different management of the MO coefficients.

    Args:
        hf (object): PySCF unrestricted Hartree-Fock Solver object.

    Returns:
        link_index_a, link_index_b (tuple): Objects that help contract 
            the many-body wavefunction and Hamiltonian.
        h2e (array): A complete Hamiltonian matrix for the system. 
    """
    norb, nelec_a, nelec_b = get_mol_attrs(hf.mol)

    # separate molecular orbital coefficients
    mo_a = hf.mo_coeff[0]
    mo_b = hf.mo_coeff[1]

    # compute 1-electron Hamiltonian
    hcore = hf.get_hcore()

    # compute one-electron integrals
    # takes hcore in AO as generated above and converts to MO basis, A = U B U^dag 
    h1e_a = reduce(np.dot, (mo_a.T, hcore, mo_a))
    h1e_b = reduce(np.dot, (mo_b.T, hcore, mo_b))

    # compute electron repulsion integrals -- NOT 2-electron integrals!
    # eri_{pq,rs} = (pq|rs) - (1/Nelec) \sum_q (pq|qs)
    # first term are two-electron integrals
    # again, hf._eri are in AO basis -- four-index object, so hard to do U B U^dag
    # instead, pass it to incore.general, with four copies of mo_a vectors, for each index
    g2e_aa = ao2mo.incore.general(hf._eri, (mo_a,)*4, compact=False)
    g2e_aa = g2e_aa.reshape(norb,norb,norb,norb)

    g2e_ab = ao2mo.incore.general(hf._eri, (mo_a,mo_a,mo_b,mo_b), compact=False)
    g2e_ab = g2e_ab.reshape(norb,norb,norb,norb)

    g2e_bb = ao2mo.incore.general(hf._eri, (mo_b,)*4, compact=False)
    g2e_bb = g2e_bb.reshape(norb,norb,norb,norb)

    # re-organize them together
    # these are 1-electron integrals, basically <|T_kin + V_ions|>
    h1e = (h1e_a, h1e_b)
    # these are NOT 2-electron integrals.
    eri = (g2e_aa, g2e_ab, g2e_bb)

    link_index_a = gen_linkstr_index_trilidx(range(norb), nelec_a)
    link_index_b = gen_linkstr_index_trilidx(range(norb), nelec_b)

    # this absorbs h1e into h2e so we only have one matrix
    h2e = absorb_h1e_uhf(h1e, eri, norb, (nelec_a, nelec_b), 0.5)

    return (link_index_a, link_index_b), h2e


def _contract_ufci_hamiltonian(mol, fcivec, link_index, h2e):
    r"""Take an FCI-format wavefunction (2D-array) and contract 
    it against the FCI Hamiltonian (unrestricted HF version).

    Args:
        mol (object): PySCF's Molecule object.
        fcivec (array): FCI wavefunction in PySCF's 2D-array format.
        link_index (object): Objects that help contract 
            the many-body wavefunction and Hamiltonian
        h2e (array): A complete Hamiltonian matrix for the system. 

    Returns:
        ci (array): The wavefunction post-contraction 
            (i.e. post-multiplication by the Hamiltonian). 
    """
    norb, nelec_a, nelec_b = get_mol_attrs(mol)
    ci = contract_2e_uhf(h2e, fcivec, norb, (nelec_a, nelec_b), link_index)
    return ci


def ufci_moments(hf, kmax, fcidict):
    r"""Return the moments of the FCI wavefunction (unrestricted HF case)
    by computing them directly through contracting the FCI wavefunction
    with the FCI Hamiltonian. The higher-order moments are obtained 
    recursively from the lower-order moments by successive contractions
    of the wavefunction with the Hamiltonian. 

    Args:
        hf (object): PySCF unrestricted Hartree-Fock Solver object.
        kmax (int): Largest moment to be calculated. 
        fcidict (dict): A wavefunction in Overlapper's sparse 
            dictionary format.

    Returns:
        moments (array): Array of the computed Hamiltonian moments of the 
            provided wavefunction `fcidict`. 
    """

    moments = np.zeros((kmax))

    fcivec = fcidict_to_fcivec(fcidict, hf.mol)

    # construct the FCI Hamiltonian
    link_index, h2e = _ufci_hamiltonian(hf)

    # compute the moments
    for ii in range(kmax):
        if ii == 0:
            ci1 = _contract_ufci_hamiltonian(hf.mol, fcivec, link_index, h2e)
            moments[ii] = np.dot(fcivec.reshape(-1), ci1.reshape(-1))
        else:
            ci1 = _contract_ufci_hamiltonian(hf.mol, ci1, link_index, h2e)
            moments[ii] = np.dot(fcivec.reshape(-1), ci1.reshape(-1))

    return moments



