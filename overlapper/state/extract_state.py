# Overlapper: Prepare and evaluate wavefunctions for quantum
# algorithms using computational chemistry techniques
# Copyright 2024 Xanadu Quantum Technologies Inc.
#
# Author: Stepan Fomichev <stepan.fomichev@xanadu.ai>
#         Joana Fraxanet <joana.fraxanet@icfo.eu>
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
#
# The extract state functions for CISD/CCSD/DMRG/SHCI in this source
# file are borrowed from the quantum computing library PennyLane, 
# specifically from _{cisd/ccsd/dmrg/shci}_state methods in
# https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/qchem/convert.py
# under the terms of the Apache License, Version 2.0 (the "License").

import numpy as np
from scipy.sparse import coo_matrix

from pyscf.fci.cistring import addrs2str
from itertools import product

from overlapper.utils.utils import get_mol_attrs, _excited_configurations, _sitevec_to_fock

def hf_state(solver_hf):
    r""" Construct a sparse dictionary object representing the wavefunction from the Hartree-Fock
    solution in PySCF. 
    [This function is inspired by PennyLane's qchem.convert._rcisd_state method 
    https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/qchem/convert.py#L664.]

    The generated wavefunction is a dictionary where the keys represent a configuration, which
    corresponds to a Slater determinant, and the values are the CI coefficients of the Slater
    determinant. Each dictionary key is a tuple of two integers. The binary representation of these
    integers correspond to a specific configuration: the first number represents the
    configuration of the alpha electrons and the second number represents the configuration of the
    beta electrons. For instance, the Hartree-Fock state :math:`|1 1 0 0 \rangle` will be
    represented by the flipped binary string ``0011` which is split to ``01`` and ``01`` for
    the alpha and beta electrons. The integer corresponding to ``01`` is ``1`` and the dictionary
    representation of the Hartree-Fock state will be ``{(1, 1): 1.0}``. The dictionary
    representation of a state with ``0.99`` contribution from the Hartree-Fock state and ``0.01``
    contribution from the doubly-excited state, i.e., :math:`|0 0 1 1 \rangle`, will be
    ``{(1, 1): 0.99, (2, 2): 0.01}``.

    Args:
        solver_hf (object): PySCF Hartree-Fock Solver object (restricted or unrestricted)

    Returns:
        wf (dict): wavefunction in sparse dict format
    """

    norb, nelec_a, nelec_b = get_mol_attrs(solver_hf.mol)

    # numbers representing the Hartree-Fock vector, e.g., bin(ref_a)[::-1] = 1111...10...0
    ref_a = int(2**nelec_a - 1)
    ref_b = int(2**nelec_b - 1)

    dict_fcimatr = dict( zip( list(zip([ref_a],[ref_b])) , [1.] ) )

    return dict_fcimatr

def _rcisd_state(solver_cisd, state=0, tol=1e-15):
    r""" Construct a sparse dictionary object representing the wavefunction from the restricted
    configuration interaction with singles and doubles (RCISD) solution in PySCF. 
    [This function is copied on PennyLane's qchem.convert._rcisd_state method 
    https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/qchem/convert.py#L664.]
        
    The generated wavefunction is a dictionary where the keys represent a configuration, which
    corresponds to a Slater determinant, and the values are the CI coefficients of the Slater
    determinant. Each dictionary key is a tuple of two integers. The binary representation of these
    integers correspond to a specific configuration: the first number represents the
    configuration of the alpha electrons and the second number represents the configuration of the
    beta electrons. For instance, the Hartree-Fock state :math:`|1 1 0 0 \rangle` will be
    represented by the flipped binary string ``0011` which is split to ``01`` and ``01`` for
    the alpha and beta electrons. The integer corresponding to ``01`` is ``1`` and the dictionary
    representation of the Hartree-Fock state will be ``{(1, 1): 1.0}``. The dictionary
    representation of a state with ``0.99`` contribution from the Hartree-Fock state and ``0.01``
    contribution from the doubly-excited state, i.e., :math:`|0 0 1 1 \rangle`, will be
    ``{(1, 1): 0.99, (2, 2): 0.01}``.

    Args:
        solver_cisd (object): PySCF RCISD Solver object (restricted)
        state (int): which state to do the conversion for, if within CISD multiple states were solved for (i.e. `nroots > 1`) -- default is 0 (the lowest-energy state)
        tol (float): the tolerance for discarding Slater determinants based on their coefficients

    Returns:
        wf (dict): wavefunction in sparse dict format
    """

    mol = solver_cisd.mol
    norb = mol.nao
    nelec = mol.nelectron
    nocc, nvir = nelec // 2, norb - nelec // 2

    # extract the CI coeffs from the right state
    if not (state in range(solver_cisd.nroots)):
        raise IndexError("State requested has not been solved for. Re-run CISD with larger nroots.")
    if solver_cisd.nroots > 1:
        cisdvec = solver_cisd.ci[state]
    else:
        cisdvec = solver_cisd.ci 

    c0, c1, c2 = (
        cisdvec[0],
        cisdvec[1 : nocc * nvir + 1],
        cisdvec[nocc * nvir + 1 :].reshape(nocc, nocc, nvir, nvir),
    )

    # numbers representing the Hartree-Fock vector, e.g., bin(ref_a)[::-1] = 1111...10...0
    ref_a = int(2**nocc - 1)
    ref_b = ref_a

    dict_fcimatr = dict(zip(list(zip([ref_a], [ref_b])), [c0]))

    # alpha -> alpha excitations
    c1a_configs, c1a_signs = _excited_configurations(nocc, norb, 1)
    dict_fcimatr.update(
        dict(zip(list(zip(c1a_configs, [ref_b] * len(c1a_configs))), c1 * c1a_signs))
    )
    # beta -> beta excitations
    dict_fcimatr.update(
        dict(zip(list(zip([ref_a] * len(c1a_configs), c1a_configs)), c1 * c1a_signs))
    )

    # check if double excitations within one spin sector (aa->aa and bb->bb) are possible
    if nocc > 1 and nvir > 1:
        # get rid of excitations from identical orbitals, double-count the allowed ones
        c2_tr = c2 - c2.transpose(1, 0, 2, 3)
        # select only unique excitations, via lower triangle of matrix (already double-counted)
        ooidx, vvidx = np.tril_indices(nocc, -1), np.tril_indices(nvir, -1)
        c2aa = c2_tr[ooidx][:, vvidx[0], vvidx[1]].ravel()

        # alpha, alpha -> alpha, alpha excitations
        c2aa_configs, c2aa_signs = _excited_configurations(nocc, norb, 2)
        dict_fcimatr.update(
            dict(zip(list(zip(c2aa_configs, [ref_b] * len(c2aa_configs))), c2aa * c2aa_signs))
        )
        # beta, beta -> beta, beta excitations
        dict_fcimatr.update(
            dict(zip(list(zip([ref_a] * len(c2aa_configs), c2aa_configs)), c2aa * c2aa_signs))
        )

    # alpha, beta -> alpha, beta excitations
    # generate all possible pairwise combinations of _single_ excitations of alpha and beta sectors
    rowvals, colvals = np.array(list(product(c1a_configs, c1a_configs)), dtype=int).T
    c2ab = (c2.transpose(0, 2, 1, 3).reshape(nocc * nvir, -1)).ravel()
    dict_fcimatr.update(
        dict(zip(list(zip(rowvals, colvals)), c2ab * np.kron(c1a_signs, c1a_signs)))
    )

    # filter based on tolerance cutoff
    dict_fcimatr = {key: value for key, value in dict_fcimatr.items() if abs(value) > tol}

    return dict_fcimatr

def _ucisd_state(solver_cisd, state=0, tol=1e-15):
    r""" Construct a sparse dictionary object representing the wavefunction from the unrestricted
    configuration interaction with singles and doubles (UCISD) solution in PySCF. 
    [This function is copied on PennyLane's qchem.convert._ucisd_state method 
    https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/qchem/convert.py#L476.]

    The generated wavefunction is a dictionary where the keys represent a configuration, which
    corresponds to a Slater determinant, and the values are the CI coefficients of the Slater
    determinant. Each dictionary key is a tuple of two integers. The binary representation of these
    integers correspond to a specific configuration: the first number represents the
    configuration of the alpha electrons and the second number represents the configuration of the
    beta electrons. For instance, the Hartree-Fock state :math:`|1 1 0 0 \rangle` will be
    represented by the flipped binary string ``0011` which is split to ``01`` and ``01`` for
    the alpha and beta electrons. The integer corresponding to ``01`` is ``1`` and the dictionary
    representation of the Hartree-Fock state will be ``{(1, 1): 1.0}``. The dictionary
    representation of a state with ``0.99`` contribution from the Hartree-Fock state and ``0.01``
    contribution from the doubly-excited state, i.e., :math:`|0 0 1 1 \rangle`, will be
    ``{(1, 1): 0.99, (2, 2): 0.01}``.

    Args:
        solver_cisd (object): PySCF UCISD Solver object (unrestricted)
        state (int): which state to do the conversion for, if within CISD multiple states were solved for (i.e. `nroots > 1`) -- default is 0 (the lowest-energy state)
        tol (float): the tolerance for discarding Slater determinants based on their coefficients

    Returns:
        wf (dict): wavefunction in sparse dict format
    """

    mol = solver_cisd.mol
    norb = mol.nao
    nelec = mol.nelectron
    nelec_a = int((nelec + mol.spin) / 2)
    nelec_b = int((nelec - mol.spin) / 2)

    nvir_a, nvir_b = norb - nelec_a, norb - nelec_b

    size_a, size_b = nelec_a * nvir_a, nelec_b * nvir_b
    size_aa = int(nelec_a * (nelec_a - 1) / 2) * int(nvir_a * (nvir_a - 1) / 2)
    size_bb = int(nelec_b * (nelec_b - 1) / 2) * int(nvir_b * (nvir_b - 1) / 2)
    size_ab = nelec_a * nelec_b * nvir_a * nvir_b

    # extract the CI coeffs from the right state
    if not (state in range(solver_cisd.nroots)):
        raise IndexError("State requested has not been solved for. Re-run CISD with larger nroots.")
    if solver_cisd.nroots > 1:
        cisdvec = solver_cisd.ci[state]
    else:
        cisdvec = solver_cisd.ci 

    sizes = [1, size_a, size_b, size_aa, size_ab, size_bb]
    cumul = np.cumsum(sizes)
    idxs = [0] + [slice(cumul[ii], cumul[ii + 1]) for ii in range(len(cumul) - 1)]
    c0, c1a, c1b, c2aa, c2ab, c2bb = [cisdvec[idx] for idx in idxs]

    # numbers representing the Hartree-Fock vector, e.g., bin(ref_a)[::-1] = 1111...10...0
    ref_a = int(2**nelec_a - 1)
    ref_b = int(2**nelec_b - 1)

    dict_fcimatr = dict(zip(list(zip([ref_a], [ref_b])), [c0]))

    # alpha -> alpha excitations
    c1a_configs, c1a_signs = _excited_configurations(nelec_a, norb, 1)
    dict_fcimatr.update(dict(zip(list(zip(c1a_configs, [ref_b] * size_a)), c1a * c1a_signs)))

    # beta -> beta excitations
    c1b_configs, c1b_signs = _excited_configurations(nelec_b, norb, 1)
    dict_fcimatr.update(dict(zip(list(zip([ref_a] * size_b, c1b_configs)), c1b * c1b_signs)))

    # alpha, alpha -> alpha, alpha excitations
    c2aa_configs, c2aa_signs = _excited_configurations(nelec_a, norb, 2)
    dict_fcimatr.update(dict(zip(list(zip(c2aa_configs, [ref_b] * size_aa)), c2aa * c2aa_signs)))

    # alpha, beta -> alpha, beta excitations
    rowvals, colvals = np.array(list(product(c1a_configs, c1b_configs)), dtype=int).T
    dict_fcimatr.update(
        dict(zip(list(zip(rowvals, colvals)), c2ab * np.kron(c1a_signs, c1b_signs)))
    )

    # beta, beta -> beta, beta excitations
    c2bb_configs, c2bb_signs = _excited_configurations(nelec_b, norb, 2)
    dict_fcimatr.update(dict(zip(list(zip([ref_a] * size_bb, c2bb_configs)), c2bb * c2bb_signs)))

    # filter based on tolerance cutoff
    dict_fcimatr = {key: value for key, value in dict_fcimatr.items() if abs(value) > tol}

    return dict_fcimatr

def cisd_state(solver_cisd, state=0, tol=1e-15):
    r"""Wrapper that constructs a sparse dictionary object representing 
    the wavefunction from the restricted or unrestricted configuration interaction 
    with singles and doubles (RCISD/UCISD) solution in PySCF. It does so 
    by redirecting the flow to the appropriate constructor function.

    The generated wavefunction is a dictionary where the keys represent a configuration, which
    corresponds to a Slater determinant, and the values are the CI coefficients of the Slater
    determinant. Each dictionary key is a tuple of two integers. The binary representation of these
    integers correspond to a specific configuration: the first number represents the
    configuration of the alpha electrons and the second number represents the configuration of the
    beta electrons. For instance, the Hartree-Fock state :math:`|1 1 0 0 \rangle` will be
    represented by the flipped binary string ``0011` which is split to ``01`` and ``01`` for
    the alpha and beta electrons. The integer corresponding to ``01`` is ``1`` and the dictionary
    representation of the Hartree-Fock state will be ``{(1, 1): 1.0}``. The dictionary
    representation of a state with ``0.99`` contribution from the Hartree-Fock state and ``0.01``
    contribution from the doubly-excited state, i.e., :math:`|0 0 1 1 \rangle`, will be
    ``{(1, 1): 0.99, (2, 2): 0.01}``.

    Args:
        solver_cisd (object): PySCF RCISD/UCISD Solver object
        tol (float): the tolerance for discarding Slater determinants based on their coefficients

    Returns:
        wf (dict): wavefunction in sparse dict format
    """
    hftype = str(solver_cisd.__str__)

    if 'ucisd' in hftype.lower():
        wf = _ucisd_state(solver_cisd, state=state, tol=tol)
    elif 'cisd' in hftype.lower() and not ("ucisd" in hftype.lower()):
        wf = _rcisd_state(solver_cisd, state=state, tol=tol)
    else:
        raise ValueError("Unknown HF reference character. The only supported types are RHF, ROHF and UHF.")

    return wf

def _rccsd_state(solver_ccsd, tol=1e-15):
    r""" Construct a sparse dictionary object representing the wavefunction from the restricted
    coupled cluster with singles and doubles (RCCSD) solution in PySCF. 
    [This function is copied on PennyLane's qchem.convert._rccsd_state method
    https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/qchem/convert.py#L760.]

    The generated wavefunction is a dictionary where the keys represent a configuration, which
    corresponds to a Slater determinant, and the values are the CI coefficients of the Slater
    determinant. Each dictionary key is a tuple of two integers. The binary representation of these
    integers correspond to a specific configuration: the first number represents the
    configuration of the alpha electrons and the second number represents the configuration of the
    beta electrons. For instance, the Hartree-Fock state :math:`|1 1 0 0 \rangle` will be
    represented by the flipped binary string ``0011` which is split to ``01`` and ``01`` for
    the alpha and beta electrons. The integer corresponding to ``01`` is ``1`` and the dictionary
    representation of the Hartree-Fock state will be ``{(1, 1): 1.0}``. The dictionary
    representation of a state with ``0.99`` contribution from the Hartree-Fock state and ``0.01``
    contribution from the doubly-excited state, i.e., :math:`|0 0 1 1 \rangle`, will be
    ``{(1, 1): 0.99, (2, 2): 0.01}``.

    In the current version, the exponential ansatz :math:`\exp(\hat{T}_1 + \hat{T}_2) \ket{\text{HF}}`
    is expanded to second order, with only single and double excitation terms collected and kept.
    In the future this will be amended to also collect terms from higher order. The expansion gives

    .. math::
        \exp(\hat{T}_1 + \hat{T}_2) \ket{\text{HF}} = \left[ 1 + \hat{T}_1 +
        \left( \hat{T}_2 + 0.5 * \hat{T}_1^2 \right) \right] \ket{\text{HF}}

    The coefficients in this expansion are the CI coefficients used to build the wavefunction
    representation.

    Args:
        solver_ccsd (object): PySCF CCSD Solver object (restricted)
        tol (float): the tolerance for discarding Slater determinants based on their coefficients

    Returns:
        wf (dict): wavefunction in sparse dict format
    """

    mol = solver_ccsd.mol

    norb, nelec_a, nelec_b = get_mol_attrs(mol)
    if not (nelec_a == nelec_b):
        raise ValueError("For RHF-based CCSD the molecule must be closed shell.")

    nvir_a, nvir_b = norb - nelec_a, norb - nelec_b

    # build the full, unrestricted representation of the coupled cluster amplitudes
    t1a = solver_ccsd.t1
    t1b = t1a
    t2aa = solver_ccsd.t2 - solver_ccsd.t2.transpose(1, 0, 2, 3)
    t2ab = solver_ccsd.t2.transpose(0, 2, 1, 3)
    t2bb = t2aa

    # add in the disconnected part ( + 0.5 T_1^2) of double excitations
    t2aa = (
        t2aa
        - 0.5
        * np.kron(t1a, t1a).reshape(nelec_a, nvir_a, nelec_a, nvir_a).transpose(0, 2, 1, 3)
    )
    t2bb = (
        t2bb
        - 0.5
        * np.kron(t1b, t1b).reshape(nelec_b, nvir_b, nelec_b, nvir_b).transpose(0, 2, 1, 3)
    )
    # align the entries with how the excitations are ordered when generated by _excited_configurations()
    t2ab = t2ab - 0.5 * np.kron(t1a, t1b).reshape(nelec_a, nvir_a, nelec_b, nvir_b)

    # numbers representing the Hartree-Fock vector, e.g., bin(ref_a)[::-1] = 1111...10...0
    ref_a = int(2**nelec_a - 1)
    ref_b = int(2**nelec_b - 1)

    dict_fcimatr = dict(zip(list(zip([ref_a], [ref_b])), [1.0]))

    # alpha -> alpha excitations
    t1a_configs, t1a_signs = _excited_configurations(nelec_a, norb, 1)
    dict_fcimatr.update(
        dict(zip(list(zip(t1a_configs, [ref_b] * len(t1a_configs))), t1a.ravel() * t1a_signs))
    )

    # beta -> beta excitations
    t1b_configs, t1b_signs = _excited_configurations(nelec_b, norb, 1)
    dict_fcimatr.update(
        dict(zip(list(zip([ref_a] * len(t1b_configs), t1b_configs)), t1b.ravel() * t1b_signs))
    )

    # alpha, alpha -> alpha, alpha excitations
    if nelec_a > 1 and nvir_a > 1:
        t2aa_configs, t2aa_signs = _excited_configurations(nelec_a, norb, 2)
        # select only unique excitations, via lower triangle of matrix
        ooidx = np.tril_indices(nelec_a, -1)
        vvidx = np.tril_indices(nvir_a, -1)
        t2aa = t2aa[ooidx][:, vvidx[0], vvidx[1]]
        dict_fcimatr.update(
            dict(
                zip(list(zip(t2aa_configs, [ref_b] * len(t2aa_configs))), t2aa.ravel() * t2aa_signs)
            )
        )

    if nelec_b > 1 and nvir_b > 1:
        t2bb_configs, t2bb_signs = _excited_configurations(nelec_b, norb, 2)
        # select only unique excitations, via lower triangle of matrix
        ooidx = np.tril_indices(nelec_b, -1)
        vvidx = np.tril_indices(nvir_b, -1)
        t2bb = t2bb[ooidx][:, vvidx[0], vvidx[1]]
        dict_fcimatr.update(
            dict(
                zip(list(zip([ref_a] * len(t2bb_configs), t2bb_configs)), t2bb.ravel() * t2bb_signs)
            )
        )

    # alpha, beta -> alpha, beta excitations
    rowvals, colvals = np.array(list(product(t1a_configs, t1b_configs)), dtype=int).T
    dict_fcimatr.update(
        dict(zip(list(zip(rowvals, colvals)), t2ab.ravel() * np.kron(t1a_signs, t1b_signs)))
    )

    # renormalize, to get the HF coefficient (CC wavefunction not normalized)
    norm = np.sqrt(np.sum(np.array(list(dict_fcimatr.values())) ** 2))
    dict_fcimatr = {key: value / norm for (key, value) in dict_fcimatr.items()}

    # filter based on tolerance cutoff
    dict_fcimatr = {key: value for key, value in dict_fcimatr.items() if abs(value) > tol}

    return dict_fcimatr

def _uccsd_state(solver_ccsd, tol=1e-15):
    r""" Construct a sparse dictionary object representing the wavefunction from the unrestricted
    coupled cluster with singles and doubles (UCCSD) solution in PySCF. 
    [This function is copied on PennyLane's qchem.convert._uccsd_state method
    https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/qchem/convert.py#L893.]

    The generated wavefunction is a dictionary where the keys represent a configuration, which
    corresponds to a Slater determinant, and the values are the CI coefficients of the Slater
    determinant. Each dictionary key is a tuple of two integers. The binary representation of these
    integers correspond to a specific configuration: the first number represents the
    configuration of the alpha electrons and the second number represents the configuration of the
    beta electrons. For instance, the Hartree-Fock state :math:`|1 1 0 0 \rangle` will be
    represented by the flipped binary string ``0011` which is split to ``01`` and ``01`` for
    the alpha and beta electrons. The integer corresponding to ``01`` is ``1`` and the dictionary
    representation of the Hartree-Fock state will be ``{(1, 1): 1.0}``. The dictionary
    representation of a state with ``0.99`` contribution from the Hartree-Fock state and ``0.01``
    contribution from the doubly-excited state, i.e., :math:`|0 0 1 1 \rangle`, will be
    ``{(1, 1): 0.99, (2, 2): 0.01}``.

    In the current version, the exponential ansatz :math:`\exp(\hat{T}_1 + \hat{T}_2) \ket{\text{HF}}`
    is expanded to second order, with only single and double excitation terms collected and kept.
    In the future this will be amended to also collect terms from higher order. The expansion gives

    .. math::
        \exp(\hat{T}_1 + \hat{T}_2) \ket{\text{HF}} = \left[ 1 + \hat{T}_1 +
        \left( \hat{T}_2 + 0.5 * \hat{T}_1^2 \right) \right] \ket{\text{HF}}

    The coefficients in this expansion are the CI coefficients used to build the wavefunction
    representation.

    Args:
        solver_ccsd (object): PySCF UCCSD Solver object (unrestricted)
        tol (float): the tolerance for discarding Slater determinants based on their coefficients

    Returns:
        wf (dict): wavefunction in sparse dict format
    """

    mol = solver_ccsd.mol

    norb = mol.nao
    nelec = mol.nelectron
    nelec_a = int((nelec + mol.spin) / 2)
    nelec_b = int((nelec - mol.spin) / 2)

    nvir_a, nvir_b = norb - nelec_a, norb - nelec_b

    t1a, t1b = solver_ccsd.t1
    t2aa, t2ab, t2bb = solver_ccsd.t2
    # add in the disconnected part ( + 0.5 T_1^2) of double excitations
    t2aa = (
        t2aa
        - 0.5
        * np.kron(t1a, t1a).reshape(nelec_a, nvir_a, nelec_a, nvir_a).transpose(0, 2, 1, 3)
    )
    t2bb = (
        t2bb
        - 0.5
        * np.kron(t1b, t1b).reshape(nelec_b, nvir_b, nelec_b, nvir_b).transpose(0, 2, 1, 3)
    )
    # align the entries with how the excitations are ordered when generated by _excited_configurations()
    t2ab = (
        t2ab.transpose(0, 2, 1, 3)
        - 0.5 * np.kron(t1a, t1b).reshape(nelec_a, nvir_a, nelec_b, nvir_b)
    )

    # numbers representing the Hartree-Fock vector, e.g., bin(ref_a)[::-1] = 1111...10...0
    ref_a = int(2**nelec_a - 1)
    ref_b = int(2**nelec_b - 1)

    dict_fcimatr = dict(zip(list(zip([ref_a], [ref_b])), [1.0]))

    # alpha -> alpha excitations
    t1a_configs, t1a_signs = _excited_configurations(nelec_a, norb, 1)
    dict_fcimatr.update(
        dict(zip(list(zip(t1a_configs, [ref_b] * len(t1a_configs))), t1a.ravel() * t1a_signs))
    )

    # beta -> beta excitations
    t1b_configs, t1b_signs = _excited_configurations(nelec_b, norb, 1)
    dict_fcimatr.update(
        dict(zip(list(zip([ref_a] * len(t1b_configs), t1b_configs)), t1b.ravel() * t1b_signs))
    )

    # alpha, alpha -> alpha, alpha excitations
    if nelec_a > 1 and nvir_a > 1:
        t2aa_configs, t2aa_signs = _excited_configurations(nelec_a, norb, 2)
        # select only unique excitations, via lower triangle of matrix
        ooidx = np.tril_indices(nelec_a, -1)
        vvidx = np.tril_indices(nvir_a, -1)
        t2aa = t2aa[ooidx][:, vvidx[0], vvidx[1]]
        dict_fcimatr.update(
            dict(
                zip(list(zip(t2aa_configs, [ref_b] * len(t2aa_configs))), t2aa.ravel() * t2aa_signs)
            )
        )

    if nelec_b > 1 and nvir_b > 1:
        t2bb_configs, t2bb_signs = _excited_configurations(nelec_b, norb, 2)
        # select only unique excitations, via lower triangle of matrix
        ooidx = np.tril_indices(nelec_b, -1)
        vvidx = np.tril_indices(nvir_b, -1)
        t2bb = t2bb[ooidx][:, vvidx[0], vvidx[1]]
        dict_fcimatr.update(
            dict(
                zip(list(zip([ref_a] * len(t2bb_configs), t2bb_configs)), t2bb.ravel() * t2bb_signs)
            )
        )

    # alpha, beta -> alpha, beta excitations
    rowvals, colvals = np.array(list(product(t1a_configs, t1b_configs)), dtype=int).T
    dict_fcimatr.update(
        dict(zip(list(zip(rowvals, colvals)), t2ab.ravel() * np.kron(t1a_signs, t1b_signs)))
    )

    # renormalize, to get the HF coefficient (CC wavefunction not normalized)
    norm = np.sqrt(np.sum(np.array(list(dict_fcimatr.values())) ** 2))
    dict_fcimatr = {key: value / norm for (key, value) in dict_fcimatr.items()}

    # filter based on tolerance cutoff
    dict_fcimatr = {key: value for key, value in dict_fcimatr.items() if abs(value) > tol}

    return dict_fcimatr

def ccsd_state(solver_ccsd, tol=1e-15):
    r"""Wrapper that constructs a sparse dictionary object representing 
    the wavefunction from the restricted or unrestricted coupled cluster 
    with singles and doubles (RCCSD/UCCSD) solution in PySCF. It does so 
    by redirecting the flow to the appropriate constructor function.

    PySCF's implementation of CCSD does not support calculation of excited state wavefunctions,
    so unlike all other state generation methods, the `ccsd_state` method does not support the 
    `state` argument for selection of excited states.

    The generated wavefunction is a dictionary where the keys represent a configuration, which
    corresponds to a Slater determinant, and the values are the CI coefficients of the Slater
    determinant. Each dictionary key is a tuple of two integers. The binary representation of these
    integers correspond to a specific configuration: the first number represents the
    configuration of the alpha electrons and the second number represents the configuration of the
    beta electrons. For instance, the Hartree-Fock state :math:`|1 1 0 0 \rangle` will be
    represented by the flipped binary string ``0011` which is split to ``01`` and ``01`` for
    the alpha and beta electrons. The integer corresponding to ``01`` is ``1`` and the dictionary
    representation of the Hartree-Fock state will be ``{(1, 1): 1.0}``. The dictionary
    representation of a state with ``0.99`` contribution from the Hartree-Fock state and ``0.01``
    contribution from the doubly-excited state, i.e., :math:`|0 0 1 1 \rangle`, will be
    ``{(1, 1): 0.99, (2, 2): 0.01}``.

    In the current version, the exponential ansatz :math:`\exp(\hat{T}_1 + \hat{T}_2) \ket{\text{HF}}`
    is expanded to second order, with only single and double excitation terms collected and kept.
    In the future this will be amended to also collect terms from higher order. The expansion gives

    .. math::
        \exp(\hat{T}_1 + \hat{T}_2) \ket{\text{HF}} = \left[ 1 + \hat{T}_1 +
        \left( \hat{T}_2 + 0.5 * \hat{T}_1^2 \right) \right] \ket{\text{HF}}

    The coefficients in this expansion are the CI coefficients used to build the wavefunction
    representation.

    Args:
        solver_ccsd (object): PySCF RCCSD/UCCSD Solver object
        tol (float): the tolerance for discarding Slater determinants based on their coefficients

    Returns:
        wf (dict): wavefunction in sparse dict format
    """
    hftype = str(solver_ccsd.__str__)
    if "uccsd" in hftype.lower():
        wf = _uccsd_state(solver_ccsd, tol=tol)
    elif "ccsd" in hftype.lower() and not ("uccsd" in hftype.lower()):
        wf = _rccsd_state(solver_ccsd, tol=tol)
    else:
        raise ValueError("Unknown HF reference character. The only supported types are RHF, ROHF and UHF.")

    return wf

def casci_state(solver_casci,state=0,tol=1e-15):
    r""" Construct a sparse dictionary object representing the wavefunction from the complete
    active space configuration interaction (CASCI) solution in PySCF. 
    
    The generated wavefunction is a dictionary where the keys represent a configuration, which
    corresponds to a Slater determinant, and the values are the CI coefficients of the Slater
    determinant. Each dictionary key is a tuple of two integers. The binary representation of these
    integers correspond to a specific configuration: the first number represents the
    configuration of the alpha electrons and the second number represents the configuration of the
    beta electrons. For instance, the Hartree-Fock state :math:`|1 1 0 0 \rangle` will be
    represented by the flipped binary string ``0011` which is split to ``01`` and ``01`` for
    the alpha and beta electrons. The integer corresponding to ``01`` is ``1`` and the dictionary
    representation of the Hartree-Fock state will be ``{(1, 1): 1.0}``. The dictionary
    representation of a state with ``0.99`` contribution from the Hartree-Fock state and ``0.01``
    contribution from the doubly-excited state, i.e., :math:`|0 0 1 1 \rangle`, will be
    ``{(1, 1): 0.99, (2, 2): 0.01}``.

    Args:
        solver_casci (object): PySCF CASCI Solver object (restricted or unrestricted)
        state (int): which state to do the conversion for, if within CASCI multiple states were solved for (i.e. `nroots > 1`) -- default is 0 (the lowest-energy state)
        tol (float): the tolerance for discarding Slater determinants based on their coefficients

    Returns:
        wf (dict): wavefunction in sparse dict format
    """

    norb, nelec_a, nelec_b = get_mol_attrs(solver_casci.mol)
    try:
        ncore_a, ncore_b = solver_casci.ncore
    except TypeError:
        ncore_a = solver_casci.ncore
        ncore_b = ncore_a

    ncas_a = norb - ncore_a
    ncas_b = norb - ncore_b
    nelecas_a = nelec_a - ncore_a
    nelecas_b = nelec_b - ncore_b

    # extract the CI coeffs from the right state
    if not (state in range(solver_casci.fcisolver.nroots)):
        raise IndexError("State requested has not been solved for. Re-run CASCI with larger nroots.")
    if solver_casci.fcisolver.nroots > 1:
        cascivec = solver_casci.ci[state]
    else:
        cascivec = solver_casci.ci

    # filter determinants with coefficients below tol
    cascivec[abs(cascivec) < tol] = 0
    sparse_cascimatr = coo_matrix( cascivec, shape=np.shape(cascivec), dtype=float )
    row, col, dat = sparse_cascimatr.row, sparse_cascimatr.col, sparse_cascimatr.data

    ## turn FCI wavefunction matrix indices into integers representing Fock occupation vectors
    ints_row = addrs2str(ncas_a, nelecas_a, row)
    ints_col = addrs2str(ncas_b, nelecas_b, col)

    ## pad the integers to recover the full-space wavefunction
    padded_ints_row = (ints_row << ncore_a) | int(2**ncore_a-1)
    padded_ints_col = (ints_col << ncore_b) | int(2**ncore_b-1)

    ## create the FCI matrix as a dict
    dict_fcimatr = dict( zip(list(zip(padded_ints_row, padded_ints_col)), dat) )

    return dict_fcimatr

def mrpt_state(solver_mrpt, state=0, tol=1e-15):
    r""" Construct a sparse dictionary object representing the wavefunction from the multireference
    perturbation theory (MRPT) solution in PySCF. Since MRPT merely modifies coefficients of CASCI states,
    this function is just a wrapper for the equivalent CASCI method.
    
    The generated wavefunction is a dictionary where the keys represent a configuration, which
    corresponds to a Slater determinant, and the values are the CI coefficients of the Slater
    determinant. Each dictionary key is a tuple of two integers. The binary representation of these
    integers correspond to a specific configuration: the first number represents the
    configuration of the alpha electrons and the second number represents the configuration of the
    beta electrons. For instance, the Hartree-Fock state :math:`|1 1 0 0 \rangle` will be
    represented by the flipped binary string ``0011` which is split to ``01`` and ``01`` for
    the alpha and beta electrons. The integer corresponding to ``01`` is ``1`` and the dictionary
    representation of the Hartree-Fock state will be ``{(1, 1): 1.0}``. The dictionary
    representation of a state with ``0.99`` contribution from the Hartree-Fock state and ``0.01``
    contribution from the doubly-excited state, i.e., :math:`|0 0 1 1 \rangle`, will be
    ``{(1, 1): 0.99, (2, 2): 0.01}``.

    Args:
        solver_mrpt (object): PySCF CASCI Solver object post-MRPT calculations
        state (int): which state to do the conversion for, if within CASCI multiple states were solved for (i.e. `nroots > 1`) -- default is 0 (the lowest-energy state)
        tol (float): the tolerance for discarding Slater determinants based on their coefficients

    Returns:
        wf (dict): wavefunction in sparse dict format
    """
    dict_fcimatr = casci_state(solver_mrpt, state=state, tol=tol)
    return dict_fcimatr

def dmrg_state(solver_dmrg, state=0, tol=1e-15):
    r"""Construct a wavefunction from the DMRG wavefunction obtained from the Block2 library.
    [This function is copied on PennyLane's qchem.convert._dmrg_state method
    https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/qchem/convert.py#L1023.]

    The generated wavefunction is a dictionary where the keys represent a configuration, which
    corresponds to a Slater determinant, and the values are the CI coefficients of the Slater
    determinant. Each dictionary key is a tuple of two integers. The binary representation of these
    integers correspond to a specific configuration: the first number represents the
    configuration of the alpha electrons and the second number represents the configuration of the
    beta electrons. For instance, the Hartree-Fock state :math:`|1 1 0 0 \rangle` will be
    represented by the flipped binary string ``0011`` which is split to ``01`` and ``01`` for
    the alpha and beta electrons. The integer corresponding to ``01`` is ``1`` and the dictionary
    representation of the Hartree-Fock state will be ``{(1, 1): 1.0}``. The dictionary
    representation of a state with ``0.99`` contribution from the Hartree-Fock state and ``0.01``
    contribution from the doubly-excited state, i.e., :math:`|0 0 1 1 \rangle`, will be
    ``{(1, 1): 0.99, (2, 2): 0.01}``.

    The determinants and coefficients are supplied externally. They are calculated with Block2 
    DMRGDriver's `get_csf_coefficients()` method and passed as a tuple in the first argument. 
    If the DMRG calculation was executed in SZ mode, the wavefunction is built in terms of Slater
    determinants (eigenfunctions of the :math:`S_z` operator); if it was in SU(2) mode, the 
    wavefunction is automatically built out of configuration state functions (CSF -- 
    eigenfunctions of the :math:`S^2` operator).

    Args:
        wavefunction tuple(list[int], array[float]): determinants and coefficients in physicist notation, as output by Block2 DMRGDriver's `get_csf_coefficients()` methods
        tol (float): the tolerance for discarding Slater determinants with small coefficients

    Returns:
        wf (dict): wavefunction in sparse dict format
    """

    dets, coeffs = solver_dmrg[state]

    row, col, dat = [], [], []
    
    for ii, det in enumerate(dets):
        stra, strb = _sitevec_to_fock(det, format="dmrg")
        row.append(stra)
        col.append(strb)

        # compute and fix parity to stick to pyscf notation
        lsta = np.array(list(map(int, bin(stra)[2:])))[::-1]
        lstb = np.array(list(map(int, bin(strb)[2:])))[::-1]

        # pad the shorter list
        maxlen = max([len(lsta), len(lstb)])
        lsta = np.pad(lsta, (0, maxlen - len(lsta)))
        lstb = np.pad(lstb, (0, maxlen - len(lstb)))

        which_occ = np.where(lsta == 1)[0]
        parity = (-1) ** np.sum([np.sum(lstb[: int(ind)]) for ind in which_occ])
        dat.append(parity * coeffs[ii])

    ## create the FCI matrix as a dict
    dict_fcimatr = dict(zip(list(zip(row, col)), dat))

    # filter based on tolerance cutoff
    dict_fcimatr = {key: value for key, value in dict_fcimatr.items() if abs(value) > tol}

    return dict_fcimatr

def shci_state(solver_shci, state=0, tol=1e-15):    
    r"""Construct a wavefunction from the SHCI wavefunction obtained from the Dice library.
    [This function is copied on PennyLane's qchem.convert._shci_state method
    https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/qchem/convert.py#L1138.]

    The generated wavefunction is a dictionary where the keys represent a configuration, which
    corresponds to a Slater determinant, and the values are the CI coefficients of the Slater
    determinant. Each dictionary key is a tuple of two integers. The binary representation of these
    integers correspond to a specific configuration: the first number represents the
    configuration of the alpha electrons and the second number represents the configuration of the
    beta electrons. For instance, the Hartree-Fock state :math:`|1 1 0 0 \rangle` will be
    represented by the flipped binary string ``0011`` which is split to ``01`` and ``01`` for
    the alpha and beta electrons. The integer corresponding to ``01`` is ``1`` and the dictionary
    representation of the Hartree-Fock state will be ``{(1, 1): 1.0}``. The dictionary
    representation of a state with ``0.99`` contribution from the Hartree-Fock state and ``0.01``
    contribution from the doubly-excited state, i.e., :math:`|0 0 1 1 \rangle`, will be
    ``{(1, 1): 0.99, (2, 2): 0.01}``.

    The determinants and coefficients should be supplied externally. They are stored under
    SHCI.outputfile and read from disk using a special helper function.

    Args:
        wavefunction tuple(list[str], array[float]): determinants and coefficients in chemist notation
        tol (float): the tolerance for discarding Slater determinants with small coefficients
    Returns:
        wf (dict): wavefunction in sparse dict format
    """

    dets, coeffs = solver_shci[state]

    xa = []
    xb = []
    dat = []

    for coeff, det in zip(coeffs, dets):
        if abs(coeff) > tol:
            bin_a, bin_b = _sitevec_to_fock(list(det), "shci")

            xa.append(bin_a)
            xb.append(bin_b)
            dat.append(coeff)

    # pad the integers with the core electrons
    ncore_a = ncore_b = 0
    padded_ints_row = (np.array(xa) << ncore_a) | int(2**ncore_a-1)
    padded_ints_col = (np.array(xb) << ncore_b) | int(2**ncore_b-1)

    ## create the FCI matrix as a dict
    dict_fcimatr = dict( zip(list(zip(padded_ints_row, padded_ints_col)), dat) )            

    # filter based on tolerance cutoff
    dict_fcimatr = {key: value for key, value in dict_fcimatr.items() if abs(value) > tol}
        
    return dict_fcimatr