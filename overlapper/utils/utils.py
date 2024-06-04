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
#
# The auxiliary functions _sitevec_to_fock, _excitations,
#  _excited_configurations in this source file are borrowed 
# from the quantum computing library PennyLane, 
# specifically from identically named functions in qchem.convert,
# https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/qchem/convert.py
# under the terms of the Apache License, Version 2.0 (the "License").

from pyscf.fci.cistring import str2addr
from pyscf import gto, scf
import numpy as np
from scipy.special import comb

def get_mol_attrs(mol):
    '''
    Simple extractor to compute often used molecular properties. Returns
    the number of molecular orbitals (NOT spin-orbitals), and the number of
    electrons of both alpha and beta type, depending on the spin. By default,
    takes nelec_a > nelec_b to satisfy spin conditions.
    '''
    norb = mol.nao
    nelec = mol.nelectron
    nelec_a = int( (nelec + mol.spin)/2 )
    nelec_b = int( (nelec - mol.spin)/2 )

    return norb, nelec_a, nelec_b

def str_to_block2(det, ncas):
    r"""Str det comes as an integer. Need to return to DMRG form.
    Which is given by a list with entries 0, 1, 2, 3
    """

    deta, detb = det
    lsta = np.array(list(map(int, bin(deta)[2:])))[::-1]
    lstb = np.array(list(map(int, bin(detb)[2:])))[::-1]

    lsta = np.pad(lsta, (0, ncas - len(lsta)) )
    lstb = np.pad(lstb, (0, ncas - len(lstb)) )

    block2det = []
    for ii, _ in enumerate(lsta):
        if lsta[ii] == 0 and lstb[ii] == 0:
            block2det.append(0)
        elif lsta[ii] == 1 and lstb[ii] == 0:
            block2det.append(1)
        elif lsta[ii] == 0 and lstb[ii] == 1:
            block2det.append(2)
        elif lsta[ii] == 1 and lstb[ii] == 1:
            block2det.append(3)

    # need to also switch back to the physicist sign convention
    # so the state is correctly implemented in Block2
    which_occ = np.where(lsta == 1)[0]
    if len(which_occ) == 0:
        parity = 1.
    else:
        parity = (-1)**np.sum([np.sum(lstb[:int(ind)]) for ind in which_occ])

    return block2det, parity

def get_large_ci(dict_matr):
    '''
    Uses the cistring notation of PySCF to extract the symbolic
    representation of the state with the largest determinant coefficient.
    From the Mole object it extracts the necessary parameters like the
    number of alpha and beta electrons and number of molecular orbitals.

    Parameters
    ----------

    sparse_matr : scipy.sparse.coo_matrix or equivalent
        Matrix of CI coefficients representing some state wavefunction,
        obtained with some computational method provided.

    mol : pyscf.Mole() object
        Molecule class object representing the system, needed to extract nelec
        and norbitals.

    Returns
    -------

    [ float64, list, list, tuple ]
        A list with: the value of the CI coefficient, and the labels of the
        orbitals occupied, presented as separate lists for alpha and beta
        electrons, and finally a tuple of indices indicating location of CI
        coefficient.

    '''

    maxkey = max(dict_matr, key=lambda x: abs(dict_matr[x]))
    maxval = dict_matr[maxkey]

    res = get_state_repr(maxkey[0], maxkey[1])

    return [maxval, res[0], res[1], maxkey]

def get_state_repr(inda, indb):
    '''
    Uses the cistring notation of PySCF to extract the symbolic
    representation of a state with indices (inda, indb) in the FCI space.
    From the Mole object it extracts the necessary parameters like the
    number of alpha and beta electrons and number of molecular orbitals.

    Parameters
    ----------

    inda, indb : int
        Indices of an FCI state within the FCI matrix.

    Returns
    -------
    list, list
        occupancy vectors for alpha and beta electrons, labelling occupied
        molecular orbitals by a sequential number
    '''
    stra = bin(inda)
    strb = bin(indb)

    occa = []
    occb = []

    ii = 0
    for elem in stra[::-1]:
        if elem == "0":
            pass
        elif elem == "1":
            occa.append(ii)
        elif elem == "b":
            break
        ii += 1

    ii = 0
    for elem in strb[::-1]:
        if elem == "0":
            pass
        elif elem == "1":
            occb.append(ii)
        elif elem == "b":
            break
        ii += 1

    return occa, occb


def fcidict_to_fcivec(fcidict, mol):
    '''
    Given a wavefunction in Overlapper's fcidict notation, and molecule,
    this function returns the wfn in fcivec format of PySCF.
    '''

    ncas, nelec_a, nelec_b = get_mol_attrs(mol)

    Na, Nb = int(comb(ncas, nelec_a)), int(comb(ncas, nelec_b))
    fcivec = np.zeros((Na, Nb), dtype=float)

    for (keya, keyb), coeff in fcidict.items():
        inda = str2addr(ncas, nelec_a, keya)
        indb = str2addr(ncas, nelec_b, keyb)
        fcivec[inda, indb] = coeff

    return fcivec


def jobs_on_rank(jobs, worldsize, rank):
    # add average load balancing
    np.random.seed(24) 
    indices = np.arange(len(jobs), dtype=int)
    np.random.shuffle(indices)
    shuffled_indices = np.array_split(indices, worldsize)[rank].tolist()
    return np.array(jobs, dtype=object)[shuffled_indices].tolist()


def _excitations(electrons, orbitals):
    r"""Generate all possible single and double excitations from a Hartree-Fock reference state.

    This function is a more performant version of ``qchem.excitations``, where the order of the
    generated excitations is consistent with PySCF.

    Single and double excitations can be generated by acting with the operators
    :math:`\hat T_1` and :math:`\hat T_2` on the Hartree-Fock reference state:

    .. math::

        && \hat{T}_1 = \sum_{p \in \mathrm{occ} \\ q \in \mathrm{unocc}}
        \hat{c}_q^\dagger \hat{c}_p \\
        && \hat{T}_2 = \sum_{p>q \in \mathrm{occ} \\ r>s \in
        \mathrm{unocc}} \hat{c}_r^\dagger \hat{c}_s^\dagger \hat{c}_p \hat{c}_q.


    In the equations above the indices :math:`p, q, r, s` run over the
    occupied (occ) and unoccupied (unocc) *spin* orbitals and :math:`\hat c` and
    :math:`\hat c^\dagger` are the electron annihilation and creation operators,
    respectively.

    Args:
        electrons (int): number of electrons
        orbitals (int): number of spin orbitals

    Returns:
        tuple(list, list): lists with the indices of the spin orbitals involved in the excitations

    **Example**

    >>> electrons = 2
    >>> orbitals = 4
    >>> _excitations(electrons, orbitals)
    ([[0, 2], [0, 3], [1, 2], [1, 3]], [[0, 1, 2, 3]])
    """
    singles_p, singles_q = [], []
    doubles_pq, doubles_rs = [], []

    for i in range(electrons):
        singles_p += [i]
        doubles_pq += [[k, i] for k in range(i)]
    for j in range(electrons, orbitals):
        singles_q += [j]
        doubles_rs += [[k, j] for k in range(electrons, j)]

    singles = [[p] + [q] for p in singles_p for q in singles_q]
    doubles = [pq + rs for pq in doubles_pq for rs in doubles_rs]

    return singles, doubles

def _excited_configurations(electrons, orbitals, excitation):
    r"""Generate excited configurations from a Hartree-Fock reference state.

    This function generates excited configurations in the form of integers representing a binary
    string, e.g., :math:`|1 1 0 1 0 0 \rangle` is represented by :math:`int('110100', 2) = 52`.

    The excited configurations are generated from a Hartree-Fock (HF) reference state. The HF state
    is assumed to have the form :math:`|1 1 ...1 0 ... 0 0 \rangle` where the number of :math:`1`
    and :math:`0` elements are the number of occupied and unoccupied spin orbitals, respectively.
    The string representation of the state is obtained by converting the occupation-number vector to
    a string, e.g., ``111000`` to represent :math:`|1 1 1 0 0 0 \rangle.

    Each excited configuration has a sign, :math:`+1` or :math:`-1`, that is obtained by reordering
    the creation operators.

    Args:
        electrons (int): number of electrons
        orbitals (int): number of spin orbitals
        excitation (int): number of excited electrons

    Returns:
        tuple(list, list): lists of excited configurations and signs obtained by reordering the
         creation operators

    **Example**

    >>> electrons = 3
    >>> orbitals = 5
    >>> excitation = 2
    >>> _excited_configurations(electrons, orbitals, excitation)
    ([28, 26, 25], [1, -1, 1])
    """
    if excitation not in [1, 2]:
        raise ValueError(
            "Only single (excitation = 1) and double (excitation = 2) excitations are supported."
        )

    hf_state = np.array(np.where(np.arange(orbitals) < electrons, 1, 0))
    singles, doubles = _excitations(electrons, orbitals)
    states, signs = [], []

    if excitation == 1:
        for s in singles:
            state = hf_state.copy()
            state[s] = state[[s[1], s[0]]]  # apply single excitation
            states += [state]
            signs.append((-1) ** len(range(s[0], electrons - 1)))

    if excitation == 2:
        for d in doubles:
            state = hf_state.copy()
            state[d] = state[[d[2], d[3], d[0], d[1]]]  # apply double excitation
            states += [state]
            order_pq = len(range(d[0], electrons - 1))
            order_rs = len(range(d[1], electrons - 1))
            signs.append((-1) ** (order_pq + order_rs + 1))

    states_str = ["".join([str(i) for i in state]) for state in states]
    states_int = [int(state[::-1], 2) for state in states_str]

    return states_int, signs

def _sitevec_to_fock(det, format):
    r"""Covert a Slater determinant from site vector to occupation number vector representation.

    Args:
        det (list): determinant in site vector representation
        format (str): the format of the determinant

    Returns:
        tuple: tuple of integers representing binaries that correspond to occupation vectors in
            alpha and beta spin sectors

    **Example**

    >>> det = [1, 2, 1, 0, 0, 2]
    >>> _sitevec_to_fock(det, format = 'dmrg')
    (5, 34)

    >>> det = ["a", "b", "a", "0", "0", "b"]
    >>> _sitevec_to_fock(det, format = 'shci')
    (5, 34)
    """

    if format == "dmrg":
        format_map = {0: "00", 1: "10", 2: "01", 3: "11"}
    elif format == "shci":
        format_map = {"0": "00", "a": "10", "b": "01", "2": "11"}

    strab = [format_map[key] for key in det]

    stra = "".join(i[0] for i in strab)
    strb = "".join(i[1] for i in strab)

    inta = int(stra[::-1], 2)
    intb = int(strb[::-1], 2)

    return inta, intb

def verify_active_space(mol, ncas, nelecas):
    spin = mol.spin
    norb, nelec_a, nelec_b = get_mol_attrs(mol)
    ncore = nelec_a + nelec_b - nelecas[0] - nelecas[1]
    if not ncore % 2 == 0:
        raise ValueError(
            f"Only even numbers of electrons in the core are allowed."
        )
    if not (norb - ncas == ncore // 2):
        raise ValueError(
            f"Number of active electrons is inconcistent with the number of active orbitals."
        )
    if not (nelecas[0] - nelecas[1] == spin):
        raise ValueError(
            f"Active electrons {nelecas} must agree with molecule spin {spin} used in HF calculation."
        )


def _make_mock_hf(occ_vec, h1e, eri):
    r"""Creates a mock Hartree-Fock object to allow to execute 
    all PySCF-derived methods while explicitly passing the 
    one- and two-electron integral matrices and an occupation
    number vector. Whether RHF or ROHF is used is determined by
    the number of electrons being even or odd, respectively.

    Args:
        occ_vec (list or nd.array): Array with occupations, e.g. [2,2,1,0,0] for 5 electrons occupying the lowest of 10 spatial orbitals
        h1e (nd.array): one-electron integrals as square array
        eri (nd.array): two-electron integrals as four-index array, with 
            full 8-fold symmetry (real wavefunctions)

    Returns:
        hf_mock (object): PySCF Hartree-Fock object 
            (restricted or unrestricted, depending on occ_vec)
    """

    # create a mock molecule to pass to the mock HF object
    mol_mock = gto.M()
    # determine number of molecular orbitals and electrons from occ_vec
    mol_mock.nao = len(occ_vec)
    mol_mock.nelectron = np.sum(np.array(occ_vec))
    # count the unpaired electrons
    mol_mock.spin = np.sum([x for x in occ_vec if x == 1])
    # make sure this returns zero to avoid arbitrary offsets
    mol_mock.energy_nuc = lambda *args: 0

    # create a mock HF object
    if mol_mock.spin == 0:
        hf_mock = scf.RHF(mol_mock)
        # assign the occupation of orbitals
        hf_mock.mo_occ = np.array(occ_vec)
        # instantiate the molecular-to-atomic integrals as identity
        hf_mock.mo_coeff = np.eye(h1e.shape[0])
        # assign one-electron integrals
        hf_mock.get_hcore = lambda *args: h1e
        # assign two-electron integrals -- ensure they are 
        hf_mock._eri = eri

    elif mol_mock.spin != 0:
        hf_mock = scf.UHF(mol_mock)
        # assign the occupation of orbitals
        hf_mock.mo_occ = np.array(occ_vec)
        # instantiate the molecular-to-atomic integrals as identity
        hf_mock.mo_coeff = np.stack((np.eye(mol_mock.nao),\
                                     np.eye(mol_mock.nao)))
        # assign one-electron integrals
        hf_mock.get_hcore = lambda *args: h1e
        # assign two-electron integrals -- ensure they are 
        hf_mock._eri = eri

    return hf_mock
