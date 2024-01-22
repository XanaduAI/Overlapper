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


from overlapper.state import do_hf, do_casci, casci_state
from overlapper.utils.utils import get_mol_attrs
from overlapper.utils.wf_utils import wf_overlap
import numpy as np

def spectrum_overlaps(wf, fcisolver, tol=1e-4):
    r"""Computes the overlap <wfn|E> for a given input wavefunction 
    (in sparse dictionary format) and a given exact set of 
    eigenstates |E> of the Hamiltonian, stored as .ci attribute 
    of an fci or casci object from PySCF.
    
    Args:
        wf (dict): Overlapper's sparse dictionary format for wavefunctions.
        fcisolver (object): PySCF's CASCI Solver object for the full space, 
            with a number of exact eigenstates already solved for. Typically
            at least 100-200 are needed for optimal results.
        tol (float): Tolerance cutoff for wavefunction reconstruction from 
            `fcisolver`: determinants with coefficients below this cutoff 
            are neglected.

    Returns:
        overlaps (array[float]): Array of overlaps of `wf` with the 
            spectrum of the Hamiltonian.
    """
    overlaps = np.zeros(len(fcisolver.e_tot))
    for ii in range(len(fcisolver.e_tot)):
        wf_casci = casci_state(fcisolver, state=ii, tol=tol)
        overlaps[ii] = wf_overlap(wf, wf_casci)
    print(f"Exact ground-state overlap^2: {overlaps[0]**2}")
    return overlaps


def energy_pdf(mol, hftype, wf, eta=0.05, nroots=100, verbose=0, tol=1e-4):
    r"""Given a molecule and an initial state `wf`, returns a generator for 
    the probability density P(E)dE of that state. 
    
    It is obtained by finding the exact spectrum of the Hamiltonian using 
    FCI, calculating overlaps of the initial state with the exact spectrum,
    and then placing Lorentzian peaks at the exact eigenstate positions 
    with the weight equal to the square of the overlap, and broadening eta.
    
    Args:
        mol (dict): PySCF's `Molecule` object.
        hftype (str): Indicates what type of Hartree-Fock calculation to 
            run: "rhf" for restricted, "uhf" for unrestricted, and "rohf" for 
            restricted open-shell.
        wf (dict): Overlapper's sparse dictionary format for wavefunctions.
        eta (float): Lorentzian broadening factor for generating a 
            probability density. Default is 0.05 (for intermediate-level 
            broadening).
        nroots (int): Number of exact eigenstates obtained to build the energy 
            distribution. Default is 100 (for reasonable resolution).
        verbose (int): Integer specifying the verbosity level.
        tol (float): Tolerance cutoff for wavefunction reconstruction from 
            `fcisolver`: determinants with coefficients below this cutoff 
            are neglected.

    Returns:
        prob_density (func): Lambda function that can be evaluated at any 
            value of energy to give the energy distribution associated to `wf`.
        fci_energies (array(float)): List of the exact eigenstate energies.
    """
    
    # do HF to get orbitals
    hf, _, _, _ = do_hf(mol, hftype)

    # obtain exact spectrum
    norb, nelec_a, nelec_b = get_mol_attrs(mol)
    fcisolver, fci_energies, _, _ = do_casci(hf, norb, (nelec_a, nelec_b), \
                                nroots=nroots, verbose=verbose)

    # get overlaps with exact spectrum
    ovlps = spectrum_overlaps(wf, fcisolver, tol=tol)
    weights = ovlps**2
    # correct exact spectrum for nuclear energy
    e0 = fci_energies # - mol.energy_nuc()
    print(f"\nTotal projected weight of the wavefunction: {np.sum(weights):.3f}.")
    print(f"\nEnergy of the state from the energy distribution: "\
                f"{np.sum(fci_energies * ovlps**2) - mol.energy_nuc()}.")

    # build function to generate energy distribution
    prob_density = lambda e: np.sum( weights * (eta/np.pi) / ( (e-e0)**2 + eta**2 ) )
    return prob_density, fci_energies
