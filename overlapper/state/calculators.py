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



from pyscf import scf, mcscf, mrpt, ci, cc
import numpy as np
import copy
from pyscf.ci.cisd import to_fcivec as to_fcivec_rhf
from pyscf.ci.cisd import amplitudes_to_cisdvec as amplitudes_to_cisdvec_rhf
from pyscf.ci.ucisd import to_fcivec as to_fcivec_uhf
from pyscf.ci.ucisd import amplitudes_to_cisdvec as amplitudes_to_cisdvec_uhf
from pyscf.fci.spin_op import spin_square0, spin_square

try:
    from pyscf.shciscf import shci
    from overlapper.utils.shci_utils import getinitialStateSHCI, get_dets_coeffs_output
except:
    print("Need to install dice and PySCF shciscf extension -- see the docs.")

try:
    from block2 import VectorUInt8
    from pyblock2._pyscf.ao2mo import integrals as itg
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes
except:
    print("Need to install Block2 -- see the docs.")

from overlapper.utils.utils import get_mol_attrs, verify_active_space


def do_hf(mol, hftype, verbose=0):
    r"""Execute PySCF's Hartree-Fock solvers.

    Args:
        mol (object): PySCF Molecule object
        hftype (str): String specifying the type of HF calculation to do. Currently supported are
            rhf, uhf and rohf
        verbose (int): Integer specifying the verbosity level (passed directly into PySCF's solver).

    Returns:
        solver_hf (object): PySCF Hartree-Fock Solver object
        e (array): HF energy
        ss (array): Spin-squared ($S^2$) of the output wavefunction
        sz (array): Spin projection ($S_z$) of the output wavefunction
    """

    if hftype == "rhf":
        assert mol.spin % 2 == 0, (
            f"RHF inconsistent with open-shell spins. " f"Pick UHF, or add only closed-shell spins."
        )
        solver_hf = scf.RHF(mol).run(verbose=verbose)
    elif hftype == "rohf":
        solver_hf = scf.ROHF(mol).run(verbose=verbose)
    elif hftype == "uhf":
        solver_hf = scf.UHF(mol).run(verbose=verbose)
    else:
        raise ValueError(
            "Unknown Hartree-Fock calculation type. Allowed types "
            "are restricted (rhf), unrestricted (uhf), and restricted open-shell (rohf)."
        )

    if verbose > 0:
        solver_hf.analyze()

    e = np.atleast_1d(solver_hf.e_tot)

    ss, mult = solver_hf.spin_square()
    ss = np.array([ss])
    sz = np.array([mult - 1])

    return solver_hf, e, ss, sz


def do_cisd(solver_hf, nroots=1, verbose=0):
    r"""Execute PySCF's configuration interaction with singles and doubles (CISD) solver.

    Args:
        solver_hf (object): PySCF's Hartree-Fock Solver object.
        nroots (int): Number of low-energy eigenstates to solve for.
        verbose (int): Integer specifying the verbosity level (passed directly into PySCF's solver).

    Returns:
        solver_cisd (object): PySCF's CISD Solver object.
        e (array): Energies of `nroots` lowest energy eigenstates.
        ss (array): Spin-squared ($S^2$) of `nroots` lowest energy eigenstates.
        sz (array): Spin projection ($S_z$) of `nroots` lowest energy eigenstates.
    """

    norb, nelec_a, nelec_b = get_mol_attrs(solver_hf.mol)
    nelec = nelec_a + nelec_b
    ss = []
    sz = []

    hftype = str(solver_hf.__str__)

    if "uhf" in hftype.lower():
        solver_cisd = ci.UCISD(solver_hf)
    elif "rhf" in hftype.lower():
        solver_cisd = ci.RCISD(solver_hf)
    solver_cisd.nroots = nroots
    solver_cisd.run(verbose=verbose)

    e = np.atleast_1d(solver_cisd.e_tot)

    if nroots > 1:
        for ii in range(nroots):
            try:
                if "rhf" in hftype.lower():
                    fcivec = to_fcivec_rhf(solver_cisd.ci[ii], norb, nelec)
                    ssval, multval = spin_square0(fcivec, norb, nelec)
                elif "uhf" in hftype.lower():
                    fcivec = to_fcivec_uhf(solver_cisd.ci[ii], norb, (nelec_a, nelec_b))
                    ssval, multval = spin_square(
                        fcivec,
                        norb,
                        (nelec_a, nelec_b),
                        mo_coeff=solver_hf.mo_coeff,
                        ovlp=solver_hf.get_ovlp(),
                    )
                ss.append(ssval)
                sz.append(multval - 1)
            except ValueError:  # if wavefunction is too big
                ss.append("N/A")
                sz.append(solver_hf.mol.spin)
    else:
        try:
            if "rhf" in hftype.lower():
                fcivec = to_fcivec_rhf(solver_cisd.ci, norb, nelec)
                ssval, multval = spin_square0(fcivec, norb, nelec)
            elif "uhf" in hftype.lower():
                fcivec = to_fcivec_uhf(solver_cisd.ci, norb, (nelec_a, nelec_b))
                ssval, multval = spin_square(
                    fcivec,
                    norb,
                    (nelec_a, nelec_b),
                    mo_coeff=solver_hf.mo_coeff,
                    ovlp=solver_hf.get_ovlp(),
                )
            ss.append(ssval)
            sz.append(multval - 1)
        except ValueError:  # if wavefunction too big
            ss.append("N/A")
            sz.append(solver_hf.mol.spin)

    return solver_cisd, e, np.array(ss), np.array(sz)


def do_ccsd(solver_hf, nroots=1, max_cycle=50, diis_space=10, verbose=0):
    r"""Execute PySCF's coupled cluster with singles and doubles (CCSD) solver.
    Unlike the other solvers, CCSD from PySCF has no capability to return wavefunctions
    for excited states, only energies.

    Args:
        solver_hf (object): PySCF's Hartree-Fock Solver object.
        nroots (int): Number of low-energy eigenstates to solve for.
        max_cycle (int): Maximum number of iterations in the CCSD procedure.
        diis_space (int): Number of solutions to use for the direct inversion
            in the iterative subspace (DIIS) procedure.
        verbose (int): Integer specifying the verbosity level (passed directly into PySCF's solver).

    Returns:
        solver_ccsd (object): PySCF's CCSD Solver object.
        e (array): Energies of `nroots` lowest energy eigenstates.
        ss (array): Spin-squared ($S^2$) of lowest energy eigenstate. If `nroots > 1`, remainder are marked N/A.
        sz (array): Spin projection ($S_z$) of lowest energy eigenstate. If `nroots > 1`, remainder are marked N/A.
    """

    norb, nelec_a, nelec_b = get_mol_attrs(solver_hf.mol)
    nelec = nelec_a + nelec_b

    hftype = str(solver_hf.__str__)

    if "uhf" in hftype.lower() or "rohf" in hftype.lower():
        solver_ccsd = cc.UCCSD(solver_hf).run(
            verbose=verbose, max_cycle=max_cycle, diis_space=diis_space
        )
        ss, mult = solver_ccsd.spin_square()
    elif "rhf" in hftype.lower():
        assert solver_hf.mol.spin == 0, f"Cannot run RCCSD with nonzero Sz."
        solver_ccsd = cc.CCSD(solver_hf).run(
            verbose=verbose, max_cycle=max_cycle, diis_space=diis_space
        )
        cisdvec = amplitudes_to_cisdvec_rhf(1.0, solver_ccsd.t1, solver_ccsd.t2)
        fcivec = to_fcivec_rhf(cisdvec, norb, nelec)
        from pyscf.fci.spin_op import spin_square0

        ss, mult = spin_square0(fcivec, norb, nelec)

    e = [solver_ccsd.e_tot] + ["N/A"] * (nroots - 1)
    ss = [ss] + ["N/A"] * (nroots - 1)
    sz = [mult - 1] + ["N/A"] * (nroots - 1)

    return solver_ccsd, np.array(e), np.array(ss), np.array(sz)


def do_casci(solver_hf, ncas, nelecas, nroots=1, mem=5000, maxiter=2000, verbose=0):
    r"""Execute PySCF's complete active space configuration interaction (CASCI) solver.

    Args:
        solver_hf (object): PySCF's Hartree-Fock Solver object
        ncas (int): Number of active orbitals
        nelecas (tuple(int, int)): Number of active electrons in spin-up (alpha) and
            spin-down (beta) sectors. A warning is raised if `spin` variable of mf.mol
            disagrees with the number of active electrons.
        nroots (int): Number of low-energy eigenstates to solve for.
        mem (int): Memory allocated to the solver, in megabytes.
        maxiter (int): Maximum allowed number of iterations for the Davidson iteration.
        verbose (int): Integer specifying the verbosity level (passed directly into PySCF's solver).

    Returns:
        solver_casci (object): PySCF CASCI Solver object
        e (array): Energies of `nroots` lowest energy eigenstates.
        ss (array): Spin-squared ($S^2$) of `nroots` lowest energy eigenstates.
        sz (array): Spin projection ($S_z$) of `nroots` lowest energy eigenstates.
    """

    verify_active_space(solver_hf.mol, ncas, nelecas)

    # infer restricted or unrestricted type from the passed HF solver
    hftype = str(solver_hf.__str__)
    if "rhf" in hftype.lower() or "rohf" in hftype.lower():
        solver_casci = mcscf.CASCI(solver_hf, ncas, nelecas)
        solver_casci.fix_spin_(ss=solver_hf.mol.spin)
    elif "uhf" in hftype.lower():
        solver_casci = mcscf.UCASCI(solver_hf, ncas, nelecas)
    else:
        raise ValueError("Only RHF/ROHF and UHF solvers are supported.")

    # make sure input orbitals are same as output -- turn off
    # canonicalization and natural orbitals
    solver_casci.canonicalization = False
    solver_casci.natorb = False
    solver_casci.fcisolver.nroots = nroots
    solver_casci.fcisolver.spin = solver_hf.mol.spin
    solver_casci.fcisolver.max_cycle = maxiter
    solver_casci.fcisolver.max_memory = mem

    solver_casci.run(verbose=verbose)
    if verbose > 2:
        solver_casci.analyze()

    e = np.atleast_1d(solver_casci.e_tot)

    ss, sz = [], []
    try:
        if nroots > 1:
            for ii in range(len(solver_casci.ci)):
                ssval, multval = solver_casci.fcisolver.spin_square(
                    solver_casci.ci[ii], ncas, nelecas
                )
                ss.append(ssval)
                sz.append(multval - 1)
        else:
            ssval, multval = solver_casci.fcisolver.spin_square(solver_casci.ci, ncas, nelecas)
            ss.append(ssval)
            sz.append(multval - 1)
    except ValueError:  # if wavefunction is too big
        ss = ["N/A"] * nroots
        sz = [solver_hf.mol.spin] * nroots

    return solver_casci, e, np.array(ss), np.array(sz)


def do_mrpt(solver_casci, nroots=1, verbose=0):
    r"""Execute PySCF's multireference perturbation theory (MRPT) solver.

    Args:
        solver_casci (object): PySCF's CASCI Solver object
        nroots (int): Number of low-energy eigenstates to solve for.
        verbose (int): Integer specifying the verbosity level (passed directly into PySCF's solver).

    Returns:
        solver_mrpt (object): A copy of the input PySCF CASCI Solver object, whose
            wavefunction has been modified from the MRPT procedure.
        e (array): Energies of `nroots` lowest energy eigenstates.
        ss (array): Spin-squared ($S^2$) of `nroots` lowest energy eigenstates.
        sz (array): Spin projection ($S_z$) of `nroots` lowest energy eigenstates.
    """

    if len(solver_casci.mo_coeff.shape) > 2:
        raise NotImplementedError("MRPT in PySCF only works with restricted HF solver.")

    solver_mrpt = copy.copy(solver_casci)
    solver_mrpt.verbose = verbose

    solver_mrpt.fcisolver.nroots = nroots
    if not len(np.atleast_1d(solver_mrpt.e_tot)) == nroots:
        raise ValueError("CASCI must be run with the same value for nroots as MRPT.")

    e = []
    for root in range(nroots):
        e.append(
            np.atleast_1d(solver_mrpt.e_tot)[root] + mrpt.NEVPT(solver_mrpt, root=root).kernel()
        )

    # infer active space from CASCI solver
    ncas = solver_mrpt.ncas
    nelecas_a, nelecas_b = solver_mrpt.nelecas

    ss, sz = [], []
    if nroots > 1:
        for ii in range(nroots):
            ssval, multval = solver_mrpt.fcisolver.spin_square(
                solver_mrpt.ci[ii], ncas, (nelecas_a, nelecas_b)
            )
            ss.append(ssval)
            sz.append(multval - 1)
    else:
        ssval, multval = solver_mrpt.fcisolver.spin_square(
            solver_mrpt.ci, ncas, (nelecas_a, nelecas_b)
        )
        ss.append(ssval)
        sz.append(multval - 1)

    return solver_mrpt, np.array(e), np.array(ss), np.array(sz)


def do_shci(
    solver_hf,
    ncas,
    nelecas,
    eps1_arr,
    n_sweeps,
    output_file,
    nroots=1,
    mpiprefix="",
    n_threads=1,
    dets_budget=200,
    tol=1e-6,
    verbose=0,
):
    r"""Execute the semistochastic heat-bath configuration interaction (SHCI)
    solver from the Dice library via the SHCI-to-PySCF interface.

    Args:
        solver_hf (object): PySCF's Hartree-Fock Solver object. Only RHF solvers are admissible with Dice's SHCI.
        ncas (int): Number of active orbitals
        nelecas (tuple(int, int)): Number of active electrons in spin-up (alpha) and
            spin-down (beta) sectors. The `spin` variable of mf.mol is used to override the
            spin-up vs spin-down split if they give an incorrect spin.
        eps1_arr (list(float)): Array of cutoff values for selection of determinants into the variational space. The smaller the cutoff, the more determinants are included.
        n_sweeps (list(int)): Array of maximum allowed iterations of the SHCI algorithm to be executed with each of the eps1_arr values. Must be same length as eps1_arr.
        output_file (path): Path for the output file of SHCI calculations.
        nroots (int): Number of low-energy eigenstates to solve for.
        mpiprefix (str): String to specify the mpi command to use when running SHCI, e.g. to run with 4 MPI processes, set mpiprefix="mprirun -np 4" or equivalent for your MPI configuration.
        n_threads (int): Number of threads to use for multiprocessing of the algorithm within a single shared-memory node.
        dets_budget (int): Number of determinants whose coefficients are printed in the output file and are used to build the wavefunction dictionary.
        tol (float): Convergence tolerance of the energy for the variational part of the SHCI algorithm.
        verbose (int): Integer specifying the verbosity level (passed directly into PySCF's solver).

    Returns:
        wfs (tuple(list[str],array[float]) of list of such tuples): Tuples containing as the first element all the Slater determinants of a given wavefunction, and as the second the corresponding coefficients.
        e (array): Energies of `nroots` lowest energy eigenstates.
        ss (array): Spin-squared ($S^2$) of `nroots` lowest energy eigenstates.
        sz (array): Spin projection ($S_z$) of `nroots` lowest energy eigenstates."""

    verify_active_space(solver_hf.mol, ncas, nelecas)

    hftype = str(solver_hf.__str__)
    if "rhf" in hftype.lower():
        solver_shci = mcscf.CASCI(solver_hf, ncas, nelecas)
    else:
        raise ValueError("Only RHF solvers are supported by Dice/SHCI.")

    initialStates = getinitialStateSHCI(solver_hf, nelecas)

    solver_shci.fcisolver = shci.SHCI(solver_hf.mol)
    # general
    solver_shci.canonicalization = False
    solver_shci.fcisolver.internal_rotation = False
    solver_shci.fcisolver.mpiprefix = mpiprefix
    solver_shci.fcisolver.num_thrds = n_threads
    solver_shci.fcisolver.spin = solver_hf.mol.spin
    solver_shci.fcisolver.dE = tol
    solver_shci.fcisolver.initialStates = initialStates
    solver_shci.fcisolver.outputFile = output_file
    # variational
    solver_shci.fcisolver.nroots = nroots
    solver_shci.fcisolver.sweep_iter = np.atleast_1d(n_sweeps)
    solver_shci.fcisolver.sweep_epsilon = eps1_arr
    # perturbative
    solver_shci.fcisolver.stochastic = False
    solver_shci.fcisolver.nPTiter = 0  # no PT correction
    # misc
    solver_shci.fcisolver.DoRDM = False
    solver_shci.fcisolver.DoSpinRDM = False
    solver_shci.fcisolver.shci_extra_keyword = [f"printBestDeterminants {dets_budget}"]

    solver_shci.run(verbose=verbose)

    e = np.atleast_1d(solver_shci.e_tot)
    ssval, mult = solver_shci.fcisolver.spin_square(solver_shci.ci, ncas, nelecas)
    ss = np.atleast_1d(ssval)
    sz = np.atleast_1d(mult) - 1

    # parse the output file to extract the determinants and coefficients
    wfs = []
    for state in range(nroots):
        dets, coeffs = get_dets_coeffs_output(output_file, state=state)
        wfs.append((dets, coeffs))

    return wfs, e, np.array(ss), np.array(sz)


def do_dmrg(
    solver_hf,
    ncas,
    nelecas,
    schedule,
    mem=6 * 1024**3,
    dot=2,
    workdir="dmrg_calc_tmp",
    nroots=1,
    n_threads=1,
    tol=1e-6,
    restart_ket=None,
    smp_tol=1e-3,
    eshift=0,
    verbose=0,
    occs=None,
    reorder=None,
    return_objects=False,
    mpssym="sz",
    proj_state=None,
    proj_weight=None,
):
    r"""Execute the density-matrix renormalization group (DMRG) solver
    from the Block2 library, wrapping Block2's Python bindings.

    The DMRG method in Overlapper is special. DMRG is also used for the calculation of 
    Hamiltonian moments with respect to a given wavefunction, and for running the 
    resolvent method to obtain the state's energy distribution. For this reason, `do_dmrg()`
    can be run in two modes: a) normal mode to generate an initial state, and b) assessment
    mode (triggered with return_objects=True) where the DMRGDriver, MPS and MPO are returned 
    for further operations (such as computing moments using DMRG or executing the resolvent 
    method).

    Args:
        solver_hf (object): PySCF's Hartree-Fock Solver object.
        ncas (int): Number of active orbitals
        nelecas (tuple(int, int)): Number of active electrons in spin-up (alpha) and
            spin-down (beta) sectors. The `spin` variable of mf.mol is used to override the
            spin-up vs spin-down split if they give an incorrect spin.
        schedule (list(list(int), list(int), list(float), list(float))): Schedule of DMRG calculations: the first array is a list of bond dimensions; the second is a list of the number of sweeps to execute at each corresponding bond dimension; the third is a list of noises to be added to the calculation at each bond dimension; the fourth is a list of tolerances for the Davidson iteration in the sweeps.
        mem (list(int)): Total memory allocated to the solver. Default is 6 GB.
        dot (int): Type of MPS to execute the calculation for: could be 1 or 2.
        workdir (path): Path to scratch folder for use during the calculation.
        nroots (int): Number of low-energy eigenstates to solve for.
        n_threads (int): Number of threads to use for multiprocessing of the algorithm within a single shared-memory node.
        tol (float): Convergence tolerance criterion for the energy.
        restart_ket (str): If it is desired to restart a previous DMRG calculation, one can specify the tag of the corresponding MPS to restart from.
        eshift (float): Value to shift the constant of the energy. Can be used to adjust against nuclear energy during Hamiltonian moment calculation.
        smp_tol (float): Tolerance for reconstructing the MPS into a list of Slater determinants: all determinants with coefficients below this value will be neglected.
        verbose (int): Integer specifying the verbosity level.
        occs (list(int)): An initial guess for the list of occupancies of the orbitals: entires can be 0 and 1 if specified in terms of spin-orbitals (alpha and beta orbitals alternate), or 0, 1, 2, 3 if specified in terms of spatial orbitals (1 is spin-up, 2 is spin-down, 3 is double occupancy).
        reorder (None or boolean): Specifies whether reordering of the orbitals will be done before DMRG: if True, reordering is done according to the fiedler approach.
        return_objects (boolean): Whether to return the results of the calculation, or the Block2 objects (DriverDMRG, MPS and MPO).
        mpssym (str): Whether to run DMRG in SU(2) symmetry mode ("su2") or SZ symmetry mode ("sz").
        proj_state (list(MPS)): Advanced users only -- an MPS to project against during DMRG. Can be used to stabilize particular spin states
        proj_weight (list(float)): Advanced users only -- weights for the projection.

    Returns:
        wfs (tuple(list[int],array[float]) of list of such tuples): Tuples containing as the first element all the Slater determinants of a given wavefunction, and as the second the corresponding coefficients.
        e (array): Energies of `nroots` lowest energy eigenstates.
        ss (array): Spin-squared ($S^2$) of `nroots` lowest energy eigenstates.
        sz (array): Spin projection ($S_z$) of `nroots` lowest energy eigenstates."""

    verify_active_space(solver_hf.mol, ncas, nelecas)
    norb, nelec_a, nelec_b = get_mol_attrs(solver_hf.mol)
    ncore = (nelec_a + nelec_b - nelecas[0] - nelecas[1]) // 2

    hftype = str(solver_hf.__str__)

    if mpssym == "sz":
        SpinSym = SymmetryTypes.SZ
    elif mpssym == "su2":
        if not ("rhf" in hftype.lower()):
            raise ValueError("SU2 MPS calculation only possible with RHF molecular integrals.")
        SpinSym = SymmetryTypes.SU2

    if "rhf" in hftype.lower():
        ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(
            solver_hf, ncore, ncas, g2e_symm=8
        )
    elif "uhf" in hftype.lower() or "rohf" in hftype.lower():
        ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_uhf_integrals(
            solver_hf, ncore, ncas, g2e_symm=8
        )

    driver = DMRGDriver(scratch=workdir, symm_type=SpinSym, n_threads=n_threads, stack_mem=mem)

    driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)
    mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore+eshift, reorder=reorder, iprint=verbose)

    if restart_ket is None:
        ket = driver.get_random_mps(
            tag="GS", bond_dim=schedule[0][0], occs=occs, nroots=nroots, dot=dot
        )
    else:
        ket = driver.load_mps(restart_ket)

    # need to reorder the occupations provided
    if not (occs is None) and not (reorder is None):
        if len(occs) == ncas:
            occs = np.array(occs)[driver.reorder_idx]
        elif len(occs) == 2 * ncas:
            occs_a = (np.array(occs)[0 : len(occs) : 2])[driver.reorder_idx]
            occs_b = (np.array(occs)[1 : len(occs) : 2])[driver.reorder_idx]
            occs = []
            for ii in range(len(occs_a)):
                occs.append(occs_a[ii])
                occs.append(occs_b[ii])

    bond_dims, n_sweeps, noises, thresholds = schedule
    for ii, M in enumerate(bond_dims):
        Mvals = [M] * n_sweeps[ii]
        noisevals = [noises[ii]] * n_sweeps[ii]
        thrdsvals = [thresholds[ii]] * n_sweeps[ii]
        energies = driver.dmrg(
            mpo,
            ket,
            n_sweeps=n_sweeps[ii],
            bond_dims=Mvals,
            noises=noisevals,
            thrds=thrdsvals,
            iprint=verbose,
            tol=tol,
            proj_mpss=proj_state,
            proj_weights=proj_weight,
        )

    wfs = []
    e = []
    ss = []
    sz = []
    smpo = driver.get_spin_square_mpo(iprint=verbose)

    if nroots > 1:
        for ii in range(nroots):
            aux_ket = driver.split_mps(ket, ii, f"state{ii}")
            aux_ket_e = driver.expectation(aux_ket, mpo, aux_ket)
            e.append(aux_ket_e)
            # this part reconstructs the Slater determinants from the GS MPS
            dets, coeffs = driver.get_csf_coefficients(aux_ket, cutoff=smp_tol, iprint=verbose)
            # re-attach the frozen core electrons
            dets = [[3] * ncore + det.tolist() for det in dets]
            wfs.append((dets, coeffs))

            ### compute the S^2 spin number
            ss.append(driver.expectation(aux_ket, smpo, aux_ket))
            sz.append(solver_hf.mol.spin)
    else:
        ket_e = driver.expectation(ket, mpo, ket)
        e.append(ket_e)
        # this part reconstructs the Slater determinants from the GS MPS
        dets, coeffs = driver.get_csf_coefficients(ket, cutoff=smp_tol, iprint=verbose)
        # re-attach the frozen core electrons
        dets = [[3] * ncore + det.tolist() for det in dets]
        wfs.append((dets, coeffs))

        ### compute the S^2 spin number
        ss.append(driver.expectation(ket, smpo, ket))
        sz.append(solver_hf.mol.spin)

    if return_objects:
        return mpo, ket, driver

    return wfs, np.atleast_1d(e), np.array(ss), np.array(sz)
