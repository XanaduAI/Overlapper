# Overlapper: Prepare and evaluate wavefunctions for quantum
# algorithms using computational chemistry techniques
# Copyright 2024 Xanadu Quantum Technologies Inc.
#
# Author: Kasra Hejazi <kasra.hejazi@xanadu.ai>
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
# The contents of this source file are original contributions to Overlapper, 
# with the exception of the function to_block2_dot_modified, which has been
# copied and modified from a GNU General Public License version 3.0 package
#  pyblock3 (pyblock3.block2.io.to_block2, code link: 
# https://github.com/block-hczhai/pyblock3-preview/blob/master/pyblock3/block2/io.py).
# The function has been modified for the specific use case of Overlapper. 
# The original source of the function to_block2_dot_modified is copyright of
# respective authors of pyblock3.
# Modifications of the source in to_block2_dot_modified is copyright of Xanadu.

import numpy as np

try:
    from pyblock2._pyscf.ao2mo import integrals as itg
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    from overlapper.utils.utils import str_to_block2
    from overlapper.utils.wf_utils import wf_overlap
    from overlapper.state import dmrg_state
except:
    print("Need to install Block2 -- see the docs.")


def sos_to_mps_block2(hf, wf, bond_dim, ipg=0, pg='c1', uhf=False, \
                                h1e=0, g2e=0, const_e=0, mu=0, flat=True, \
                                    non_flat_required=True, \
                                        second_dmrg_round=False, \
                                            pyblock3_compression_cutoff=1E-9, \
                                                print_progress=False):
    """
    Translates a sum of Slater determinants state into a MPS form in block2, 
    done through pyblock3.
    
    Args:
        hf (object): PySCF's Hartree-Fock Solver object

        wf (dict): state in Overlapper's sparse dictionary format

        bond_dim: bond dimension of the final MPS.
            If equal to the number of dets, the transformation is exact. 
            If a smaller bond dimension is obtained through pyblock3, 
            that smaller bond dimension+20 is used.
                    
        flat: Bool
            Whether more optimized C++ codes are used, checked only with True
            
        non_flat_required: Bool    
            Sometimes required for larger bond dimensions, safer to keep True
            
        second_dmrg_round: Bool
            Sometimes a more well behaved MPS is obtained through a second round of block2 DMRG

        pyblock3_compression_cutoff: float
            The cutoff used in pyblock3 compression. (Some actual compression might be possible if some modifications are considered here)
            
            Other standard quantum chemistry keywords can be used. 
            h1e and g2e can be chosen equal to zero as the integrals do not matter for the transformation.
    Returns:
        mps_to_b2: block2 MPS
            Note that this MPS is normalized
    """

    ### get block2 driver and decompose the Overlapper dict wavefunction ###
    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(hf)
    block2_driver = aux_block2driver(hf)
    dets, coeffs = wfdict_to_detscoeffs(wf, hf.mol.nao)

    ### pyblock3 sector ###
    hamil = pyblock3_hamil(pg, ncas, n_elec, spin, orb_sym, \
                                        ipg, uhf, h1e, g2e, const_e, mu, flat)
    
    occ_init = get_occ_from_det(dets[0])
    
    psi_init = get_psi_initial(hamil, occ_init)
    
    bitstring_init = occupations_bd1_to_bitstring(occupations(psi_init, ncas, hamil))
    
    indices = np.argsort(-abs( np.array(coeffs)) )
    coeffs_sorted = np.array(coeffs)[indices]
    
    bitstrings = []
    for j in range(len(dets)):
        bitstrings = bitstrings + [det_to_bit_string( dets[indices[j]] )]
        
    mps_sum = sum_slaters(bitstrings, coeffs_sorted, hamil, bitstring_init, psi_init, pyblock3_compression_cutoff, print_progress, non_flat_required=non_flat_required) 
    
    if print_progress:
        print()
    
    print("Bond dimensions IN PYBLOCK3:")
    print(mps_sum.show_bond_dims())
    
    bond_dim = min( mps_sum.bond_dim+20 , bond_dim)
    
    if non_flat_required:
        mps_sum = mps_sum.to_non_flat()
    
    impo_b2 = block2_driver.get_identity_mpo()
    
    mps_to_b2 = to_block2_dot_modified(mps_sum, tag='SOSKET')
    print("TRANSLATION TO BLOCK2 SUCCESSFUL.")
    
    block2_driver.dmrg(impo_b2, mps_to_b2, bond_dims=[bond_dim]*1 , n_sweeps=1, iprint=0)
    print("ONE ROUND OF DMRG SUCCESSFUL.")
    
    if second_dmrg_round:
        #More well behaved MPS obtained through DMRG
        aux_mps = block2_driver.get_random_mps(tag='SOSOUT', bond_dim=bond_dim, dot=2)
        block2_driver.multiply(aux_mps, impo_b2, mps_to_b2)
        block2_driver.dmrg(impo_b2, aux_mps, bond_dims=[bond_dim] , n_sweeps=30, proj_mpss=[mps_to_b2], proj_weights=[-10], iprint=0)

        #Checking whether the latter MPS matches the one previously found
        ket_exp = mps_to_b2.deep_copy("EXPECKET")
        bra_exp = aux_mps.deep_copy("EXPECBRA")
        if abs( block2_driver.expectation(bra_exp, impo_b2, ket_exp) ) > 0.99:
            mps_to_b2 = aux_mps
            print("SECOND ROUND OF DMRG SUCCESSFUL.")
        else:
            print("CAUTION: THE AUXILIARY PROJECTED DMRG (SECOND ROUND) DID NOT CONVERGE.")
        
        
    print("Maximum bond dimension of the block2 MPS = ", mps_to_b2.info.bond_dim)

    #################################
    ### some checks
    ket_exp = mps_to_b2.deep_copy("EXPECKET")
    bra_exp = mps_to_b2.deep_copy("EXPECBRA")
    print(f"Norm of the transformed block2 MPS = "
            f"{block2_driver.expectation(bra_exp, impo_b2, ket_exp)}")

    ### Slater decomposition after transformation
    tuple_after = block2_driver.get_csf_coefficients(mps_to_b2, cutoff=1e-5, iprint=0)
    wf_after = dmrg_state([tuple_after], tol=1e-5)
    ovlp = wf_overlap(wf_after, wf)
    print(f"Overlap between the converted and original wavefunctions is {ovlp:.4f}.")

    return mps_to_b2


    
####steps of the transformation

def aux_block2driver(hf):
    ### pyblock2 sector ###

    # get attributes
    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(hf)

    # initialize auxiliary driver for the conversion
    driver = DMRGDriver(scratch="dmrg_temp", symm_type=SymmetryTypes.SZ, \
                                                        stack_mem=int(6 * 1024**3))
    driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)
    mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=0)

    return driver

def wfdict_to_detscoeffs(wf, ncas):
    # split the SOS state into dets and coeffs
    dets, parities = np.array([ np.array(str_to_block2(key, ncas)) for key in wf.keys() ]).T
    coeffs = parities * np.array(list(wf.values()))
    return dets, coeffs

def pyblock3_hamil(pg, n_sites, n_elec, twos, orb_sym, ipg=0, uhf=False, h1e=0, g2e=0, const_e=0, mu=0, flat=True):
    
    from pyblock3.hamiltonian import Hamiltonian
    from pyblock3.fcidump import FCIDUMP

    fd = FCIDUMP(pg=pg, n_sites=n_sites, n_elec=n_elec, twos=twos, ipg=ipg, uhf=uhf,
                h1e=h1e, g2e=g2e, const_e=const_e, mu=mu, orb_sym=orb_sym)
    return Hamiltonian(fd, flat=flat)


def get_psi_initial(hamil, occ):

    impo_b3 = hamil.build_identity_mpo()
    from pyblock3.algebra.mpe import MPE
    
    psi_init = hamil.build_mps(10, occ=occ)
    MPE(psi_init, impo_b3, psi_init).dmrg(bdims=[1]*1, noises=[0], dav_thrds=[1E-5], iprint=-1, n_sweeps=1,tol=1e-10)
    
    psi_init = psi_init / np.sqrt( np.dot(psi_init , psi_init) )
    
    return psi_init


def sum_slaters(bitstrings, coeffs, hamil, bitstring_init, psi_init, cutoff, print_progress, non_flat_required=True):

    current_state = bitstring_init

    import copy
    mps_bd_1_aux = copy.deepcopy(psi_init)
    
    while current_state != bitstrings[0]:
        site_right, site_left, create_at_right, antisymmetry_phase, current_state = \
        mpo_properties_single_step(current_state , bitstrings[0])

        mps_bd_1_aux = single_move(mps_bd_1_aux, site_right, site_left, create_at_right, antisymmetry_phase, hamil)

    mps_sum = coeffs[0] * mps_bd_1_aux

    if print_progress:
        print("Printing the number of terms added:")
        print(0 , " , " , end="")
    
    for j in range(1,len(bitstrings)):
        if print_progress:
            print(j , " , " , end="")

        while current_state != bitstrings[j]:
            site_right, site_left, create_at_right, antisymmetry_phase, current_state = \
            mpo_properties_single_step(current_state , bitstrings[j])

            mps_bd_1_aux = single_move(mps_bd_1_aux, site_right, site_left, create_at_right, antisymmetry_phase, hamil)
        
        mps_sum = mps_sum + coeffs[j] * mps_bd_1_aux
        
        if ((j-1)%20 == 0) or (j==len(bitstrings)-1):
            if non_flat_required:
                mps_sum = mps_sum.to_non_flat()
            
            mps_sum, _ = mps_sum.compress(cutoff=cutoff)
            
            if non_flat_required:
                mps_sum = mps_sum.to_flat()

    return mps_sum


def to_block2_dot_modified(mps, center=None, basis=None, tag='KET', save_dir=None):
    """
    Translate pyblock3 MPS to block2 MPS. [This function has been copied from 
    pyblock3.block2.io.to_block2 in the package pyblock3, licensed 
    under GNU General Public License version 3.0 (code link: 
    https://github.com/block-hczhai/pyblock3-preview/blob/master/pyblock3/block2/io.py).
    The function has been modified for the specific use case of Overlapper. ]

    Args:
        mps : pyblock3 MPS
            More than one physical index is not supported.
            But fused index can be supported.
        center : int or None
            If not None, the pyblock3 MPS is already canonicalized
            at the given index of the center.
            If None (default), the pyblock3 MPS is transformed after
            canonicalization at site 0.
        basis : List(BondInfo) or None
            If None (default), the site basis will be constructed based on
            blocks in the MPS tensor, which may be incomplete if the
            bond dimension of the MPS is small.
            If not None, the given basis will be used.
        tag : str
            Tag of the block2 MPS. Default is "KET".
        save_dir : str or None
            If not None, the block2 MPS will be saved to the given dir.
            If None and the block2 global scratch is not set before
            entering this function, the block2 MPS will be saved to './nodex'.
            If None and the block2 global scratch is set, the block2 MPS will
            be saved to the current block2 global scratch folder.

    Returns:
        bmps : block2 MPS
            To inspect this MPS, please make sure that the block2 global
            scratch folder and stack memory are properly initialized.
    """
    from block2 import SZ, Global, VectorInt
    from block2.sz import StateInfo, MPSInfo, VectorStateInfo
    
    if mps.dtype == np.float32 or mps.dtype == np.float64 or mps.dtype == float:
        import block2.sz as bx, block2 as bf
    else:
        import block2.cpx.sz as bx, block2.cpx as bf
    frame_back = Global.frame
    inited = False
    if Global.frame is None or save_dir is not None:
        if save_dir is None:
            save_dir = './nodex'
        init_memory(isize=1 << 20, dsize=1 << 30, save_dir=save_dir)
        inited = True
    if save_dir is None:
        save_dir = Global.frame.save_dir
    if center is None:
        mps = mps.canonicalize(0)
        center = 0      
    mps = mps.to_non_flat()
    mps_infos = [x.infos for x in mps.tensors]
    vacuum = list(mps_infos[0][0].keys())[0]
    target = list(mps_infos[-1][-1].keys())[0]
    vacuum = SZ(vacuum.n, vacuum.twos, vacuum.pg)
    target = SZ(target.n, target.twos, target.pg)
    if basis is None:
        basis = [info[1] for info in mps_infos]
    else:
        basis = basis.copy()
    assert len(basis) == len(mps)
    for ib, b in enumerate(basis):
        p = StateInfo()
        p.allocate(len(b))
        for ix, (k, v) in enumerate(b.items()):
            p.quanta[ix] = SZ(k.n, k.twos, k.pg)
            p.n_states[ix] = v
        basis[ib] = p
        p.sort_states()
    minfo = MPSInfo(mps.n_sites, vacuum, target, VectorStateInfo(basis))
    minfo.left_dims[0] = StateInfo(vacuum)
    for i, info in enumerate(mps_infos):
        p = minfo.left_dims[i + 1]
        p.allocate(len(info[-1]))
        for ix, (k, v) in enumerate(info[-1].items()):
            p.quanta[ix] = SZ(k.n, k.twos, k.pg)
            p.n_states[ix] = v
        p.sort_states()
    minfo.right_dims[mps.n_sites] = StateInfo(vacuum)
    for i, info in enumerate(mps_infos):
        p = minfo.right_dims[i]
        p.allocate(len(info[0]))
        for ix, (k, v) in enumerate(info[0].items()):
            p.quanta[ix] = target - SZ(k.n, k.twos, k.pg)
            p.n_states[ix] = v
        p.sort_states()
    minfo.tag = tag
    minfo.save_mutable()
    minfo.save_data("%s/%s-mps_info.bin" % (save_dir, tag))
    tensors = [None] * len(mps)
    for i, b in enumerate(basis):
        tensors[i] = bx.SparseTensor()
        tensors[i].data = bx.VectorVectorPSSTensor([bx.VectorPSSTensor() for _ in range(b.n)])
        for block in mps[i].blocks:
            ql, qm, qr = [SZ(x.n, x.twos, x.pg) for x in block.q_labels]
            im = b.find_state(qm)
            assert im != -1
            tensors[i].data[im].append(((ql, qr), bf.Tensor(VectorInt(block.shape))))
            np.array(tensors[i].data[im][-1][1], copy=False)[:] = block
    umps = bx.UnfusedMPS()
    umps.info = minfo
    umps.n_sites = len(mps)
    umps.canonical_form = "L" * center + ("S" if center == len(mps) - 1 else "K") + \
        "R" * (len(mps) - center - 1)
    umps.center = center
    umps.dot = 2
    umps.tensors = bx.VectorSpTensor(tensors)
    umps = umps.finalize()
    if inited:
        release_memory()
        Global.frame = frame_back
    return umps





#### Auxiliary functions:

def occupations(mps, ncas, hamil):
    
    from pyblock3.symbolic.expr import OpElement, OpNames
    from pyblock3.algebra.symmetry import SZ
    
    occupations_up = []
    occupations_dn = []

    for j in range(ncas):
        dop = OpElement(OpNames.D, (j, 0), q_label=SZ(-1, -1, hamil.orb_sym[j]))
        dmpo = hamil.build_site_mpo(dop)
        cop = OpElement(OpNames.C, (j, 0), q_label=SZ(+1, +1, hamil.orb_sym[j]))
        cmpo = hamil.build_site_mpo(cop)

        mps_aux = dmpo @ mps

        if np.dot(mps_aux,mps_aux)<0.0001:
            occupations_up = np.append( occupations_up ,  0 )
        else:    
            occupations_up = np.append( occupations_up , np.dot(mps, cmpo @ mps_aux ) )


        dop = OpElement(OpNames.D, (j, 1), q_label=SZ(-1, +1, hamil.orb_sym[j]))
        dmpo = hamil.build_site_mpo(dop)
        cop = OpElement(OpNames.C, (j, 1), q_label=SZ(+1, -1, hamil.orb_sym[j]))
        cmpo = hamil.build_site_mpo(cop)

        mps_aux = dmpo @ mps

        if np.dot(mps_aux,mps_aux)<0.0001:
            occupations_dn = np.append( occupations_dn ,  0 )
        else:    
            occupations_dn = np.append( occupations_dn , np.dot(mps, cmpo @ mps_aux ) )
            
        
    return [occupations_up, occupations_dn] 


def occupations_bd1_to_bitstring(occupations):
    occupations_up = occupations[0]
    occupations_dn = occupations[1]
    
    bitstring_out = [False]*(2*len(occupations_up))
    
    for j in range( len(occupations_up) ):
        bitstring_out[2*j  ] = ( abs(occupations_up[j]-1) < 0.01 )
        bitstring_out[2*j+1] = ( abs(occupations_dn[j]-1) < 0.01 )
        
    return bitstring_out


def det_to_bit_string(det):
    bitstring_out = [False]*(2*len(det))
    
    for j in range(len(det)):
        if det[j] in [1,3]:
            bitstring_out[2*j] = True
        if det[j] in [2,3]:
            bitstring_out[2*j+1] = True
            
            
    return bitstring_out


def get_occ_from_det(det):
    occ_out = [0]*(len(det))
    
    for j in range(len(det)):
        if det[j] == 3:
            occ_out[j] = 2
        if det[j] in [1,2]:
            occ_out[j] = 1
            
    return occ_out


def single_move_difference(current_state_string , next_state_string):
    site_right = len(current_state_string)
    site_left = len(current_state_string)
    create_at_right = True
    
    
    not_found = True
    
    j_r=len(current_state_string)-1
    
    while not_found and j_r>=0:

        if current_state_string[j_r] != next_state_string[j_r]:
            not_found = False
            site_right = j_r
            
            create_at_right = next_state_string[j_r]         
    
        j_r = j_r-1
        
    not_found_left = True
    
    if not not_found:
        j_l = site_right-2
        
        while not_found_left:
            if current_state_string[j_l] != next_state_string[j_l]:
                if next_state_string[j_l] != create_at_right:
                    not_found_left = False
                    site_left = j_l
                    
                
            j_l = j_l-2

    return site_right, site_left, create_at_right



def mpo_properties_single_step(current_state_string , next_state_string):
    site_right, site_left, create_at_right = single_move_difference(current_state_string , next_state_string)
    
    antisymmetry_phase = (-1)**sum(current_state_string[site_left+1:site_right])
    
    updated_state = current_state_string
    
    updated_state[site_left] = not updated_state[site_left]
    updated_state[site_right] = not updated_state[site_right]

    return site_right, site_left, create_at_right, antisymmetry_phase, updated_state



def single_move(mps, site_right, site_left, create_at_right, antisymmetry_phase, hamil):
    
    from pyblock3.symbolic.expr import OpElement, OpNames
    from pyblock3.algebra.symmetry import SZ
    
    if create_at_right :
        j_c = site_right //2
        j_d = site_left  //2
    else:
        j_d = site_right //2
        j_c = site_left  //2
    
    s_ind = site_right % 2
    # s_ind=0 -> α and s_ind=1 -> β
    
    
    
    dop = OpElement(OpNames.D, (j_d, s_ind), q_label=SZ(-1, +(2*s_ind-1), hamil.orb_sym[j_d]))
    dmpo = hamil.build_site_mpo(dop)
    cop = OpElement(OpNames.C, (j_c, s_ind), q_label=SZ(+1, -(2*s_ind-1), hamil.orb_sym[j_c]))
    cmpo = hamil.build_site_mpo(cop)
    
    
    return antisymmetry_phase * cmpo @(dmpo @ mps) 