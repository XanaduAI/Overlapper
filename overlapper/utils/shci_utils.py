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

import numpy as np
import os
from overlapper.utils.utils import get_mol_attrs

def get_dets_coeffs_output(output_file, state=0):
    '''
    Get CI coeff of SHCI from output file and parse them.
    '''    
    coeffs = []
    dets = []

    with open(output_file) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            search = line.strip()
            while search[:5] == 'State' and search[-1:] == str(state):
                line = fp.readline()
                cnt += 1
                try:
                    num = int(line.strip()[0])
                    data = line.strip().split('  ')
                    coeffs.append(float(data[3]))
                    data_dets = ''
                    for i in np.arange(4, len(data)):
                        data_dets += data[i]
                    dets.append((data_dets).split())
                except:
                    search = line.strip()
            line = fp.readline()
            cnt += 1
    
    return dets, coeffs

def convert_bin_ab(list_string):
    '''
    Change of notation for Slater determinants
    '''
    bin_a = ''
    bin_b = ''
    for el in list_string:

        if el == '2':
            bin_a += '1'
            bin_b += '1'
        elif el == 'a':
            bin_a += '1'
            bin_b += '0'
        elif el == 'b':
            bin_a += '0'
            bin_b += '1'           
        elif el == '0':
            bin_a += '0'
            bin_b += '0'
            
    return bin_a[::-1], bin_b[::-1]


def getinitialStateSHCI(mf, nelecas):
    
    '''Get initial state for SHCI - specially for spin sectior different from zero'''
    
    norb, nelec_a, nelec_b = get_mol_attrs(mf.mol)
    n_total = nelec_a + nelec_b
    n_frozen_s = int((n_total-np.sum(np.array(nelecas)))/2)
    nelec_a -= n_frozen_s
    nelec_b -= n_frozen_s

    initialState = []
    for i in range(int(nelec_a)):
        initialState.append((2 * i))
    for i in range(int(nelec_b)):
        initialState.append((2 * i + 1))
        
    return np.array([initialState])

def cleanup_dice_files(filedir="."):
    all_local_files = os.listdir(filedir)
    for datafile in all_local_files:
        if ".bkp" in datafile or "RDM" in datafile or "FCIDUMP" in datafile or "input.dat" in datafile or "shci.e" in datafile:
            os.remove(datafile)
    print(f"Dice files cleaned up!")