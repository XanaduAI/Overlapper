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
import matplotlib.pyplot as plt

def wf_overlap(wf_dict1, wf_dict2):
    '''
    Computes overlap between two wavefunctions represented as dicts.
    Reasonably fast and benchmarked against the sparse matrix approach
    where it was possible.
    '''
    common_keys = list( set(wf_dict1.keys()).intersection( set(wf_dict2.keys()) ) )
    overlap = 0
    for key in common_keys:
        overlap += wf_dict1[key] * wf_dict2[key]
    return abs(overlap)

def wf_norm(wf_dict):
    return wf_overlap(wf_dict, wf_dict)

def compare_subspaces(wf_dict1, wf_dict2, tol):
    '''
    Given two wavefunctions in the fcidict representation,
    counts 
    a) the percent total of Slaters from the first one that 
    do not appear in the second one,
    b) the opposite,
    '''

    # start by filtering based on tolerance cutoff 
    # make sure they are both the same 
    dict1 = {key: value for key, value in \
                    wf_dict1.items() if abs(value) > tol}
    dict2 = {key: value for key, value in \
                    wf_dict2.items() if abs(value) > tol}

    wf1_keys = set(dict1.keys())
    wf2_keys = set(dict2.keys())

    num_S_not_in_1 = len(list(wf1_keys - wf2_keys)) / len(list(wf1_keys))
    num_S_not_in_2 = len(list(wf2_keys - wf1_keys)) / len(list(wf2_keys))

    return num_S_not_in_1, num_S_not_in_2

def wf_normalize(wf_dict):
    '''
    Normalizes a given input wavefunction in FCIDICT format.
    '''
    norm_squared = wf_norm(wf_dict)
    wf_dict_normed = {key: value / np.sqrt(norm_squared) for key, value in wf_dict.items()}
    return wf_dict_normed

def wf_budget(wf_dict, N):
    '''
    Given an fcidict, this function selects N Slater determinants
    starting with those with the highest coefficients, where N 
    is the budget of Slaters we can afford to prepare on a quantum 
    computer.
    '''
    if len(wf_dict.items()) < N:
        print(f"Not enough Slater determinants to budget! Returning the whole thing.")
        return wf_dict
    else:
        wf_onabudget = dict(sorted(wf_dict.items(), \
                    key = lambda x: abs(x[1]), reverse = True)[:N])
        return wf_onabudget 

def print_topdets(wf, D, ncas):
    '''
    Pretty-prints the key and coefficient of the top D 
    largest coefficient Slater determinants.
    '''
    wf = wf_budget(wf, D)
    slaters = list(wf.keys())
    coeffs = np.array(list(wf.values()))
    
    idx = np.argsort(np.abs(coeffs))
    slaters = [slaters[ii] for ii in idx]
    coeffs = coeffs[idx]

    np.set_printoptions(linewidth=300)

    headstr = "".join([str(ii % 10) for ii in range(1, ncas+1)])
    print(f"\n{headstr}   {headstr}    coeff")

    for ii in range(len(slaters))[::-1]:

        stra = bin(slaters[ii][0])[2:][::-1]
        if len(stra) < ncas:
            stra = stra + "0" * int(ncas - len(stra))

        strb = bin(slaters[ii][1])[2:][::-1]
        if len(strb) < ncas:
            strb = strb + "0" * int(ncas - len(strb))

        print(f"\n{stra}   {strb}    {coeffs[ii]:.5f}")