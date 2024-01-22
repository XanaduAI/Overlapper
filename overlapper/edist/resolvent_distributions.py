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


import time

def resolvent_calc(driver, mps, mpo, om, eta, calcbd = 100, \
                                gf_tol=1e-3, thrds=1e-4, rank=0):
    bra = driver.get_random_mps(tag='bra', bond_dim=calcbd, dot=2)
    impo = driver.get_identity_mpo()

    start = time.time()
    gfval = driver.greens_function(bra, mpo, impo, mps, -om, \
                                eta, tol = gf_tol, thrds = [thrds])
    dt = time.time() - start 
    print(f"rank = {rank} doing omega = {om:.3f} result = {gfval:.3f} took {dt:.2f} seconds")
    return gfval
