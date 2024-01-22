<div align="center">

# Overlapper: preparing and assessing initial states for quantum algorithms on molecules

<img src="/media/readme/pipeline.png" width="85%" height="85%"/>

[![arXiv](http://img.shields.io/badge/application-2310.18410-B31B1B.svg "Example application of Overlapper")](https://arxiv.org/abs/2310.18410)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/XanaduAI/initial_state/blob/main/LICENSE)
</div>

Many quantum algorithms for ground-state energy estimation of molecules require a high-quality initial state as a starting point. One way of preparing these states is by pre-computing a wavefunction using a traditional quantum chemistry wavefunction-based method. 

The purpose of the Overlapper software library is to simplify the preparation of such initial states for molecules. By interfacing with packages implementing state-of-the-art quantum chemistry methods, most prominently coupled cluster with singles and doubles (CCSD), density-matrix renormalization group (DMRG) and semistochastic heat-bath configuration integration (SHCI), Overlapper provides easy access to initial states from these methods. 

Beyond that, it transforms them into one of two unified formats, the sum of Slater determinants (SOS) or matrix-product state (MPS), and enables, for the first time, their straightforward comparison to each other -- using either wavefunction overlap or the newly developed energy distribution picture.

## Functionality

Overlapper is at present capable of the following

1.  Execute traditional wavefunction-based methods to pre-compute an initial state, including
    *  Hartree-Fock (HF)
    *  Configuration interaction with singles and doubles (CISD)
    *  Coupled cluster with singles and doubles (CCSD)
    *  Complete active space configuration interaction (CASCI)
    *  Multireference perturbation theory (MRPT)
    *  Density-matrix renormalization group (DMRG)
    *  Semistochastic heat-bath configuration interaction (SHCI)
2.  Convert the incredibly diverse outputs of all wavefunction-based methods into either the sum of Slater determinants (SOS) unified format or the matrix-product state (MPS) format.
3.  Compare and evaluate these initial states using either wavefunction overlap or the energy distribution method.

In the future, we intend Overlapper to continue incorporating support for the emerging state-of-the-art quantum chemistry techniques for initial state preparation for quantum algorithms. Let us know if you think you know of a method that is not currently supported that should be!

## Install

The software package [PySCF](https://pyscf.org) is a fundamental dependency of our library. We rely on PySCF for the fundamental object `Molecule` and the implementation of several post-Hartree-Fock methods (CISD/CCSD/CASCI/MRPT) to generate candidate initial states. Beyond that, to generate initial states with DMRG the library [Block2](https://block2.readthedocs.io/en/latest/) is required; and to get an initial state with the SHCI method, the library [Dice](https://sanshar.github.io/Dice/index.html) is needed.

To install the base version of our package with only PySCF support, create a fresh conda environment [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment), clone the current repository, and execute the following command from its head-level directory

```bash
pip install -e .
```

This will install the base package `overlapper` with support only for the PySCF-based quantum chemistry methods (the first five methods above). If you want support for the state-of-the-art techniques of DMRG and SHCI, read on.

### Install Block2 for DMRG support in Overlapper

Overlapper interfaces with the well-established [Block2](https://block2.readthedocs.io/en/latest/) package to pre-compute initial states for molecules using the highly successful combination of the MPS ansatz and DMRG algorithm. To make sure your install of Block2 includes all the features we rely on, install the latest version of Block2 by executing the following command

```bash
pip install block2
```

### Install Block3 for SOS to MPS conversion

To enable the conversion of SOS wavefunctions into MPS wavefunctions -- thus allowing to use DMRG for energy distribution computations -- Overlapper additionally relies on `pyblock3`. It can be installed in the following way

```bash
pip install pyblock3
```

### Install Dice for SHCI support in Overlapper

To provide support for a representative member of the selective configuration interaction family of methods, Overlapper interfaces with the [Dice](https://sanshar.github.io/Dice/index.html) library's implementation of the SHCI method. As this library is not available on PyPI, we advise that the installation process could be more involved. Below is an example installation guide that ought to work in many standard Linux environments:

1. Clone the latest version of the master branch of the [Dice repository](https://github.com/sanshar/Dice).
2. Follow the instructions in the Dice README to install the dependencies (Boost and Eigen are sufficient). Note that Eigen is now shipped with Dice, whereas the installation of Boost will require compilation of the `mpi`-specific libraries. Make sure to run `./b2 install` after executing the template instructions in the Dice README file to place the libraries in a place discoverable by Dice. 
3. Set the paths to the dependencies at the top of the Makefile in the root directory of the cloned Dice repo, and set other key environment variables such as MPI-support, AVX2 system and so on.
4. Compile only the SHCI portion of the library by executing `make Dice -jX` where `X` is your desired number of processes for the compilation. 
5. Check your installation by running tests under `tests/SHCI/runSmallTests.sh`
6. Install the SHCI-to-PySCF interface [shciscf](https://github.com/pyscf/shciscf/tree/master) by executing `pip install git+https://github.com/pyscf/shciscf`. 
7. Follow the instructions in the interface's README to create a `settings.py` file and specify the location of the compiled Dice executable. Notice that both `SHCISCRATCHDIR` and `SHCIRUNTIMEDIR` must be set to local directory `"."`, otherwise segmentation faults might occur.
8. Test the installation of the SHCI-to-PySCF interface by running Python examples from the [shciscf](https://github.com/pyscf/shciscf/tree/master/examples) repository. 
9. In order to run SHCI calculations for non-zero spin, modify the `shci.py` file in the same directory as the `settings.py` file above by including the following at line 1091
```python
f.write("spin %r\n" %SHCI.spin)
```

10. In order to obtain more than 6 Slater determinants in the representation of SHCI wavefunctions, modify the `shci.py` file by including the following at line 1092
```python
for elem in SHCI.shci_extra_keyword:
    f.write(elem + "\n")
```
This writes anything passed as `shci_extra_keyword` to the SHCI Solver to the input file: support for the option for additional determinants `printBestDeterminants X` is already incorporated into Overlapper.

## Typical workflow

Here is an example of what Overlapper can enable you to do:

1. Easily execute methods as diverse as CCSD, DMRG and SHCI with only one to three lines of code.
2. Convert the outputs of these diverse methods into a unified wavefunction format: sum of Slater determinants, or matrix-product state.
3. Compute wavefunction overlaps across all these candidate initial states to evaluate them.
4. Generate energy distributions of initial states, and use them to compare and select the highest-quality initial state.

In code, this workflow would look like this 

### Create a molecule

The first step is to create a `Molecule` object using PySCF routines.

```python
from pyscf import gto
from overlapper.state import do_hf

# Define a PySCF mol object for the N2 molecule with the stretched bonds
mol = gto.M(atom = [['N', (0, 0, 0)], ['N', (1.11 * 2.25, 0, 0)]], basis = 'sto-6g')
# Use Overlapper to execute PySCF's Hartree-Fock solver
hf, hf_e, hf_ss, hf_sz = do_hf(mol, hftype="rhf")
```
See at a glance the energy, as well as $\langle S^2 \rangle$ and $\langle S_z \rangle$ of your state -- the same uniform return types for all methods in Overlapper.

### Pre-compute a few initial states in the unified SOS format

Choose the computational chemistry methods you are interested in and execute them in a few lines

```python
from overlapper.state import do_cisd, do_dmrg, do_shci

mycisd, mycisd_e, mycisd_ss, mycisd_sz = do_cisd(hf)

schedule = [ [100,200], [20,20], [1e-3,1e-4], [1e-5,1e-6] ]
ncas, nelecas = 10, (4,4)
mydmrg, mydmrg_e, mydmrg_ss, mydmrg_sz = do_dmrg(hf, ncas, nelecas, sch)

outfile = "SHCI_output.dat"
eps1_arr = [1e-2,1e-3,1e-4]
n_sweeps = [20,20,20]
myshci, myshci_e, myshci_ss, myshci_sz = \
    do_shci(hf, ncas, nelecas, eps1_arr, n_sweeps, outfile)
```
We have tuned default values for many of the optional arguments to improve ease-of-use: however we still support customization of calculations with extensive keyword arguments for the advanced user. See the docstrings for the `do_xxx` functions for details.

### Comparing states through overlap

Easily convert all outputs to the same wavefunction format -- and compute overlaps

```python
from overlapper.state import cisd_state, dmrg_state, shci_state
from overlapper.utils.wf_utils import wf_overlap

wf_cisd = cisd_state(mycisd)
wf_dmrg = dmrg_state(mydmrg)
wf_shci = shci_state(myshci)

ovlp = wf_overlap(wf_cisd, wf_dmrg)
```
Computing a many-body wavefunction overlap takes one function call.

### Evaluating states using the energy distribution

Compute the energy distribution -- either through series expansion or the resolvent method (MPI-parallelized examples available)

```python
mpo, mps, driver = do_dmrg(hf, ncas, nelecas, sch, return_objects=True)
freqs = np.linspace(-5,0, 100)
eta = 0.1
results = [resolvent_calc(driver, mps, mpo, freq, eta) for freq in freqs]
```

For the energy distribution of a non-MPS state, first convert it to MPS form using Overlapper's converter in `sos_to_mps.py`.

For more detailed examples, check out the `~/examples` folder.

<p align="center">

<img src="/media/readme/edists.png" width="52%" height="65%"/>
<img src="/media/readme/ovlps.png" width="38%" height="45%"/>
</p>

## License

Overlapper is free and open source, licensed under GNU General Public License version 3.0. It has a core dependency on PySCF (Apache 2.0), as well as optional dependencies on Dice (GNU General Public License version 3.0), and Block2/pyblock3 (both GNU General Public License version 3.0). See the LICENSE and NOTICE files and the source code for further details. 

## Acknowledgements

We gratefully acknowledge fruitful discussions with Huanchen Zhai, Soran Jahangiri, Alain Delgado, Modjtaba Shokrian Zini, Pablo Antonio Moreno Casares, Joonsuk Huh, Arne-Christian Voigt, and Jonathan E. Mueller. S.F. acknowledges support by Mitacs through the Mitacs Accelerate Program. 

## Bibtex

If you use Overlapper for your work, please cite it as a software package
```
@misc{Overlapper,
  author = {Stepan Fomichev, Kasra Hejazi, Joana Fraxanet, Juan Miguel Arrazola},
  title = {Overlapper},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/XanaduAI/Overlapper/}},
}
```

For an example application of Overlapper, see the [following manuscript](https://arxiv.org/abs/2310.18410):
```
@article{overlapper,
  title={Initial state preparation for quantum chemistry on quantum computers},
  author={Stepan Fomichev, Kasra Hejazi, Modjtaba Shokrian Zini, Matthew Kiser, Joana Fraxanet Morales, Pablo Antonio Moreno Casares, Alain Delgado, Joonsuk Huh, Arne-Christian Voigt, Jonathan E. Mueller, and Juan Miguel Arrazola},
  journal={arXiv preprint arxiv:2310.18410},
  year={2023}
}
```
