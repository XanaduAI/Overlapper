# Overlapper: Prepare and evaluate wavefunctions for quantum
# algorithms using computational chemistry techniques
# Copyright 2024 Xanadu Quantum Technologies Inc.
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

from setuptools import setup, find_packages

# Parse requirements from file
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

short_description = \
    """
    A library for pre-computing initial states for quantum algorithms for 
    quantum chemistry using post-Hartree-Fock wavefunction-based methods.
    """

setup(
    name='overlapper',
    version='0.1.0',
    author='Xanadu Quantum Technologies',
    author_email='stepan.fomichev@xanadu.ai',
    description=short_description,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/XanaduAI/initial_state',
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "examples": [
            "matplotlib>=3.7.2", "jupyter", "block2>=0.5.2r10", 
            "pyblock3", "shciscf", "mpi4py"
        ],
    },  
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research"
        "License :: OSI Approved ::  Apache 2.0",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)