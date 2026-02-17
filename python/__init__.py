"""
SBD (Selected Basis Diagonalization) Python Package

This package provides Python bindings for the SBD library's Tensor Product Basis (TPB)
diagonalization functionality.

Main Functions:
    - tpb_diag: Perform TPB diagonalization with pre-loaded data structures
    - tpb_diag_from_files: Perform TPB diagonalization from files (convenience)
    - LoadFCIDump: Load FCIDUMP file
    - LoadAlphaDets: Load alpha determinants from file
    - makestring: Convert bitstring to string representation

Classes:
    - TPB_SBD: Configuration for TPB diagonalization
    - FCIDump: FCIDUMP data structure

Example:
    >>> from mpi4py import MPI
    >>> import sbd
    >>> 
    >>> comm = MPI.COMM_WORLD
    >>> config = sbd.TPB_SBD()
    >>> config.max_it = 100
    >>> config.eps = 1e-6
    >>> 
    >>> results = sbd.tpb_diag_from_files(
    ...     comm=comm,
    ...     sbd_data=config,
    ...     fcidumpfile="fcidump.txt",
    ...     adetfile="alphadets.txt"
    ... )
    >>> 
    >>> print(f"Energy: {results['energy']}")
"""

from ._core import (
    # Classes
    FCIDump,
    TPB_SBD,
    
    # Functions
    LoadFCIDump,
    LoadAlphaDets,
    makestring,
    tpb_diag,
    tpb_diag_from_files,
    
    # Version
    __version__
)

__all__ = [
    'FCIDump',
    'TPB_SBD',
    'LoadFCIDump',
    'LoadAlphaDets',
    'makestring',
    'tpb_diag',
    'tpb_diag_from_files',
    '__version__',
]

# Made with Bob
