"""
SBD solver wrapper compatible with qiskit-addon-sqd interface.

This module provides functions that wrap the SBD (Selected Basis Diagonalization)
library to be compatible with the qiskit-addon-sqd diagonalize_fermionic_hamiltonian
interface, similar to how qiskit-addon-dice-solver works.
"""

from __future__ import annotations

import tempfile
import shutil
from pathlib import Path
from collections.abc import Sequence
from typing import Callable

import numpy as np
from mpi4py import MPI
from pyscf import tools

try:
    from qiskit_addon_sqd.fermion import SCIResult, SCIState
except ImportError:
    raise ImportError(
        "qiskit-addon-sqd is required. Install it with: pip install qiskit-addon-sqd"
    )

# Import SBD Python bindings
try:
    import sbd
except ImportError:
    raise ImportError(
        "SBD Python bindings not found. Please install SBD with Python support."
    )


def solve_sci(
    ci_strings: tuple[np.ndarray, np.ndarray],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    spin_sq: float | None = None,
    mpi_comm: MPI.Comm | None = None,
    sbd_config: dict | None = None,
    temp_dir: str | Path | None = None,
    clean_temp_dir: bool = True,
) -> SCIResult:
    """
    Diagonalize Hamiltonian in subspace defined by CI strings using SBD.

    Args:
        ci_strings: Pair (strings_a, strings_b) of arrays of spin-alpha CI
            strings and spin-beta CI strings whose Cartesian product give the basis of
            the subspace in which to perform a diagonalization.
        one_body_tensor: The one-body tensor of the Hamiltonian.
        two_body_tensor: The two-body tensor of the Hamiltonian.
        norb: The number of spatial orbitals.
        nelec: The numbers of alpha and beta electrons.
        spin_sq: Target value for the total spin squared (currently unused by SBD).
        mpi_comm: MPI communicator. If None, uses MPI.COMM_WORLD.
        sbd_config: Dictionary of SBD configuration parameters. If None, uses defaults.
        temp_dir: An absolute path to a directory for storing temporary files.
        clean_temp_dir: Whether to delete intermediate files.

    Returns:
        The diagonalization result as SCIResult.
    """
    n_alpha, n_beta = nelec
    
    # Set up MPI communicator
    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD
    
    mpi_rank = mpi_comm.Get_rank()
    
    # Set up temp directory
    temp_dir = temp_dir or tempfile.gettempdir()
    sbd_dir = Path(tempfile.mkdtemp(prefix="sbd_files_", dir=temp_dir))
    
    try:
        # Write FCIDUMP file
        fcidump_path = sbd_dir / "fcidump.txt"
        if mpi_rank == 0:
            tools.fcidump.from_integrals(
                str(fcidump_path),
                one_body_tensor,
                two_body_tensor,
                norb,
                nelec,
            )
        mpi_comm.Barrier()
        
        # Convert CI strings to SBD determinant format
        strings_a, strings_b = ci_strings
        adet = _ci_strings_to_sbd_dets(strings_a, norb)
        bdet = _ci_strings_to_sbd_dets(strings_b, norb)
        
        # Set up SBD configuration
        sbd_data = _create_sbd_config(sbd_config)
        
        # Load FCIDUMP
        fcidump = sbd.LoadFCIDump(str(fcidump_path))
        
        # Run SBD diagonalization
        results = sbd.tpb_diag(
            mpi_comm,
            sbd_data,
            fcidump,
            adet,
            bdet,
            loadname="",
            savename=""
        )
        
        # Extract results
        energy = results["energy"]
        density = np.array(results["density"])
        
        # Convert density to orbital occupancies
        # SBD returns density as [alpha_0, beta_0, alpha_1, beta_1, ...]
        # We need to separate into (alpha_occs, beta_occs)
        occupancies_a = density[::2]  # Even indices
        occupancies_b = density[1::2]  # Odd indices
        occupancies = (occupancies_a, occupancies_b)
        
        # Get carryover determinants for wavefunction reconstruction
        co_adet = results["carryover_adet"]
        co_bdet = results["carryover_bdet"]
        
        # Convert carryover determinants back to CI strings
        co_strings_a = _sbd_dets_to_ci_strings(co_adet, norb)
        co_strings_b = _sbd_dets_to_ci_strings(co_bdet, norb)
        
        # Create a simple wavefunction representation
        # Note: SBD doesn't return full CI coefficients, so we approximate
        # with uniform weights for carryover determinants
        n_a = len(co_strings_a)
        n_b = len(co_strings_b)
        
        if n_a > 0 and n_b > 0:
            # Create approximate amplitudes (uniform distribution)
            amplitudes = np.ones((n_a, n_b)) / np.sqrt(n_a * n_b)
            sci_state = SCIState(
                amplitudes=amplitudes,
                ci_strs_a=co_strings_a,
                ci_strs_b=co_strings_b
            )
        else:
            # Fallback: use input strings with uniform amplitudes
            n_a = len(strings_a)
            n_b = len(strings_b)
            amplitudes = np.ones((n_a, n_b)) / np.sqrt(n_a * n_b)
            sci_state = SCIState(
                amplitudes=amplitudes,
                ci_strs_a=strings_a,
                ci_strs_b=strings_b
            )
        
        return SCIResult(energy, sci_state, orbital_occupancies=occupancies)
    
    finally:
        # Clean up temp directory
        if clean_temp_dir and mpi_rank == 0:
            shutil.rmtree(sbd_dir, ignore_errors=True)


def solve_sci_batch(
    ci_strings: list[tuple[np.ndarray, np.ndarray]],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    spin_sq: float | None = None,
    mpi_comm: MPI.Comm | None = None,
    sbd_config: dict | None = None,
    temp_dir: str | Path | None = None,
    clean_temp_dir: bool = True,
) -> list[SCIResult]:
    """
    Diagonalize Hamiltonian in multiple subspaces using SBD.

    Args:
        ci_strings: List of pairs (strings_a, strings_b) of arrays of spin-alpha CI
            strings and spin-beta CI strings whose Cartesian product give the basis of
            the subspace in which to perform a diagonalization.
        one_body_tensor: The one-body tensor of the Hamiltonian.
        two_body_tensor: The two-body tensor of the Hamiltonian.
        norb: The number of spatial orbitals.
        nelec: The numbers of alpha and beta electrons.
        spin_sq: Target value for the total spin squared (currently unused by SBD).
        mpi_comm: MPI communicator. If None, uses MPI.COMM_WORLD.
        sbd_config: Dictionary of SBD configuration parameters.
        temp_dir: An absolute path to a directory for storing temporary files.
        clean_temp_dir: Whether to delete intermediate files.

    Returns:
        The results of the diagonalizations in the subspaces given by ci_strings.
    """
    return [
        solve_sci(
            ci_strs,
            one_body_tensor,
            two_body_tensor,
            norb=norb,
            nelec=nelec,
            spin_sq=spin_sq,
            mpi_comm=mpi_comm,
            sbd_config=sbd_config,
            temp_dir=temp_dir,
            clean_temp_dir=clean_temp_dir,
        )
        for ci_strs in ci_strings
    ]


def _ci_strings_to_sbd_dets(
    ci_strings: np.ndarray, norb: int
) -> list[list[int]]:
    """
    Convert CI strings (integers) to SBD determinant format (list of size_t words).
    
    Args:
        ci_strings: Array of CI strings as integers
        norb: Number of orbitals
        
    Returns:
        List of determinants in SBD format
    """
    bit_length = 64  # Standard word size
    dets = []
    
    for ci_str in ci_strings:
        # Convert integer to binary string
        binary_str = format(int(ci_str), f'0{norb}b')
        # Convert to SBD determinant format using from_string
        det = sbd.from_string(binary_str, bit_length, norb)
        dets.append(det)
    
    return dets


def _sbd_dets_to_ci_strings(
    dets: list[list[int]], norb: int
) -> np.ndarray:
    """
    Convert SBD determinants to CI strings (integers).
    
    Args:
        dets: List of determinants in SBD format
        norb: Number of orbitals
        
    Returns:
        Array of CI strings as integers
    """
    bit_length = 64
    ci_strings = []
    
    for det in dets:
        # Convert SBD determinant to binary string
        binary_str = sbd.makestring(det, bit_length, norb)
        # Convert binary string to integer
        ci_str = int(binary_str, 2)
        ci_strings.append(ci_str)
    
    return np.array(ci_strings, dtype=np.int64)


def _create_sbd_config(config_dict: dict | None = None) -> sbd.TPB_SBD:
    """
    Create SBD configuration object from dictionary.
    
    Args:
        config_dict: Dictionary of configuration parameters
        
    Returns:
        SBD configuration object
    """
    sbd_data = sbd.TPB_SBD()
    
    # Set defaults
    sbd_data.method = 0  # Davidson
    sbd_data.max_it = 100
    sbd_data.max_nb = 50
    sbd_data.eps = 1e-8
    sbd_data.max_time = 3600
    sbd_data.init = 0
    sbd_data.do_shuffle = 0
    sbd_data.do_rdm = 0  # Only density
    sbd_data.carryover_type = 1
    sbd_data.ratio = 0.1
    sbd_data.threshold = 1e-4
    sbd_data.bit_length = 64
    
    # Override with user config
    if config_dict:
        for key, value in config_dict.items():
            if hasattr(sbd_data, key):
                setattr(sbd_data, key, value)
    
    return sbd_data


# Convenience function for use with functools.partial
def create_sbd_solver(
    mpi_comm: MPI.Comm | None = None,
    sbd_config: dict | None = None,
    temp_dir: str | Path | None = None,
    clean_temp_dir: bool = True,
) -> Callable:
    """
    Create a configured SBD solver function for use with diagonalize_fermionic_hamiltonian.
    
    Example:
        >>> from functools import partial
        >>> sbd_solver = create_sbd_solver(sbd_config={"method": 0, "eps": 1e-10})
        >>> result = diagonalize_fermionic_hamiltonian(
        ...     hcore, eri, bit_array,
        ...     sci_solver=sbd_solver,
        ...     ...
        ... )
    
    Args:
        mpi_comm: MPI communicator
        sbd_config: SBD configuration dictionary
        temp_dir: Temporary directory path
        clean_temp_dir: Whether to clean up temp files
        
    Returns:
        Configured solve_sci_batch function
    """
    from functools import partial
    
    return partial(
        solve_sci_batch,
        mpi_comm=mpi_comm,
        sbd_config=sbd_config,
        temp_dir=temp_dir,
        clean_temp_dir=clean_temp_dir,
    )

# Made with Bob
