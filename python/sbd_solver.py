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

# Import SBD Python bindings using the init() approach from __init__.py
# This avoids the pybind11 "already registered" conflict
import os

# Use the sbd.init() API which properly handles backend selection
from . import init as sbd_init, finalize as sbd_finalize
from . import _device_module, _initialized

# Backend will be set by init() call
sbd = None
_selected_backend = None
_backend_info = {
    'selected': None,
    'cpu_available': False,
    'gpu_available': False,
}


def _ensure_sbd_initialized():
    """Ensure SBD is initialized, initialize with auto-detect if not."""
    global sbd, _selected_backend, _backend_info
    
    if not _initialized:
        # Auto-initialize with device from environment or auto-detect
        device = os.environ.get('SBD_BACKEND', 'auto').lower()
        if device not in ['cpu', 'gpu', 'auto']:
            device = 'auto'
        
        try:
            sbd_init(device=device, comm_backend='mpi')
        except RuntimeError as e:
            # Already initialized in another way, that's ok
            pass
    
    # Get the device module that was initialized
    from . import _device_module as dm, _initialized as init_flag
    if init_flag and dm is not None:
        sbd = dm
        # Determine which backend was loaded
        if 'gpu' in str(type(dm)):
            _selected_backend = 'gpu'
        else:
            _selected_backend = 'cpu'
        
        _backend_info['selected'] = _selected_backend
        _backend_info['cpu_available'] = True  # Assume available if we got here
        _backend_info['gpu_available'] = _selected_backend == 'gpu'
    
    return sbd


def _get_backend_module(use_gpu: bool):
    """
    Get the appropriate SBD backend module based on device configuration.
    
    Args:
        use_gpu: Whether to use GPU backend
        
    Returns:
        The backend module
        
    Raises:
        ImportError: If requested backend doesn't match initialized backend
    """
    global sbd, _selected_backend
    
    # Ensure SBD is initialized
    if sbd is None:
        sbd = _ensure_sbd_initialized()
    
    # Validate the request matches what was initialized
    if use_gpu and _selected_backend != 'gpu':
        raise ImportError(
            f"GPU backend requested but {_selected_backend.upper() if _selected_backend else 'UNKNOWN'} backend was initialized. "
            f"Call sbd.init(device='gpu') before using sbd_solver, or set SBD_BACKEND=gpu environment variable."
        )
    elif not use_gpu and _selected_backend == 'gpu':
        raise ImportError(
            f"CPU backend requested but GPU backend was initialized. "
            f"Call sbd.init(device='cpu') before using sbd_solver, or set SBD_BACKEND=cpu environment variable."
        )
    
    return sbd


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
    device_config = None,  # DeviceConfig object
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
        device_config: DeviceConfig object to select CPU/GPU backend. If None, uses default.

    Returns:
        The diagonalization result as SCIResult.
    """
    n_alpha, n_beta = nelec
    
    # Select backend based on device configuration
    if device_config is not None:
        backend = _get_backend_module(device_config.use_gpu)
    else:
        backend = sbd  # Use default backend
    
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
        adet = _ci_strings_to_sbd_dets(strings_a, norb, backend)
        bdet = _ci_strings_to_sbd_dets(strings_b, norb, backend)
        
        # Set up SBD configuration
        sbd_data = _create_sbd_config(sbd_config, backend, device_config)
        
        # Set up file to dump wavefunction in matrix form
        wf_dump_file = sbd_dir / "wavefunction.txt"
        sbd_data.dump_matrix_form_wf = str(wf_dump_file)
        
        # Load FCIDUMP
        fcidump = backend.LoadFCIDump(str(fcidump_path))
        
        # Run SBD diagonalization
        results = backend.tpb_diag(
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
        
        # Debug: Print what we got from SBD
        if mpi_rank == 0:
            print(f"\n[DEBUG] SBD Results:")
            print(f"  Energy type: {type(energy)}, value: {energy}")
            print(f"  Density shape: {density.shape}, dtype: {density.dtype}")
            print(f"  Density first 10 elements: {density[:10] if len(density) >= 10 else density}")
            print(f"  Energy is NaN: {np.isnan(energy)}")
            print(f"  Energy is finite: {np.isfinite(energy)}")
        
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
        co_strings_a = _sbd_dets_to_ci_strings(co_adet, norb, backend)
        co_strings_b = _sbd_dets_to_ci_strings(co_bdet, norb, backend)
        
        # Read wavefunction coefficients from dumped file
        mpi_comm.Barrier()  # Ensure file is written
        amplitudes = None
        if mpi_rank == 0 and wf_dump_file.exists():
            amplitudes = _read_wavefunction_matrix(wf_dump_file)
        
        # Broadcast amplitudes to all ranks
        amplitudes = mpi_comm.bcast(amplitudes, root=0)
        
        # Create SCIState with actual wavefunction
        if amplitudes is not None and amplitudes.shape == (len(co_strings_a), len(co_strings_b)):
            sci_state = SCIState(
                amplitudes=amplitudes,
                ci_strs_a=co_strings_a,
                ci_strs_b=co_strings_b,
                norb=norb,
                nelec=nelec
            )
        elif len(co_strings_a) > 0 and len(co_strings_b) > 0:
            # Fallback: use carryover dets with uniform amplitudes
            n_a = len(co_strings_a)
            n_b = len(co_strings_b)
            amplitudes = np.ones((n_a, n_b)) / np.sqrt(n_a * n_b)
            sci_state = SCIState(
                amplitudes=amplitudes,
                ci_strs_a=co_strings_a,
                ci_strs_b=co_strings_b,
                norb=norb,
                nelec=nelec
            )
        else:
            # Last resort: use input strings with uniform amplitudes
            n_a = len(strings_a)
            n_b = len(strings_b)
            amplitudes = np.ones((n_a, n_b)) / np.sqrt(n_a * n_b)
            sci_state = SCIState(
                amplitudes=amplitudes,
                ci_strs_a=strings_a,
                ci_strs_b=strings_b,
                norb=norb,
                nelec=nelec
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
    device_config = None,  # DeviceConfig object
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
            device_config=device_config,
        )
        for ci_strs in ci_strings
    ]


def _ci_strings_to_sbd_dets(
    ci_strings: np.ndarray, norb: int, backend=None
) -> list[list[int]]:
    """
    Convert CI strings (integers) to SBD determinant format (list of size_t words).
    
    Args:
        ci_strings: Array of CI strings as integers
        norb: Number of orbitals
        backend: SBD backend module to use (if None, uses default)
        
    Returns:
        List of determinants in SBD format
    """
    if backend is None:
        backend = sbd
    
    bit_length = 64  # Standard word size
    dets = []
    
    for ci_str in ci_strings:
        # Convert integer to binary string
        binary_str = format(int(ci_str), f'0{norb}b')
        # Convert to SBD determinant format using from_string
        det = backend.from_string(binary_str, bit_length, norb)
        dets.append(det)
    
    return dets


def _sbd_dets_to_ci_strings(
    dets: list[list[int]], norb: int, backend=None
) -> np.ndarray:
    """
    Convert SBD determinants to CI strings (integers).
    
    Args:
        dets: List of determinants in SBD format
        norb: Number of orbitals
        backend: SBD backend module to use (if None, uses default)
        
    Returns:
        Array of CI strings as integers
    """
    if backend is None:
        backend = sbd
    
    bit_length = 64
    ci_strings = []
    
    for det in dets:
        # Convert SBD determinant to binary string
        binary_str = backend.makestring(det, bit_length, norb)
        # Convert binary string to integer
        ci_str = int(binary_str, 2)
        ci_strings.append(ci_str)
    
    return np.array(ci_strings, dtype=np.int64)


def _read_wavefunction_matrix(filepath: Path) -> np.ndarray | None:
    """
    Read wavefunction coefficients from SBD matrix form dump file.
    
    The actual file format from SBD is:
    coefficient # ia: alpha_bitstring ib: beta_bitstring
    coefficient # ia: alpha_bitstring ib: beta_bitstring
    ...
    
    We need to parse this and reconstruct the matrix.
    
    Args:
        filepath: Path to wavefunction dump file
        
    Returns:
        2D array of wavefunction coefficients (n_alpha x n_beta), or None if file doesn't exist
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return None
        
        # Parse all coefficients and their indices
        coeffs = []
        for line in lines:
            parts = line.strip().split('#')
            if len(parts) < 2:
                continue
            
            # Extract coefficient
            coeff = float(parts[0].strip())
            
            # Extract indices: "ia: bitstring ib: bitstring"
            indices_part = parts[1].strip()
            ia_part, ib_part = indices_part.split('ib:')
            
            # Extract ia index
            ia = int(ia_part.split(':')[0].strip())
            
            # Extract ib index
            ib = int(ib_part.split(':')[0].strip())
            
            coeffs.append((ia, ib, coeff))
        
        if not coeffs:
            return None
        
        # Determine matrix dimensions
        max_ia = max(c[0] for c in coeffs)
        max_ib = max(c[1] for c in coeffs)
        n_alpha = max_ia + 1
        n_beta = max_ib + 1
        
        # Build matrix
        amplitudes = np.zeros((n_alpha, n_beta))
        for ia, ib, coeff in coeffs:
            amplitudes[ia, ib] = coeff
        
        return amplitudes
        
    except (FileNotFoundError, IOError, ValueError, IndexError) as e:
        return None


def _create_sbd_config(config_dict: dict | None = None, backend=None, device_config=None):
    """
    Create SBD configuration object from dictionary.
    
    Args:
        config_dict: Dictionary of configuration parameters
        backend: SBD backend module to use (if None, uses default)
        device_config: DeviceConfig object for GPU settings
        
    Returns:
        SBD configuration object
    """
    if backend is None:
        backend = sbd
    
    sbd_data = backend.TPB_SBD()
    
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
    
    # Apply device configuration if provided
    if device_config is not None:
        device_config.apply(sbd_data)
    
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
