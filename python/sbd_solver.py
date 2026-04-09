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

try:
    from pyscf import tools as pyscf_tools
except ImportError:
    pyscf_tools = None

try:
    from qiskit_addon_sqd.fermion import SCIResult, SCIState
except ImportError:
    SCIResult = None
    SCIState = None


def _resolve_backend(device_config=None):
    """
    Resolve a backend module from a DeviceConfig or the default.

    Uses sbd.get_backend() which supports runtime CPU/GPU switching.
    """
    from . import get_backend
    if device_config is not None:
        device = 'gpu' if device_config.use_gpu else 'cpu'
        return get_backend(device)
    return get_backend()


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
    device_config=None,
) -> SCIResult:
    """
    Diagonalize Hamiltonian in subspace defined by CI strings using SBD.

    Args:
        ci_strings: Pair (strings_a, strings_b) of CI string arrays.
        one_body_tensor: The one-body tensor of the Hamiltonian.
        two_body_tensor: The two-body tensor of the Hamiltonian.
        norb: The number of spatial orbitals.
        nelec: The numbers of alpha and beta electrons.
        spin_sq: Target value for total spin squared (unused by SBD).
        mpi_comm: MPI communicator. If None, uses MPI.COMM_WORLD.
        sbd_config: Dictionary of SBD configuration parameters.
        temp_dir: Directory for temporary files.
        clean_temp_dir: Whether to delete intermediate files.
        device_config: DeviceConfig object to select CPU/GPU backend.

    Returns:
        The diagonalization result as SCIResult.
    """
    if SCIResult is None:
        raise ImportError(
            "qiskit-addon-sqd is required for solve_sci. "
            "Install with: pip install qiskit-addon-sqd"
        )
    backend = _resolve_backend(device_config)

    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()

    # Rank 0 creates temp dir, broadcasts to all ranks
    temp_dir = temp_dir or tempfile.gettempdir()
    if mpi_rank == 0:
        sbd_dir = Path(tempfile.mkdtemp(prefix="sbd_files_", dir=temp_dir))
        sbd_dir_str = str(sbd_dir)
    else:
        sbd_dir_str = None
    sbd_dir_str = mpi_comm.bcast(sbd_dir_str, root=0)
    sbd_dir = Path(sbd_dir_str)

    try:
        fcidump_path = sbd_dir / "fcidump.txt"
        if mpi_rank == 0:
            pyscf_tools.fcidump.from_integrals(
                str(fcidump_path), one_body_tensor, two_body_tensor, norb, nelec,
            )
        mpi_comm.Barrier()

        fcidump = backend.LoadFCIDump(str(fcidump_path))

        return _solve_sci_core(
            ci_strings,
            norb=norb,
            nelec=nelec,
            spin_sq=spin_sq,
            mpi_comm=mpi_comm,
            mpi_rank=mpi_rank,
            sbd_config=sbd_config,
            sbd_dir=sbd_dir,
            backend=backend,
            fcidump=fcidump,
            device_config=device_config,
        )
    finally:
        if clean_temp_dir and mpi_rank == 0:
            shutil.rmtree(sbd_dir, ignore_errors=True)


def _solve_sci_core(
    ci_strings: tuple[np.ndarray, np.ndarray],
    *,
    norb: int,
    nelec: tuple[int, int],
    spin_sq: float | None,
    mpi_comm,
    mpi_rank: int,
    sbd_config: dict | None,
    sbd_dir: Path,
    backend,
    fcidump,
    device_config=None,
) -> SCIResult:
    """
    Inner diagonalization kernel that operates on a pre-loaded FCIDUMP object.

    Separated from solve_sci so that solve_sci_batch can write and load
    the FCIDUMP only once and reuse it across all batches.
    """
    strings_a, strings_b = ci_strings
    adet = _ci_strings_to_sbd_dets(strings_a, norb, backend)
    bdet = _ci_strings_to_sbd_dets(strings_b, norb, backend)

    sbd_data = _create_sbd_config(sbd_config, backend, device_config)

    # Use .bin extension to trigger SBD's fast binary write path
    # (SaveMatrixFormWF in restart.h checks extension: .bin -> raw doubles)
    wf_dump_file = sbd_dir / "wavefunction.bin"
    sbd_data.dump_matrix_form_wf = str(wf_dump_file)

    results = backend.tpb_diag(
        mpi_comm, sbd_data, fcidump, adet, bdet, loadname="", savename=""
    )

    # Rank 0 reads the wavefunction file; Barrier ensures it's flushed.
    mpi_comm.Barrier()

    if mpi_rank != 0:
        return SCIResult(
            0.0,
            SCIState(
                amplitudes=np.empty((0, 0), dtype=np.float64),
                ci_strs_a=np.array([], dtype=np.int64),
                ci_strs_b=np.array([], dtype=np.int64),
                norb=norb,
                nelec=nelec,
            ),
            orbital_occupancies=(
                np.zeros(norb, dtype=np.float64),
                np.zeros(norb, dtype=np.float64),
            ),
        )

    # --- rank 0 only ---

    energy = results["energy"]
    density = np.array(results["density"])
    occupancies_a = density[::2]
    occupancies_b = density[1::2]
    occupancies = (occupancies_a, occupancies_b)

    co_strings_a = _sbd_dets_to_ci_strings(results["carryover_adet"], norb, backend)
    co_strings_b = _sbd_dets_to_ci_strings(results["carryover_bdet"], norb, backend)

    # Read wavefunction coefficients from binary dump
    n_alpha_co = len(co_strings_a)
    n_beta_co = len(co_strings_b)
    amplitudes = None
    if n_alpha_co > 0 and n_beta_co > 0 and wf_dump_file.exists():
        flat = np.fromfile(str(wf_dump_file), dtype=np.float64)
        if flat.size == n_alpha_co * n_beta_co:
            amplitudes = flat.reshape(n_alpha_co, n_beta_co)

    # Build SCIState with fallbacks if wavefunction file is missing/malformed
    if amplitudes is not None:
        sci_state = SCIState(
            amplitudes=amplitudes,
            ci_strs_a=co_strings_a,
            ci_strs_b=co_strings_b,
            norb=norb,
            nelec=nelec,
        )
    elif n_alpha_co > 0 and n_beta_co > 0:
        amplitudes = np.ones((n_alpha_co, n_beta_co)) / np.sqrt(n_alpha_co * n_beta_co)
        sci_state = SCIState(
            amplitudes=amplitudes,
            ci_strs_a=co_strings_a,
            ci_strs_b=co_strings_b,
            norb=norb,
            nelec=nelec,
        )
    else:
        n_a = len(strings_a)
        n_b = len(strings_b)
        amplitudes = np.ones((n_a, n_b)) / np.sqrt(n_a * n_b)
        sci_state = SCIState(
            amplitudes=amplitudes,
            ci_strs_a=strings_a,
            ci_strs_b=strings_b,
            norb=norb,
            nelec=nelec,
        )

    return SCIResult(energy, sci_state, orbital_occupancies=occupancies)


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
    device_config=None,
) -> list[SCIResult]:
    """
    Diagonalize Hamiltonian in multiple subspaces using SBD.

    The FCIDUMP file is written once and loaded once for all batches.

    Args:
        ci_strings: List of (strings_a, strings_b) pairs.
        one_body_tensor: The one-body tensor of the Hamiltonian.
        two_body_tensor: The two-body tensor of the Hamiltonian.
        norb: The number of spatial orbitals.
        nelec: The numbers of alpha and beta electrons.
        spin_sq: Target value for total spin squared (unused by SBD).
        mpi_comm: MPI communicator. If None, uses MPI.COMM_WORLD.
        sbd_config: Dictionary of SBD configuration parameters.
        temp_dir: Directory for temporary files.
        clean_temp_dir: Whether to delete intermediate files.
        device_config: DeviceConfig object to select CPU/GPU backend.

    Returns:
        List of SCIResult for each batch.
    """
    if not ci_strings:
        return []

    backend = _resolve_backend(device_config)

    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()

    temp_dir = temp_dir or tempfile.gettempdir()
    if mpi_rank == 0:
        sbd_dir = Path(tempfile.mkdtemp(prefix="sbd_files_", dir=temp_dir))
        sbd_dir_str = str(sbd_dir)
    else:
        sbd_dir_str = None
    sbd_dir_str = mpi_comm.bcast(sbd_dir_str, root=0)
    sbd_dir = Path(sbd_dir_str)

    try:
        fcidump_path = sbd_dir / "fcidump.txt"
        if mpi_rank == 0:
            pyscf_tools.fcidump.from_integrals(
                str(fcidump_path), one_body_tensor, two_body_tensor, norb, nelec,
            )
        mpi_comm.Barrier()

        fcidump = backend.LoadFCIDump(str(fcidump_path))

        return [
            _solve_sci_core(
                ci_strs,
                norb=norb,
                nelec=nelec,
                spin_sq=spin_sq,
                mpi_comm=mpi_comm,
                mpi_rank=mpi_rank,
                sbd_config=sbd_config,
                sbd_dir=sbd_dir,
                backend=backend,
                fcidump=fcidump,
                device_config=device_config,
            )
            for ci_strs in ci_strings
        ]
    finally:
        if clean_temp_dir and mpi_rank == 0:
            shutil.rmtree(sbd_dir, ignore_errors=True)


def _ci_strings_to_sbd_dets(
    ci_strings: np.ndarray, norb: int, backend
) -> list[list[int]]:
    """Convert CI strings (integers) to SBD determinant format.

    Determinants are sorted in canonical order (matching C++ sort_bitarray)
    which is required by the GPU Correlation kernel (do_rdm=1).
    """
    bit_length = 64
    dets = []
    for ci_str in ci_strings:
        binary_str = format(int(ci_str), f'0{norb}b')
        det = backend.from_string(binary_str, bit_length, norb)
        dets.append(det)
    return backend.sort_bitarray(dets)


def _sbd_dets_to_ci_strings(
    dets: list[list[int]], norb: int, backend
) -> np.ndarray:
    """Convert SBD determinants to CI strings (integers)."""
    bit_length = 64
    ci_strings = []
    for det in dets:
        binary_str = backend.makestring(det, bit_length, norb)
        ci_str = int(binary_str, 2)
        ci_strings.append(ci_str)
    return np.array(ci_strings, dtype=np.int64)


def _create_sbd_config(config_dict: dict | None = None, backend=None, device_config=None):
    """Create SBD configuration object from dictionary."""
    if backend is None:
        backend = _resolve_backend(device_config)

    sbd_data = backend.TPB_SBD()

    # Defaults
    sbd_data.method = 0  # Davidson
    sbd_data.max_it = 100
    sbd_data.max_nb = 50
    sbd_data.eps = 1e-8
    sbd_data.max_time = 3600
    sbd_data.init = 0
    sbd_data.do_shuffle = 0
    sbd_data.do_rdm = 0
    sbd_data.carryover_type = 1
    sbd_data.ratio = 0.1
    sbd_data.threshold = 1e-4
    sbd_data.bit_length = 64

    if config_dict:
        for key, value in config_dict.items():
            if hasattr(sbd_data, key):
                setattr(sbd_data, key, value)

    if device_config is not None:
        device_config.apply(sbd_data)

    return sbd_data


def create_sbd_solver(
    mpi_comm: MPI.Comm | None = None,
    sbd_config: dict | None = None,
    temp_dir: str | Path | None = None,
    clean_temp_dir: bool = True,
    device_config=None,
) -> Callable:
    """
    Create a configured SBD solver function for use with
    diagonalize_fermionic_hamiltonian.

    Example:
        >>> from functools import partial
        >>> sbd_solver = create_sbd_solver(sbd_config={"method": 0, "eps": 1e-10})
        >>> result = diagonalize_fermionic_hamiltonian(
        ...     hcore, eri, bit_array, sci_solver=sbd_solver, ...
        ... )
    """
    from functools import partial

    return partial(
        solve_sci_batch,
        mpi_comm=mpi_comm,
        sbd_config=sbd_config,
        temp_dir=temp_dir,
        clean_temp_dir=clean_temp_dir,
        device_config=device_config,
    )
