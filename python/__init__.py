"""
SBD (Selected Basis Diagonalization) Python Bindings

This package provides Python bindings for the SBD library.

Usage:
    import sbd
    sbd.init()                          # initialize MPI, auto-detect device
    results = sbd.tpb_diag_from_files(fcidump, adets, config)
    sbd.finalize()

Device switching (CPU/GPU) within the same process:
    sbd.init()
    result_cpu = sbd.tpb_diag(..., device='cpu')
    result_gpu = sbd.tpb_diag(..., device='gpu')
"""

import os
import subprocess

__version__ = "1.5.0"

# ---------------------------------------------------------------------------
# Backend registry — eagerly load all available backends at import time.
# Both _core_cpu and _core_gpu can coexist: they are separate .so files
# with separate pybind11 namespaces, no global C++ state conflicts.
# ---------------------------------------------------------------------------
_backends = {}

try:
    from . import _core_cpu
    _backends['cpu'] = _core_cpu
except ImportError:
    pass

try:
    from . import _core_gpu
    _backends['gpu'] = _core_gpu
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Global session state
# ---------------------------------------------------------------------------
_default_device = None   # set by init(), can be overridden per-call
_comm_backend = None     # 'mpi'
_comm_module = None      # mpi4py.MPI module
_global_comm = None      # MPI communicator
_initialized = False

# Cache GPU detection to avoid repeated subprocess calls
_gpu_check_cache = None


def _gpu_available():
    """Check if GPU is available via nvidia-smi (cached)."""
    global _gpu_check_cache
    if _gpu_check_cache is not None:
        return _gpu_check_cache
    try:
        result = subprocess.run(
            ['nvidia-smi'], capture_output=True, timeout=2
        )
        _gpu_check_cache = result.returncode == 0
    except Exception:
        _gpu_check_cache = False
    return _gpu_check_cache


def _resolve_device(device):
    """Resolve 'auto' to a concrete device name."""
    if device == 'auto':
        if 'gpu' in _backends and _gpu_available():
            return 'gpu'
        return 'cpu'
    if device in ('gpu', 'cuda'):
        return 'gpu'
    return device


def init(device='auto', comm_backend='mpi'):
    """
    Initialize SBD with MPI and set the default compute device.

    The device can be overridden per-call via the ``device`` parameter on
    ``tpb_diag()``, ``tpb_diag_from_files()``, and ``get_backend()``.

    Args:
        device: Default compute device — 'cpu', 'gpu', 'cuda', or 'auto'.
        comm_backend: Communication backend — 'mpi'.

    Raises:
        RuntimeError: If MPI is not available or no backends are compiled.
    """
    global _default_device, _comm_backend, _comm_module, _global_comm, _initialized

    if _initialized:
        raise RuntimeError(
            "sbd.init() already called. Call sbd.finalize() first to reinitialize."
        )

    if not _backends:
        raise RuntimeError(
            "No SBD backends available. Build with:\n"
            "  pip install -e . --no-build-isolation      (CPU)\n"
            "  SBD_BUILD_BACKEND=gpu pip install -e . --no-build-isolation  (GPU)"
        )

    # MPI setup
    if comm_backend == 'mpi':
        try:
            from mpi4py import MPI
            _comm_module = MPI
            _global_comm = MPI.COMM_WORLD
            _comm_backend = 'mpi'
        except ImportError:
            raise RuntimeError(
                "MPI backend requires mpi4py. Install with: pip install mpi4py"
            )
    else:
        raise ValueError(f"Unknown comm_backend: '{comm_backend}'. Supported: 'mpi'")

    # Resolve default device
    resolved = _resolve_device(device)
    if resolved not in _backends:
        available = list(_backends.keys())
        raise RuntimeError(
            f"Device '{resolved}' requested but backend not available. "
            f"Available: {available}"
        )
    _default_device = resolved
    _initialized = True

    # Print init info on rank 0
    rank = _global_comm.Get_rank()
    size = _global_comm.Get_size()
    if rank == 0:
        print(f"SBD initialized:")
        print(f"  Default device: {_default_device}")
        print(f"  Available backends: {list(_backends.keys())}")
        print(f"  Communication: {comm_backend}")
        print(f"  MPI ranks: {size}")
        print(f"  Version: {__version__}")


def finalize():
    """
    Finalize SBD and reset session state.

    Synchronizes GPU (if used) but does NOT call MPI_Finalize — mpi4py
    handles MPI lifecycle automatically.

    After finalize(), init() can be called again.
    """
    global _default_device, _comm_backend, _comm_module, _global_comm, _initialized

    # Synchronize GPU backends
    for name, backend in _backends.items():
        if name == 'gpu' and hasattr(backend, 'cleanup_device'):
            try:
                backend.cleanup_device()
            except Exception:
                pass

    _default_device = None
    _comm_backend = None
    _comm_module = None
    _global_comm = None
    _initialized = False


def is_initialized():
    """Check if SBD has been initialized."""
    return _initialized


# ---------------------------------------------------------------------------
# Backend access
# ---------------------------------------------------------------------------

def get_backend(device=None):
    """
    Get the backend module for the given device.

    Can be called anytime after init(). Passing ``device`` overrides the
    default set by init() — this is how you switch between CPU and GPU
    within the same process.

    Args:
        device: 'cpu', 'gpu', 'auto', or None (use default).

    Returns:
        The pybind11 backend module (_core_cpu or _core_gpu).
    """
    if device is None:
        device = _default_device or 'auto'
    device = _resolve_device(device)
    if device not in _backends:
        available = list(_backends.keys())
        raise RuntimeError(
            f"Backend '{device}' not available. Available: {available}"
        )
    return _backends[device]


def _check_initialized():
    if not _initialized:
        raise RuntimeError("SBD not initialized. Call sbd.init() first.")


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

def get_device():
    """Get the default compute device name."""
    _check_initialized()
    return _default_device


def get_comm_backend():
    """Get the communication backend name."""
    _check_initialized()
    return _comm_backend


def get_rank():
    """Get MPI rank of current process."""
    _check_initialized()
    return _global_comm.Get_rank()


def get_world_size():
    """Get total number of MPI processes."""
    _check_initialized()
    return _global_comm.Get_size()


def get_comm():
    """Get the MPI communicator."""
    _check_initialized()
    return _global_comm


def barrier():
    """MPI barrier — synchronize all processes."""
    _check_initialized()
    _global_comm.Barrier()


# ---------------------------------------------------------------------------
# Wrapper functions — forward to the selected backend
# ---------------------------------------------------------------------------

def TPB_SBD(device=None):
    """Create TPB_SBD configuration object."""
    _check_initialized()
    return get_backend(device).TPB_SBD()


def FCIDump(device=None):
    """Create FCIDump object."""
    _check_initialized()
    return get_backend(device).FCIDump()


def LoadFCIDump(filename, device=None):
    """Load FCIDUMP file."""
    _check_initialized()
    return get_backend(device).LoadFCIDump(filename)


def LoadAlphaDets(filename, bit_length, total_bit_length, device=None):
    """Load alpha determinants from file."""
    _check_initialized()
    return get_backend(device).LoadAlphaDets(filename, bit_length, total_bit_length)


def makestring(config, bit_length, total_bit_length, device=None):
    """Convert determinant to string representation."""
    _check_initialized()
    return get_backend(device).makestring(config, bit_length, total_bit_length)


def from_string(s, bit_length, total_bit_length, device=None):
    """Convert binary string to determinant format."""
    _check_initialized()
    return get_backend(device).from_string(s, bit_length, total_bit_length)


def tpb_diag_from_files(fcidumpfile, adetfile, sbd_data,
                        loadname="", savename="", device=None):
    """
    Perform TPB diagonalization from files.

    Args:
        fcidumpfile: Path to FCIDUMP file.
        adetfile: Path to alpha determinants file.
        sbd_data: TPB_SBD configuration object.
        loadname: Path to load initial wavefunction (optional).
        savename: Path to save final wavefunction (optional).
        device: Override device ('cpu', 'gpu', or None for default).

    Returns:
        dict with keys: energy, density, carryover_adet, carryover_bdet,
        one_p_rdm, two_p_rdm.
    """
    _check_initialized()
    backend = get_backend(device)
    return backend.tpb_diag_from_files(
        _global_comm, sbd_data, fcidumpfile, adetfile, loadname, savename
    )


def tpb_diag(fcidump, adet, bdet, sbd_data,
             loadname="", savename="", device=None):
    """
    Perform TPB diagonalization with data structures.

    Args:
        fcidump: FCIDump object.
        adet: Alpha determinants.
        bdet: Beta determinants.
        sbd_data: TPB_SBD configuration object.
        loadname: Path to load initial wavefunction (optional).
        savename: Path to save final wavefunction (optional).
        device: Override device ('cpu', 'gpu', or None for default).

    Returns:
        dict with keys: energy, density, carryover_adet, carryover_bdet,
        one_p_rdm, two_p_rdm.
    """
    _check_initialized()
    backend = get_backend(device)
    return backend.tpb_diag(
        _global_comm, sbd_data, fcidump, adet, bdet, loadname, savename
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def available_backends():
    """Get list of compiled backends ('cpu', 'gpu')."""
    return list(_backends.keys())


def print_info():
    """Print SBD information."""
    print("=" * 60)
    print("SBD (Selected Basis Diagonalization) Python Bindings")
    print("=" * 60)
    print(f"Version: {__version__}")
    print(f"Compiled backends: {', '.join(available_backends()) or 'none'}")

    if _initialized:
        print(f"\nCurrent session:")
        print(f"  Default device: {_default_device}")
        print(f"  Communication: {_comm_backend}")
        print(f"  MPI rank: {get_rank()}/{get_world_size()}")
    else:
        print(f"\nNot initialized. Call sbd.init() to start.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

from . import sbd_solver

__all__ = [
    # Initialization
    'init',
    'finalize',
    'is_initialized',

    # Backend access
    'get_backend',

    # Query
    'get_device',
    'get_comm_backend',
    'get_rank',
    'get_world_size',
    'get_comm',
    'barrier',

    # Main API
    'TPB_SBD',
    'FCIDump',
    'LoadFCIDump',
    'LoadAlphaDets',
    'makestring',
    'from_string',
    'tpb_diag_from_files',
    'tpb_diag',

    # Utilities
    'available_backends',
    'print_info',

    # Sub-modules
    'sbd_solver',

    # Version
    '__version__',
]
