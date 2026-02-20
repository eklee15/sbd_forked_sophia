"""
SBD (Selected Basis Diagonalization) Python Bindings

This package provides Python bindings for the SBD library with simplified API.

Two usage modes:

1. Simplified API (Recommended - no mpi4py needed):
    import sbd
    sbd.init(device='gpu', comm_backend='mpi')
    results = sbd.tpb_diag_from_files(...)
    sbd.finalize()

2. Legacy API (backward compatible):
    from mpi4py import MPI
    import sbd
    results = sbd.tpb_diag_from_files(comm=MPI.COMM_WORLD, ...)
"""

import sys
import os
import subprocess

# Version info
__version__ = "1.4.0"

# Global state for simplified API
_device_module = None      # _core_cpu or _core_gpu
_comm_backend = None       # 'mpi', 'nccl', etc.
_comm_module = None        # MPI module from mpi4py
_global_comm = None        # MPI communicator
_initialized = False       # Whether init() was called

def _gpu_available():
    """Check if GPU is available via nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              timeout=2)
        return result.returncode == 0
    except:
        return False

def init(device='auto', comm_backend='mpi'):
    """
    Initialize SBD with device and communication backend.
    
    This is the simplified API that hides MPI initialization from user code.
    After calling init(), you can use sbd functions without passing comm parameter.
    
    Args:
        device (str): Compute device - 'cpu', 'gpu', 'cuda', or 'auto' (default: 'auto')
                     'auto' will use GPU if available, otherwise CPU
        comm_backend (str): Communication backend - 'mpi' (default: 'mpi')
                           Future: 'nccl' for GPU-only communication
    
    Example:
        import sbd
        sbd.init(device='gpu', comm_backend='mpi')
        config = sbd.TPB_SBD()
        results = sbd.tpb_diag_from_files(...)
        sbd.finalize()
    
    Raises:
        RuntimeError: If required dependencies are not available
        ValueError: If invalid device or comm_backend specified
    """
    global _device_module, _comm_backend, _comm_module, _global_comm, _initialized
    
    if _initialized:
        raise RuntimeError("sbd.init() already called. Call sbd.finalize() first to reinitialize.")
    
    # 1. Initialize communication backend
    if comm_backend == 'mpi':
        try:
            from mpi4py import MPI
            _comm_module = MPI
            _global_comm = MPI.COMM_WORLD
            _comm_backend = 'mpi'
        except ImportError:
            raise RuntimeError(
                "MPI backend requires mpi4py.\n"
                "Install with: pip install mpi4py"
            )
    elif comm_backend == 'nccl':
        raise NotImplementedError(
            "NCCL backend not yet implemented.\n"
            "Currently only 'mpi' is supported."
        )
    else:
        raise ValueError(
            f"Unknown comm_backend: '{comm_backend}'.\n"
            f"Supported: 'mpi' (future: 'nccl')"
        )
    
    # 2. Select compute device
    if device == 'auto':
        device = 'gpu' if _gpu_available() else 'cpu'
        if os.environ.get('SBD_VERBOSE', '0') == '1':
            print(f"SBD: Auto-selected device: {device}")
    
    if device in ['gpu', 'cuda']:
        try:
            from . import _core_gpu
            _device_module = _core_gpu
            device_name = 'gpu'
        except ImportError as e:
            raise RuntimeError(
                f"GPU device requires _core_gpu.so to be built.\n"
                f"Build with: SBD_BUILD_BACKEND=gpu pip install -e .\n"
                f"Error: {e}"
            )
    elif device == 'cpu':
        try:
            from . import _core_cpu
            _device_module = _core_cpu
            device_name = 'cpu'
        except ImportError as e:
            raise RuntimeError(
                f"CPU device requires _core_cpu.so to be built.\n"
                f"Build with: pip install -e .\n"
                f"Error: {e}"
            )
    else:
        raise ValueError(
            f"Unknown device: '{device}'.\n"
            f"Supported: 'cpu', 'gpu', 'cuda', 'auto'"
        )
    
    _initialized = True
    
    # Print initialization info
    rank = _global_comm.Get_rank()
    size = _global_comm.Get_size()
    
    if rank == 0 or os.environ.get('SBD_VERBOSE', '0') == '1':
        print(f"SBD initialized:")
        print(f"  Device: {device_name}")
        print(f"  Communication: {comm_backend}")
        print(f"  MPI ranks: {size}")
        if rank == 0:
            print(f"  Version: {__version__}")

def finalize():
    """
    Finalize SBD and clean up internal state.
    
    This function:
    - Synchronizes GPU device (if using GPU backend)
    - Resets internal Python state
    - Does NOT call MPI_Finalize() - MPI lifecycle is managed by mpi4py
    
    After calling finalize(), you can call init() again with different parameters.
    
    Note: Similar to torch.distributed.destroy_process_group(), this ensures
    proper cleanup of distributed computing resources.
    
    GPU Note: This calls cudaDeviceSynchronize() but NOT cudaDeviceReset() to
    avoid conflicts with CUDA-aware MPI (UCX). GPU resources are freed automatically
    when the process exits.
    """
    global _device_module, _comm_backend, _comm_module, _global_comm, _initialized
    
    # Synchronize GPU device if using GPU backend
    if _device_module is not None and hasattr(_device_module, 'cleanup_device'):
        try:
            _device_module.cleanup_device()
        except Exception as e:
            import warnings
            warnings.warn(f"GPU synchronization failed: {e}")
    
    # Reset Python state
    _device_module = None
    _comm_backend = None
    _comm_module = None
    _global_comm = None
    _initialized = False

def is_initialized():
    """Check if SBD has been initialized"""
    return _initialized

def finalize_mpi():
    """
    Explicitly finalize MPI.
    
    WARNING: Only call this if you initialized MPI yourself outside of mpi4py.
    If using mpi4py (recommended), MPI finalization is handled automatically at exit.
    
    This is provided for advanced use cases where explicit MPI lifecycle control is needed.
    Similar to calling MPI_Finalize() in C/C++ code.
    
    Note: After calling this, you cannot use any MPI functions until MPI is reinitialized.
    """
    if _device_module is not None and hasattr(_device_module, 'finalize_mpi'):
        _device_module.finalize_mpi()
    else:
        raise RuntimeError("SBD not initialized. Call sbd.init() first.")

# ============================================================================
# MPI Communication Primitives
# ============================================================================

def _check_initialized():
    """Check if SBD is initialized"""
    if not _initialized:
        raise RuntimeError(
            "SBD not initialized. Call sbd.init() before using communication primitives."
        )

def broadcast(data, root=0):
    """
    Broadcast data from root rank to all other ranks.
    
    Similar to torch.distributed.broadcast(), this sends data from one rank
    to all other ranks. Useful for distributing quantum samples from rank 0.
    
    Args:
        data: Data to broadcast (any picklable Python object)
        root: Source rank (default: 0)
    
    Returns:
        Broadcasted data on all ranks
    
    Example:
        # Rank 0 has quantum samples
        if sbd.get_rank() == 0:
            samples = {'bitstrings': [...], 'counts': [...]}
        else:
            samples = None
        
        # Broadcast to all ranks
        samples = sbd.broadcast(samples, root=0)
        # Now all ranks have the samples
    """
    _check_initialized()
    data = _global_comm.bcast(data, root=root)
    return data

def gather(data, root=0):
    """
    Gather data from all ranks to root rank.
    
    Similar to torch.distributed.gather(), this collects data from all ranks
    to a single root rank. Useful for collecting results.
    
    Args:
        data: Local data from this rank
        root: Destination rank (default: 0)
    
    Returns:
        List of data from all ranks (only on root), None on other ranks
    
    Example:
        local_energy = compute_local_energy()
        all_energies = sbd.gather(local_energy, root=0)
        
        if sbd.get_rank() == 0:
            avg_energy = sum(all_energies) / len(all_energies)
    """
    _check_initialized()
    gathered = _global_comm.gather(data, root=root)
    return gathered

def all_gather(data):
    """
    Gather data from all ranks to all ranks.
    
    Similar to torch.distributed.all_gather(), this collects data from all
    ranks and distributes the complete list to all ranks.
    
    Args:
        data: Local data from this rank
    
    Returns:
        List of data from all ranks (on all ranks)
    
    Example:
        local_dets = recover_determinants()
        all_dets = sbd.all_gather(local_dets)
        # Now every rank has determinants from all ranks
    """
    _check_initialized()
    all_data = _global_comm.allgather(data)
    return all_data

def reduce(data, op='sum', root=0):
    """
    Reduce data from all ranks using specified operation.
    
    Similar to torch.distributed.reduce(), this combines data from all ranks
    using a reduction operation and sends result to root.
    
    Args:
        data: Local data (must be numeric)
        op: Operation ('sum', 'prod', 'max', 'min', 'avg')
        root: Destination rank (default: 0)
    
    Returns:
        Reduced result (only on root), None on other ranks
    
    Example:
        local_count = len(local_determinants)
        total_count = sbd.reduce(local_count, op='sum', root=0)
        
        if sbd.get_rank() == 0:
            print(f"Total determinants: {total_count}")
    """
    _check_initialized()
    
    from mpi4py import MPI
    
    # Map operation names to MPI operations
    op_map = {
        'sum': MPI.SUM,
        'prod': MPI.PROD,
        'max': MPI.MAX,
        'min': MPI.MIN,
    }
    
    if op == 'avg':
        # Average is sum divided by size
        result = _global_comm.reduce(data, op=MPI.SUM, root=root)
        if _global_comm.Get_rank() == root:
            result = result / _global_comm.Get_size()
        return result
    elif op in op_map:
        return _global_comm.reduce(data, op=op_map[op], root=root)
    else:
        raise ValueError(f"Unknown operation: {op}. Use 'sum', 'prod', 'max', 'min', or 'avg'")

def all_reduce(data, op='sum'):
    """
    Reduce data from all ranks and distribute to all ranks.
    
    Similar to torch.distributed.all_reduce(), this combines data from all
    ranks and distributes the result to all ranks.
    
    Args:
        data: Local data (must be numeric)
        op: Operation ('sum', 'prod', 'max', 'min', 'avg')
    
    Returns:
        Reduced result (on all ranks)
    
    Example:
        local_iterations = get_iterations()
        avg_iterations = sbd.all_reduce(local_iterations, op='avg')
        print(f"Rank {sbd.get_rank()}: Average iterations = {avg_iterations}")
    """
    _check_initialized()
    
    from mpi4py import MPI
    
    op_map = {
        'sum': MPI.SUM,
        'prod': MPI.PROD,
        'max': MPI.MAX,
        'min': MPI.MIN,
    }
    
    if op == 'avg':
        result = _global_comm.allreduce(data, op=MPI.SUM)
        return result / _global_comm.Get_size()
    elif op in op_map:
        return _global_comm.allreduce(data, op=op_map[op])
    else:
        raise ValueError(f"Unknown operation: {op}. Use 'sum', 'prod', 'max', 'min', or 'avg'")

def get_device():
    """
    Get current compute device.
    
    Returns:
        str: 'cpu' or 'gpu'
    
    Raises:
        RuntimeError: If init() has not been called
    """
    if not _initialized:
        raise RuntimeError("Call sbd.init() first")
    return 'gpu' if '_core_gpu' in _device_module.__name__ else 'cpu'

def get_comm_backend():
    """
    Get current communication backend.
    
    Returns:
        str: 'mpi' (future: 'nccl')
    
    Raises:
        RuntimeError: If init() has not been called
    """
    if not _initialized:
        raise RuntimeError("Call sbd.init() first")
    return _comm_backend

def get_rank():
    """
    Get MPI rank of current process.
    
    Returns:
        int: MPI rank (0 to world_size-1)
    
    Raises:
        RuntimeError: If init() has not been called
    """
    if not _initialized:
        raise RuntimeError("Call sbd.init() first")
    return _global_comm.Get_rank()

def get_world_size():
    """
    Get total number of MPI processes.
    
    Returns:
        int: Number of MPI ranks
    
    Raises:
        RuntimeError: If init() has not been called
    """
    if not _initialized:
        raise RuntimeError("Call sbd.init() first")
    return _global_comm.Get_size()

def barrier():
    """
    MPI barrier - synchronize all processes.
    
    Raises:
        RuntimeError: If init() has not been called
    """
    if not _initialized:
        raise RuntimeError("Call sbd.init() first")
    _global_comm.Barrier()

# Simplified API wrapper functions
def TPB_SBD():
    """
    Create TPB_SBD configuration object.
    
    Returns:
        TPB_SBD: Configuration object for TPB diagonalization
    
    Raises:
        RuntimeError: If init() has not been called
    """
    if not _initialized:
        raise RuntimeError("Call sbd.init() first")
    return _device_module.TPB_SBD()

def FCIDump():
    """
    Create FCIDump object.
    
    Returns:
        FCIDump: Object for FCIDUMP data
    
    Raises:
        RuntimeError: If init() has not been called
    """
    if not _initialized:
        raise RuntimeError("Call sbd.init() first")
    return _device_module.FCIDump()

def LoadFCIDump(filename):
    """
    Load FCIDUMP file.
    
    Args:
        filename (str): Path to FCIDUMP file
    
    Returns:
        FCIDump: Loaded FCIDUMP object
    
    Raises:
        RuntimeError: If init() has not been called
    """
    if not _initialized:
        raise RuntimeError("Call sbd.init() first")
    return _device_module.LoadFCIDump(filename)

def LoadAlphaDets(filename, bit_length, total_bit_length):
    """
    Load alpha determinants from file.
    
    Args:
        filename (str): Path to determinants file
        bit_length (int): Bit length for determinants
        total_bit_length (int): Total bit length
    
    Returns:
        list: List of determinants
    
    Raises:
        RuntimeError: If init() has not been called
    """
    if not _initialized:
        raise RuntimeError("Call sbd.init() first")
    return _device_module.LoadAlphaDets(filename, bit_length, total_bit_length)

def makestring(config, bit_length, total_bit_length):
    """
    Convert determinant to string representation.
    
    Args:
        config: Determinant configuration
        bit_length (int): Bit length
        total_bit_length (int): Total bit length
    
    Returns:
        str: String representation
    
    Raises:
        RuntimeError: If init() has not been called
    """
    if not _initialized:
        raise RuntimeError("Call sbd.init() first")
    return _device_module.makestring(config, bit_length, total_bit_length)

def from_string(s, bit_length, total_bit_length):
    """
    Convert binary string to determinant format.
    
    Args:
        s (str): Binary string (e.g., "11111000000000000000")
        bit_length (int): Bit length per size_t
        total_bit_length (int): Total bit length
    
    Returns:
        list: Determinant in SBD format (list of size_t)
    
    Raises:
        RuntimeError: If init() has not been called
    
    Example:
        >>> import sbd
        >>> sbd.init()
        >>> det = sbd.from_string("11111", bit_length=20, total_bit_length=10)
        >>> print(det)  # [31] (binary 11111 = decimal 31)
    """
    if not _initialized:
        raise RuntimeError("Call sbd.init() first")
    return _device_module.from_string(s, bit_length, total_bit_length)

def tpb_diag_from_files(fcidumpfile, adetfile, sbd_data, loadname="", savename=""):
    """
    Perform TPB diagonalization from files (simplified API).
    
    This function uses the internal MPI communicator initialized by sbd.init().
    No need to pass comm parameter.
    
    Args:
        fcidumpfile (str): Path to FCIDUMP file
        adetfile (str): Path to alpha determinants file
        sbd_data (TPB_SBD): Configuration object
        loadname (str): Path to load initial wavefunction (optional)
        savename (str): Path to save final wavefunction (optional)
    
    Returns:
        dict: Results dictionary with keys:
            - 'energy': Ground state energy
            - 'density': Orbital densities
            - 'carryover_adet': Carryover alpha determinants
            - 'carryover_bdet': Carryover beta determinants
            - 'one_p_rdm': 1-particle RDM (if do_rdm=1)
            - 'two_p_rdm': 2-particle RDM (if do_rdm=1)
    
    Raises:
        RuntimeError: If init() has not been called
    
    Example:
        import sbd
        sbd.init(device='gpu')
        config = sbd.TPB_SBD()
        config.max_it = 100
        config.eps = 1e-6
        results = sbd.tpb_diag_from_files("fcidump.txt", "alphadets.txt", config)
        print(f"Energy: {results['energy']}")
        sbd.finalize()
    """
    if not _initialized:
        raise RuntimeError("Call sbd.init() first")
    
    return _device_module.tpb_diag_from_files(
        _global_comm, sbd_data, fcidumpfile, adetfile, loadname, savename
    )

def tpb_diag(fcidump, adet, bdet, sbd_data, loadname="", savename=""):
    """
    Perform TPB diagonalization with data structures (simplified API).
    
    This function uses the internal MPI communicator initialized by sbd.init().
    No need to pass comm parameter.
    
    Args:
        fcidump (FCIDump): FCIDUMP object
        adet (list): Alpha determinants
        bdet (list): Beta determinants
        sbd_data (TPB_SBD): Configuration object
        loadname (str): Path to load initial wavefunction (optional)
        savename (str): Path to save final wavefunction (optional)
    
    Returns:
        dict: Results dictionary (same as tpb_diag_from_files)
    
    Raises:
        RuntimeError: If init() has not been called
    """
    if not _initialized:
        raise RuntimeError("Call sbd.init() first")
    
    return _device_module.tpb_diag(
        _global_comm, sbd_data, fcidump, adet, bdet, loadname, savename
    )

# Legacy API support (backward compatibility)
def tpb_diag_from_files_legacy(comm, sbd_data, fcidumpfile, adetfile, loadname="", savename=""):
    """
    Legacy API: TPB diagonalization from files with explicit comm parameter.
    
    This is for backward compatibility. New code should use the simplified API.
    
    Args:
        comm: MPI communicator (from mpi4py)
        ... (other args same as simplified API)
    
    Returns:
        dict: Results dictionary
    """
    # Determine which backend to use based on environment or auto-detect
    backend_name = os.environ.get('SBD_BACKEND', 'auto').lower()
    
    if backend_name == 'auto':
        backend_name = 'gpu' if _gpu_available() else 'cpu'
    
    if backend_name == 'gpu':
        from . import _core_gpu as backend
    else:
        from . import _core_cpu as backend
    
    return backend.tpb_diag_from_files(comm, sbd_data, fcidumpfile, adetfile, loadname, savename)

def tpb_diag_legacy(comm, sbd_data, fcidump, adet, bdet, loadname="", savename=""):
    """
    Legacy API: TPB diagonalization with explicit comm parameter.
    
    This is for backward compatibility. New code should use the simplified API.
    """
    backend_name = os.environ.get('SBD_BACKEND', 'auto').lower()
    
    if backend_name == 'auto':
        backend_name = 'gpu' if _gpu_available() else 'cpu'
    
    if backend_name == 'gpu':
        from . import _core_gpu as backend
    else:
        from . import _core_cpu as backend
    
    return backend.tpb_diag(comm, sbd_data, fcidump, adet, bdet, loadname, savename)

# Utility functions
def available_backends():
    """
    Get list of backends that were compiled.
    
    Returns:
        list: List of backend names ('cpu', 'gpu')
    """
    import glob
    
    backends = []
    module_dir = os.path.dirname(__file__)
    
    if glob.glob(os.path.join(module_dir, '_core_cpu*.so')):
        backends.append('cpu')
    
    if glob.glob(os.path.join(module_dir, '_core_gpu*.so')):
        backends.append('gpu')
    
    return backends

def print_info():
    """Print SBD information"""
    print("="*70)
    print("SBD (Selected Basis Diagonalization) Python Bindings")
    print("="*70)
    print(f"Version: {__version__}")
    print(f"Compiled backends: {', '.join(available_backends())}")
    
    if _initialized:
        print(f"\nCurrent session:")
        print(f"  Device: {get_device()}")
        print(f"  Communication: {get_comm_backend()}")
        print(f"  MPI rank: {get_rank()}/{get_world_size()}")
    else:
        print(f"\nNot initialized. Call sbd.init() to start.")
    
    print("\nUsage:")
    print("  import sbd")
    print("  sbd.init(device='gpu', comm_backend='mpi')")
    print("  results = sbd.tpb_diag_from_files(...)")
    print("  sbd.finalize()")
    print("="*70)

__all__ = [
    # Initialization
    'init',
    'finalize',
    'finalize_mpi',
    'is_initialized',
    
    # MPI Communication Primitives
    'broadcast',
    'gather',
    'all_gather',
    'reduce',
    'all_reduce',
    
    # Query functions
    'get_device',
    'get_comm_backend',
    'get_rank',
    'get_world_size',
    'barrier',
    
    # Main API
    'TPB_SBD',
    'FCIDump',
    'LoadFCIDump',
    'LoadAlphaDets',
    'makestring',
    'tpb_diag_from_files',
    'tpb_diag',
    
    # Legacy API
    'tpb_diag_from_files_legacy',
    'tpb_diag_legacy',
    
    # Utilities
    'available_backends',
    'print_info',
    
    # Version
    '__version__',
]

# Made with Bob
