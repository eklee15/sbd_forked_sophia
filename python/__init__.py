"""
SBD (Selected Basis Diagonalization) Python Bindings

This package provides Python bindings for the SBD library with backend selection.

IMPORTANT: Due to pybind11 limitations, you can only use ONE backend per Python process.
The backend is selected at import time and cannot be changed.

To select a backend, set the SBD_BACKEND environment variable BEFORE importing sbd:

    # CORRECT - set env var first
    import os
    os.environ['SBD_BACKEND'] = 'cpu'  # or 'gpu'
    import sbd  # Backend selected here
    
    # WRONG - too late
    import sbd  # Backend already selected
    os.environ['SBD_BACKEND'] = 'cpu'  # Has no effect!

Or from command line:
    export SBD_BACKEND=cpu    # Force CPU
    export SBD_BACKEND=gpu    # Force GPU
    python script.py
    # (default: auto-select, prefers GPU)
"""

import sys
import os

# Determine which backend to load
_requested_backend = os.environ.get('SBD_BACKEND', 'auto').lower()
_backend = None
_backend_name = None

def _load_backend(backend_name):
    """Try to load a specific backend"""
    if backend_name == 'gpu':
        from . import _core_gpu
        return _core_gpu, 'gpu'
    elif backend_name == 'cpu':
        from . import _core_cpu
        return _core_cpu, 'cpu'
    else:
        raise ValueError(f"Invalid backend: {backend_name}")

# Load the requested backend
if _requested_backend == 'auto':
    # Try GPU first, fall back to CPU
    try:
        _backend, _backend_name = _load_backend('gpu')
        if os.environ.get('SBD_VERBOSE', '0') == '1':
            print("SBD: Using GPU backend (auto-selected)")
    except ImportError:
        try:
            _backend, _backend_name = _load_backend('cpu')
            if os.environ.get('SBD_VERBOSE', '0') == '1':
                print("SBD: Using CPU backend (GPU not available)")
        except ImportError as e:
            raise ImportError(
                f"Could not import any SBD backend. Error: {e}"
            )
elif _requested_backend in ['cpu', 'gpu']:
    try:
        _backend, _backend_name = _load_backend(_requested_backend)
        if os.environ.get('SBD_VERBOSE', '0') == '1':
            print(f"SBD: Using {_requested_backend.upper()} backend (requested)")
    except ImportError as e:
        raise ImportError(
            f"Requested backend '{_requested_backend}' not available. "
            f"Error: {e}"
        )
else:
    raise ValueError(
        f"Invalid SBD_BACKEND value: '{_requested_backend}'. "
        "Must be 'cpu', 'gpu', or 'auto'"
    )

# Export all symbols from the backend
_backend_symbols = [name for name in dir(_backend) if not name.startswith('_')]
for name in _backend_symbols:
    globals()[name] = getattr(_backend, name)

# Also make backend module available
__backend__ = _backend
__backend_name__ = _backend_name

# Version info
__version__ = "1.2.0"

def get_backend():
    """
    Get the name of the currently loaded backend
    
    Returns:
        str: Backend name ('cpu' or 'gpu')
    
    Note: The backend is fixed at import time and cannot be changed.
    """
    return _backend_name

def get_backend_module():
    """
    Get the backend module object
    
    Returns:
        module: The loaded backend module (_core_cpu or _core_gpu)
    """
    return _backend

def available_backends():
    """
    Get list of backends that were compiled
    
    Returns:
        list: List of backend names that exist as compiled modules
    
    Note: This checks which .so files exist, not which can be loaded
    (only one can be loaded per process due to pybind11 limitations).
    """
    import os
    import glob
    
    backends = []
    
    # Get the directory where this module is located
    module_dir = os.path.dirname(__file__)
    
    # Check for CPU backend
    cpu_pattern = os.path.join(module_dir, '_core_cpu*.so')
    if glob.glob(cpu_pattern):
        backends.append('cpu')
    
    # Check for GPU backend  
    gpu_pattern = os.path.join(module_dir, '_core_gpu*.so')
    if glob.glob(gpu_pattern):
        backends.append('gpu')
    
    return backends

def print_backend_info():
    """Print information about the current backend"""
    print("="*70)
    print("SBD Python Bindings")
    print("="*70)
    print(f"Version: {__version__}")
    print(f"Active backend: {_backend_name}")
    print(f"Compiled backends: {', '.join(available_backends())}")
    print()
    print("Note: Only one backend can be active per Python process.")
    print("To use a different backend, set SBD_BACKEND environment variable")
    print("before importing sbd:")
    print("  export SBD_BACKEND=cpu  # or gpu")
    print("  python your_script.py")
    print("="*70)

# Print info if requested
if os.environ.get('SBD_PRINT_INFO', '0') == '1':
    print_backend_info()

__all__ = [
    # Backend management
    'get_backend',
    'get_backend_module',
    'available_backends',
    'print_backend_info',
    
    # Version
    '__version__',
    '__backend__',
    '__backend_name__',
]

# Add backend symbols to __all__
__all__.extend(_backend_symbols)

# Made with Bob
