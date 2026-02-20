# SBD Simplified API

## Overview

The new simplified API hides MPI complexity from user code, making SBD easier to use while maintaining full functionality.

## Key Features

1. **No mpi4py import needed** - MPI is handled internally
2. **Clean initialization** - `sbd.init(device='gpu', comm_backend='mpi')`
3. **Utility functions** - `sbd.get_rank()`, `sbd.get_world_size()`, etc.
4. **Backward compatible** - Legacy API still works

## Comparison

### Old API (Legacy)
```python
from mpi4py import MPI
import sbd

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

config = sbd.TPB_SBD()
results = sbd.tpb_diag_from_files(
    comm=comm,  # Must pass comm
    sbd_data=config,
    fcidumpfile="fcidump.txt",
    adetfile="alphadets.txt"
)
```

### New API (Simplified)
```python
import sbd

sbd.init(device='gpu', comm_backend='mpi')
rank = sbd.get_rank()

config = sbd.TPB_SBD()
results = sbd.tpb_diag_from_files(
    # No comm parameter!
    fcidumpfile="fcidump.txt",
    adetfile="alphadets.txt",
    sbd_data=config
)

sbd.finalize()
```

## API Reference

### Initialization

#### `sbd.init(device='auto', comm_backend='mpi')`
Initialize SBD with device and communication backend.

**Parameters:**
- `device` (str): Compute device
  - `'auto'` - Auto-detect (GPU if available, else CPU)
  - `'cpu'` - Force CPU
  - `'gpu'` or `'cuda'` - Force GPU
- `comm_backend` (str): Communication backend
  - `'mpi'` - MPI (only option currently)
  - `'nccl'` - Future: NCCL for GPU-only

**Example:**
```python
sbd.init(device='gpu', comm_backend='mpi')
```

#### `sbd.finalize()`
Clean up SBD internal state. Does NOT call MPI_Finalize (managed by mpirun).

#### `sbd.is_initialized()`
Check if SBD has been initialized.

**Returns:** `bool`

### Query Functions

#### `sbd.get_device()`
Get current compute device.

**Returns:** `'cpu'` or `'gpu'`

#### `sbd.get_comm_backend()`
Get current communication backend.

**Returns:** `'mpi'` (future: `'nccl'`)

#### `sbd.get_rank()`
Get MPI rank of current process.

**Returns:** `int` (0 to world_size-1)

#### `sbd.get_world_size()`
Get total number of MPI processes.

**Returns:** `int`

#### `sbd.barrier()`
MPI barrier - synchronize all processes.

### Main Functions

#### `sbd.TPB_SBD()`
Create TPB_SBD configuration object.

**Returns:** `TPB_SBD` object

#### `sbd.tpb_diag_from_files(fcidumpfile, adetfile, sbd_data, loadname="", savename="")`
Perform TPB diagonalization from files.

**Parameters:**
- `fcidumpfile` (str): Path to FCIDUMP file
- `adetfile` (str): Path to alpha determinants file
- `sbd_data` (TPB_SBD): Configuration object
- `loadname` (str): Load initial wavefunction (optional)
- `savename` (str): Save final wavefunction (optional)

**Returns:** `dict` with keys:
- `'energy'`: Ground state energy
- `'density'`: Orbital densities
- `'carryover_adet'`: Carryover alpha determinants
- `'carryover_bdet'`: Carryover beta determinants
- `'one_p_rdm'`: 1-particle RDM (if do_rdm=1)
- `'two_p_rdm'`: 2-particle RDM (if do_rdm=1)

#### `sbd.tpb_diag(fcidump, adet, bdet, sbd_data, loadname="", savename="")`
Perform TPB diagonalization with pre-loaded data structures.

### Utility Functions

#### `sbd.LoadFCIDump(filename)`
Load FCIDUMP file.

#### `sbd.LoadAlphaDets(filename, bit_length, total_bit_length)`
Load alpha determinants from file.

#### `sbd.makestring(config, bit_length, total_bit_length)`
Convert determinant to string representation.

#### `sbd.available_backends()`
Get list of compiled backends.

**Returns:** `list` of `'cpu'` and/or `'gpu'`

#### `sbd.print_info()`
Print SBD information (version, backends, current state).

## Complete Example

```python
#!/usr/bin/env python3
import sbd
import sys

def main():
    # Initialize SBD
    sbd.init(device='gpu', comm_backend='mpi')
    
    # Get rank info
    rank = sbd.get_rank()
    size = sbd.get_world_size()
    
    if rank == 0:
        print(f"Running on {size} MPI ranks")
        print(f"Device: {sbd.get_device()}")
        print(f"Communication: {sbd.get_comm_backend()}")
    
    # Configure calculation
    config = sbd.TPB_SBD()
    config.max_it = 100
    config.eps = 1e-6
    config.method = 0  # Davidson
    config.do_rdm = 0  # Density only
    config.bit_length = 20
    config.adet_comm_size = 2
    config.bdet_comm_size = 2
    config.task_comm_size = 2
    
    # Run calculation
    try:
        results = sbd.tpb_diag_from_files(
            fcidumpfile="../../data/h2o/fcidump.txt",
            adetfile="../../data/h2o/h2o-1em4-alpha.txt",
            sbd_data=config
        )
        
        if rank == 0:
            print(f"Energy: {results['energy']:.10f} Hartree")
            print(f"Converged!")
        
        return 0
    
    except Exception as e:
        if rank == 0:
            print(f"Error: {e}")
        return 1
    
    finally:
        sbd.finalize()

if __name__ == "__main__":
    sys.exit(main())
```

## Running Examples

```bash
# CPU backend
mpirun -np 8 -x OMP_NUM_THREADS=4 python h2o_simplified.py --device cpu

# GPU backend
mpirun -np 8 python h2o_simplified.py --device gpu

# Auto-detect
mpirun -np 8 python h2o_simplified.py
```

## Design Rationale

### Separation of Concerns

- **`device`**: Where computation happens (CPU/GPU)
- **`comm_backend`**: How processes communicate (MPI/NCCL)

This matches the actual architecture:
- SBD uses MPI for communication (always)
- CPU/GPU selection is for local computation only

### Why Not Like PyTorch?

PyTorch's `torch.distributed.init_process_group(backend='nccl')` uses "backend" for communication because PyTorch supports multiple communication backends (MPI, NCCL, Gloo).

SBD currently only uses MPI for communication, so we separate:
- `device` = compute (CPU/GPU)
- `comm_backend` = communication (MPI, future: NCCL)

This avoids confusion and is more explicit.

### Future NCCL Support

When NCCL is added for GPU-GPU communication:

```python
sbd.init(
    device='gpu',           # Compute on GPU
    comm_backend='nccl'     # Communicate via NCCL
)
```

This will require different C++ bindings that don't depend on mpi4py.

## Backward Compatibility

The legacy API still works for existing code:

```python
from mpi4py import MPI
import sbd

# Legacy functions with _legacy suffix
results = sbd.tpb_diag_from_files_legacy(
    comm=MPI.COMM_WORLD,
    sbd_data=config,
    fcidumpfile="fcidump.txt",
    adetfile="alphadets.txt"
)
```

Or set `SBD_BACKEND` environment variable before import (old method).

## Migration Guide

### From Old API to New API

**Before:**
```python
from mpi4py import MPI
import sbd

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

config = sbd.TPB_SBD()
results = sbd.tpb_diag_from_files(comm, config, "fcidump.txt", "alphadets.txt")
```

**After:**
```python
import sbd

sbd.init(device='auto', comm_backend='mpi')
rank = sbd.get_rank()
size = sbd.get_world_size()

config = sbd.TPB_SBD()
results = sbd.tpb_diag_from_files("fcidump.txt", "alphadets.txt", config)
sbd.finalize()
```

**Changes:**
1. Remove `from mpi4py import MPI`
2. Add `sbd.init()` at start
3. Use `sbd.get_rank()` instead of `comm.Get_rank()`
4. Remove `comm` parameter from function calls
5. Add `sbd.finalize()` at end

## Benefits

1. **Simpler code**: No MPI boilerplate
2. **Cleaner API**: PyTorch-like interface
3. **Better errors**: Clear messages if init() not called
4. **Future-proof**: Easy to add NCCL without breaking API
5. **Familiar**: Matches patterns from popular frameworks

## See Also

- `README_PYTHON.md` - Complete Python bindings documentation
- `python/examples/h2o_simplified.py` - Example using simplified API
- `python/examples/h2o_cpu_gpu.py` - Example with all parameters