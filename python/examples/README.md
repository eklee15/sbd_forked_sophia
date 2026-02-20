# SBD Python Examples - Simplified API

This directory contains examples demonstrating the **simplified SBD API** where MPI is handled internally - no need to import mpi4py!

## Examples

### 1. h2o_simplified.py - H2O Calculation (Recommended)

Demonstrates the new simplified API for H2O molecule calculations.

**Key Features:**
- No mpi4py import needed
- Simple `sbd.init()` handles MPI internally
- Easy device selection (CPU/GPU/auto)
- Clean, straightforward code

**Quick Start:**

```bash
# CPU backend (8 MPI ranks, 4 OpenMP threads each)
mpirun -np 8 -x OMP_NUM_THREADS=4 python h2o_simplified.py --device cpu

# GPU backend (8 MPI ranks on 8 GPUs)
mpirun -np 8 python h2o_simplified.py --device gpu

# Auto-detect (default - uses GPU if available)
mpirun -np 8 python h2o_simplified.py
```

**Command-Line Arguments:**
- `--device {auto,cpu,gpu}` - Device selection (default: auto)
- `--fcidump FILE` - FCIDUMP file path
- `--adetfile FILE` - Alpha determinants file
- `--max_it N` - Maximum iterations (default: 100)
- `--eps FLOAT` - Convergence tolerance (default: 1e-4)
- `--adet_comm_size N` - Alpha determinant communicator size (default: 2)
- `--bdet_comm_size N` - Beta determinant communicator size (default: 2)
- `--task_comm_size N` - Task communicator size (default: 2)

**MPI Configuration:**
Total ranks must equal: `task_comm_size × adet_comm_size × bdet_comm_size`

Example: 8 ranks = 2 × 2 × 2

### 2. simple_h2o.py - Minimal Example

Ultra-minimal example showing the absolute basics.

```bash
mpirun -np 4 python simple_h2o.py
```

## Simplified API Benefits

### Before (Legacy API):
```python
from mpi4py import MPI
import sbd._core_cpu as sbd_cpu

comm = MPI.COMM_WORLD
config = sbd_cpu.TPB_SBD()
results = sbd_cpu.tpb_diag_from_files(comm, config, ...)
```

### After (Simplified API):
```python
import sbd

sbd.init(device='gpu')
config = sbd.TPB_SBD()
results = sbd.tpb_diag_from_files(config, ...)
sbd.finalize()
```

**Advantages:**
- ✅ No mpi4py import needed
- ✅ No manual communicator management
- ✅ Automatic device selection
- ✅ Cleaner, more intuitive code
- ✅ Proper resource cleanup with `finalize()`

## Resource Cleanup

All examples properly clean up resources using `sbd.finalize()`:

```python
try:
    sbd.init(device='gpu')
    results = sbd.tpb_diag_from_files(...)
finally:
    sbd.finalize()  # Synchronizes GPU and resets state
```

**Important:** Always call `sbd.finalize()` to:
- Synchronize GPU operations (calls `cudaDeviceSynchronize()` on GPU backend)
- Reset internal state for re-initialization
- Ensure proper cleanup similar to `torch.distributed.destroy_process_group()`

**Note:** 
- `finalize()` does NOT call `MPI_Finalize()` - that's handled automatically by mpi4py
- `finalize()` does NOT call `cudaDeviceReset()` to avoid conflicts with CUDA-aware MPI (UCX)
- GPU memory is freed automatically when the process exits

## Available Determinant Files

Located in `../../data/h2o/`:
- `h2o-1em3-alpha.txt` - 10⁻³ threshold (~100 determinants)
- `h2o-1em4-alpha.txt` - 10⁻⁴ threshold (~1,000 determinants)
- `h2o-1em5-alpha.txt` - 10⁻⁵ threshold (~10,000 determinants)
- `h2o-1em6-alpha.txt` - 10⁻⁶ threshold (~100,000 determinants)
- `h2o-1em7-alpha.txt` - 10⁻⁷ threshold (~1,000,000 determinants)
- `h2o-1em8-alpha.txt` - 10⁻⁸ threshold (~10,000,000 determinants)

## Expected Results

Ground state energy for H2O: approximately **-76.236 Hartree**

Convergence depends on:
- Determinant threshold (smaller = more accurate)
- Convergence tolerance (`--eps`)
- Number of iterations (`--max_it`)

## Performance Tips

**CPU Backend:**
- Set `OMP_NUM_THREADS` to cores per MPI rank
- Balance MPI ranks vs threads based on memory
- Example: 8 ranks × 4 threads = 32 cores

**GPU Backend:**
- One MPI rank per GPU is optimal
- Set `OMP_NUM_THREADS=1` (GPU does the work)
- Each rank automatically assigned to GPU: `rank % num_gpus`
- Example: 8 ranks on 8 GPUs = 1 rank per GPU

**MPI Decomposition:**
- Larger problems benefit from more parallelization
- Balance `adet_comm_size`, `bdet_comm_size`, `task_comm_size`
- Start with equal sizes and adjust based on profiling

## Notes

- **Simplified API:** These examples use the new simplified API (no mpi4py needed)
- **TPB Method Only:** Uses Two-Particle Basis (TPB) diagonalization for quantum chemistry
- **GPU Requirements:** NVIDIA HPC SDK and CUDA-capable GPUs
- **Automatic GPU Assignment:** Each MPI rank assigned to GPU automatically
- **CUDA-aware MPI:** Recommended for best GPU performance
- **Memory:** Larger determinant files require more memory per rank
- **Cleanup:** Always call `sbd.finalize()` in finally blocks for proper resource cleanup

## Troubleshooting

**"Total ranks mismatch":**
- Ensure `mpirun -np N` matches `task_comm_size × adet_comm_size × bdet_comm_size`

**"File not found":**
- Check paths to FCIDUMP and determinant files
- Use absolute paths or correct relative paths

**GPU out of memory:**
- Use smaller determinant file
- Increase number of MPI ranks to distribute memory

**Slow convergence:**
- Increase `--max_it`
- Try smaller `--eps` tolerance
- Use better initial guess with saved wavefunction

## See Also

- [Python Bindings README](../../README_PYTHON.md) - Main documentation
- [Cleanup API](../../CLEANUP_API.md) - Resource cleanup details
- [Python Bindings Architecture](../../PYTHON_BINDINGS_ARCHITECTURE.md) - Technical details