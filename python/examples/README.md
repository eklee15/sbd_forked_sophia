# SBD Python Examples

Examples demonstrating SBD's capabilities for quantum chemistry calculations.

## Overview

**Simplified API:** MPI handled internally - no mpi4py import needed
**Communication:** MPI for distributed computing (HPC standard)
**Backends:** CPU (OpenMP) and GPU (CUDA)

## Examples

### 1. chemistry_simplified.py - Basic Chemistry Calculation

Demonstrates simplified API for quantum chemistry calculations with any molecule.

**Quick Start:**
```bash
# Single rank (no MPI decomposition needed)
python chemistry_simplified.py --device cpu --fcidump /path/to/fcidump.txt --adetfile /path/to/alpha_dets.txt

# CPU backend with 8 MPI ranks (must specify MPI decomposition)
mpirun -np 8 -x OMP_NUM_THREADS=4 python chemistry_simplified.py \
    --device cpu \
    --fcidump /path/to/fcidump.txt \
    --adetfile /path/to/alpha_dets.txt \
    --adet_comm_size 2 \
    --bdet_comm_size 2 \
    --task_comm_size 2

# GPU backend (method 0 only)
mpirun -np 4 python chemistry_simplified.py \
    --device gpu \
    --fcidump /path/to/fcidump.txt \
    --adetfile /path/to/alpha_dets.txt \
    --adet_comm_size 2 \
    --bdet_comm_size 2
```

**Note:** When using more than one MPI rank, `--adet_comm_size` and `--bdet_comm_size` must be specified. Total ranks = `task_comm_size × adet_comm_size × bdet_comm_size`. GPU backend supports method 0 (matrix-free Davidson) only.

**Key Options:**
- `--device {auto,cpu,gpu}` - Device selection
- `--fcidump FILE` - FCIDUMP file path
- `--adetfile FILE` - Alpha determinants file
- `--adet_comm_size N`, `--bdet_comm_size N`, `--task_comm_size N` - MPI decomposition

**MPI Configuration:** Total ranks = `task_comm_size × adet_comm_size × bdet_comm_size`

### 2. sqd_integration_sbd.py - Qiskit-Addon-SQD Integration

Integrates SBD with **qiskit-addon-sqd** for Selected Configuration Interaction using Sample-based Quantum Diagonalization (SQD).

**Features:**
- Uses SBD as backend solver for qiskit-addon-sqd
- Supports random sampling OR pre-computed determinants
- Works with N2, H2O, and other molecules
- Full MPI configuration control
- GPU acceleration support

**Quick Examples:**
```bash
# Single rank
python sqd_integration_sbd.py \
    --fcidump ../../data/n2/fcidump.txt \
    --adetfile ../../data/n2/1em3-alpha.txt \
    --device cpu

# N2 with GPU and 4 MPI ranks (must specify MPI decomposition)
mpirun -np 4 python sqd_integration_sbd.py \
    --fcidump ../../data/n2/fcidump.txt \
    --adetfile ../../data/n2/1em3-alpha.txt \
    --device gpu \
    --adet_comm_size 2 \
    --bdet_comm_size 2

# H2O with 8 MPI ranks
mpirun -np 8 python sqd_integration_sbd.py \
    --fcidump ../../data/h2o/fcidump.txt \
    --adetfile ../../data/h2o/h2o-1em4-alpha.txt \
    --adet_comm_size 2 \
    --bdet_comm_size 2 \
    --task_comm_size 2 \
    --device gpu
```

**Note:** FCIDUMP and determinant files are from `qiskit-addon-sqd` (see `../../data/`) or your own data. When using more than one MPI rank, `--adet_comm_size` and `--bdet_comm_size` must be specified. GPU acceleration (Thrust) only applies to method 0 (matrix-free); methods 1/3 fall back to CPU even with GPU backend.

**Key Options:**
- `--fcidump FILE` - FCIDUMP file path
- `--adetfile FILE` - Pre-computed determinants (optional, uses random sampling if omitted)
- `--device {auto,cpu,gpu}` - Device selection
- `--samples N` - Random samples (default: 10000, ignored if adetfile provided)
- `--adet_comm_size N`, `--bdet_comm_size N`, `--task_comm_size N` - MPI decomposition
- `--help` - See all options

**Requirements:** `pip install qiskit-addon-sqd pyscf`

**Expected Results:**
- N2: -109.10 Hartree (FCI)
- H2O: -76.24 Hartree (FCI)

#### Comparison with qiskit-addon-dice-solver

**SBD Advantages:**

1. **GPU Acceleration**: Native CUDA support for massive speedup on large systems
   - DICE: CPU-only
   - SBD: CPU + GPU with automatic device selection

2. **Direct MPI Integration**: No subprocess spawning or file I/O overhead
   - DICE: Spawns external process, writes temp files to `/path-tmp-dir/dice_cli_files*`
   - SBD: Direct in-memory MPI communication via Python bindings

3. **Flexible Sampling**: Supports both random and pre-computed determinants
   - DICE: Requires external determinant generation
   - SBD: Built-in random sampling + pre-computed file support

4. **Better Performance**: Eliminates I/O bottlenecks
   - DICE: File-based communication between Python and C++
   - SBD: Direct memory access through Python bindings

5. **Cleaner Integration**: Native Python API
   ```python
   # DICE approach (subprocess + files)
   solve_sci_batch(..., mpirun_options=["-np", "1"], temp_dir="/tmp/dice")
   
   # SBD approach (direct MPI)
   solve_sci_batch(..., mpi_comm=MPI.COMM_WORLD, device_config={...})
   ```

**When to Use Each:**
- **Use SBD**: GPU systems, large-scale calculations, production workflows
- **Use DICE**: Quick prototyping, CPU-only systems, existing DICE workflows

## API Features

**Simplified API:**
```python
import sbd

sbd.init(device='gpu', comm_backend='mpi')
config = sbd.TPB_SBD()
results = sbd.tpb_diag_from_files(config, ...)
sbd.finalize()
```

**Benefits:**
- No mpi4py import needed
- Automatic device selection
- MPI backend for distributed computing
- Proper resource cleanup with `finalize()`

## Resource Cleanup

All examples properly clean up resources using `sbd.finalize()`:

```python
try:
    sbd.init(device='gpu', comm_backend='mpi')
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

---

## 3. test_n2_sbd.py - Qiskit-Addon-SQD Integration

Demonstrates integration of SBD solver with the **qiskit-addon-sqd** framework for Selected Configuration Interaction (SCI) calculations using the Sample-based Quantum Diagonalization (SQD) method.

**Key Features:**
- Uses SBD as the backend solver for qiskit-addon-sqd
- Supports both random sampling and pre-computed determinants
- Automatic molecule detection from FCIDUMP files
- Full MPI configuration control
- Flexible SBD and SQD parameter customization

### Quick Start

**Basic Usage (N2 with random sampling):**
```bash
mpirun -np 4 python test_n2_sbd.py --device gpu
```

**Using Pre-computed Determinants:**
```bash
# N2 with 1e-3 threshold determinants
mpirun -np 4 python test_n2_sbd.py \
    --fcidump ../../data/n2/fcidump.txt \
    --adetfile ../../data/n2/1em3-alpha.txt \
    --device gpu

# H2O with pre-computed determinants
mpirun -np 4 python test_n2_sbd.py \
    --fcidump ../../data/h2o/fcidump.txt \
    --adetfile ../../data/h2o/h2o-1em3-alpha.txt \
    --device gpu
```

**With MPI Configuration:**
```bash
mpirun -np 8 python test_n2_sbd.py \
    --fcidump ../../data/n2/fcidump.txt \
    --adetfile ../../data/n2/1em3-alpha.txt \
    --adet-comm-size 2 \
    --bdet-comm-size 2 \
    --task-comm-size 2 \
    --device gpu
```

**Custom Parameters:**
```bash
mpirun -np 4 python test_n2_sbd.py \
    --fcidump ../../data/h2o/fcidump.txt \
    --adetfile ../../data/h2o/h2o-1em4-alpha.txt \
    --samples-per-batch 1000 \
    --num-batches 10 \
    --max-iterations 10 \
    --method 0 \
    --eps 1e-10 \
    --threshold 1e-5 \
    --device gpu
```

### Command-Line Arguments

**Molecule Input:**
- `--fcidump FILE` - Path to FCIDUMP file (default: molecules/n2_fci.txt)
- `--adetfile FILE` - Alpha determinants file (optional, uses random sampling if not provided)
- `--bdetfile FILE` - Beta determinants file (optional, uses adetfile if not specified)

**Device Selection:**
- `--device {auto,cpu,gpu}` - Device to use (default: auto)
- `--max-memory-gb N` - Maximum GPU memory in GB (default: -1=auto)

**SQD Parameters:**
- `--samples N` - Total random samples to generate (default: 10000, ignored if --adetfile provided)
- `--samples-per-batch N` - Samples per batch (default: 300)
- `--num-batches N` - Number of batches (default: 5)
- `--max-iterations N` - Maximum SQD iterations (default: 5)

**SBD Solver Parameters:**
- `--method {0,1,2,3}` - Diagonalization method (default: 0=Davidson)
  - 0: Davidson
  - 1: Davidson + Hamiltonian
  - 2: Lanczos
  - 3: Lanczos + Hamiltonian
- `--eps FLOAT` - Convergence tolerance (default: 1e-8)
- `--max-it N` - Maximum SBD iterations (default: 100)
- `--max-nb N` - Maximum basis vectors (default: 50)
- `--do-rdm {0,1}` - Calculate RDM (default: 1, 0=density only, 1=full RDM)
- `--threshold FLOAT` - Carryover threshold (default: 1e-4)

**MPI Configuration:**
- `--adet-comm-size N` - Alpha determinant communicator size (default: 1)
- `--bdet-comm-size N` - Beta determinant communicator size (default: 1)
- `--task-comm-size N` - Task communicator size (default: 1)

**Note:** Total MPI ranks must equal: `task_comm_size × adet_comm_size × bdet_comm_size`

### Sampling Methods

**1. Random Sampling (Default):**
Generates random bitstrings using qiskit-addon-sqd's uniform sampling:
```bash
python test_n2_sbd.py --samples 10000 --device gpu
```

**2. Pre-computed Determinants:**
Uses determinants from files (e.g., from SHCI calculations):
```bash
python test_n2_sbd.py \
    --adetfile ../../data/n2/1em3-alpha.txt \
    --device gpu
```

**3. Separate Alpha/Beta Determinants:**
```bash
python test_n2_sbd.py \
    --adetfile ../../data/n2/1em3-alpha.txt \
    --bdetfile ../../data/n2/1em3-beta.txt \
    --device gpu
```

### Available Determinant Files

**N2 Molecule** (`../../data/n2/`):
- `1em3-alpha.txt` - 10⁻³ threshold
- `1em4-alpha.txt` - 10⁻⁴ threshold
- `1em5-alpha.txt` - 10⁻⁵ threshold
- `1em6-alpha.txt` - 10⁻⁶ threshold
- `1em7-alpha.txt` - 10⁻⁷ threshold
- `3em4-alpha.txt` - 3×10⁻⁴ threshold
- `3em5-alpha.txt` - 3×10⁻⁵ threshold
- `3em6-alpha.txt` - 3×10⁻⁶ threshold
- `3em7-alpha.txt` - 3×10⁻⁷ threshold

**H2O Molecule** (`../../data/h2o/`):
- `h2o-1em3-alpha.txt` - 10⁻³ threshold
- `h2o-1em4-alpha.txt` - 10⁻⁴ threshold
- `h2o-1em5-alpha.txt` - 10⁻⁵ threshold
- `h2o-1em6-alpha.txt` - 10⁻⁶ threshold
- `h2o-1em7-alpha.txt` - 10⁻⁷ threshold
- `h2o-1em8-alpha.txt` - 10⁻⁸ threshold

### Expected Results

**N2 Molecule:**
- Exact FCI energy: **-109.10288938 Hartree**
- Convergence depends on sampling method and parameters

**H2O Molecule:**
- Approximate FCI energy: **-76.24 Hartree**

### Integration with Qiskit-Addon-SQD

This example demonstrates how to use SBD as a backend solver for the qiskit-addon-sqd framework:

```python
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian
from sbd.sbd_solver import solve_sci_batch

# Configure SBD solver
sbd_solver = partial(
    solve_sci_batch,
    mpi_comm=MPI.COMM_WORLD,
    sbd_config={...},
    device_config=device_config,
)

# Run SQD with SBD backend
result = diagonalize_fermionic_hamiltonian(
    hcore, eri, bit_array,
    sci_solver=sbd_solver,
    ...
)
```

### Performance Comparison

**Random Sampling:**
- Fast initialization
- May require more iterations to converge
- Good for exploratory calculations

**Pre-computed Determinants:**
- Uses high-quality determinants from SHCI/FCIQMC
- Faster convergence
- Better accuracy with fewer samples
- Recommended for production calculations

### Example Workflow

```bash
# 1. Start with random sampling for quick test
mpirun -np 4 python test_n2_sbd.py \
    --fcidump ../../data/n2/fcidump.txt \
    --samples 5000 \
    --device gpu

# 2. Use pre-computed determinants for better accuracy
mpirun -np 8 python test_n2_sbd.py \
    --fcidump ../../data/n2/fcidump.txt \
    --adetfile ../../data/n2/1em4-alpha.txt \
    --adet-comm-size 2 \
    --bdet-comm-size 2 \
    --task-comm-size 2 \
    --device gpu

# 3. Fine-tune parameters for production
mpirun -np 8 python test_n2_sbd.py \
    --fcidump ../../data/n2/fcidump.txt \
    --adetfile ../../data/n2/1em5-alpha.txt \
    --samples-per-batch 1000 \
    --num-batches 20 \
    --max-iterations 20 \
    --eps 1e-10 \
    --threshold 1e-6 \
    --adet-comm-size 2 \
    --bdet-comm-size 2 \
    --task-comm-size 2 \
    --device gpu
```

### Requirements

- qiskit-addon-sqd
- pyscf
- mpi4py
- numpy
- SBD with Python bindings

Install with:
```bash
pip install qiskit-addon-sqd pyscf mpi4py numpy
```