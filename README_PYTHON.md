# SBD Python Bindings

Python bindings for the Selected Basis Diagonalization (SBD) library with dual CPU/GPU backend support.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Backend Architecture](#backend-architecture)
- [Usage](#usage)
- [Examples](#examples)
- [API Reference](#api-reference)

## Overview

SBD (Selected Basis Diagonalization) is a high-performance library for quantum chemistry calculations. The Python bindings provide easy access to SBD's **Two-Particle Basis (TPB)** diagonalization method with support for both CPU and GPU backends.

**Important Note:** The Python bindings currently support only the **TPB (Two-Particle Basis)** method for quantum chemistry calculations. The SBD library also includes other methods (CAOP for general operators, GDB for general determinant basis) which are available through C++ CLI applications in the `/apps` directory but not yet exposed in Python bindings.

**Key Features:**
- **TPB diagonalization** for quantum chemistry Hamiltonians
- Dual backend support (CPU and GPU)
- Automatic backend detection and building
- Runtime backend selection
- MPI parallelization
- OpenMP threading (CPU) / CUDA acceleration (GPU)

## Features

### Dual Backend Support
- **CPU Backend** (`_core_cpu.so`): OpenMP-parallelized, works on any system
- **GPU Backend** (`_core_gpu.so`): CUDA-accelerated, requires NVIDIA GPUs and HPC SDK
- **Automatic Building**: Detects GPU availability and builds both backends when possible
- **Runtime Selection**: Choose backend via environment variable or Python API

### Computational Methods (TPB Focus)

The Python bindings expose the **Two-Particle Basis (TPB)** method, which is specifically designed for quantum chemistry calculations where the Hamiltonian is expressed in a tensor-product basis of alpha and beta determinants.

**Available in Python:**
- Two-Particle Basis (TPB) diagonalization
- Davidson and Lanczos iterative methods
- Reduced density matrix (RDM) calculations
- Carryover determinant selection

**Other SBD Methods (C++ only):**
- CAOP: Creation/Annihilation Operator basis (see `/apps/caop_selected_basis_diagonalization`)
- GDB: General Determinant Basis (see `/apps/chemistry_gdb_selected_basis_diagonalization`)

For non-TPB methods, please use the C++ CLI applications directly.

## Installation

### Prerequisites

**Required:**
- Python 3.7+
- MPI implementation (OpenMPI, MPICH, etc.)
- BLAS library (OpenBLAS, MKL, etc.)
- pybind11
- mpi4py
- numpy

**Optional (for GPU backend):**
- NVIDIA HPC SDK (nvc++ compiler)
- CUDA-capable GPU
- CUDA-aware MPI (recommended)

### Environment Variables

Set these before installation:

```bash
# MPI configuration
export MPI_HOME=/path/to/mpi

# BLAS configuration
export BLAS_LIB_PATH=/path/to/blas/lib
export BLAS_LIBS=openblas  # or mkl_rt, blas,lapack, etc.

# GPU configuration (optional)
export NVHPC_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/2025/compilers
```

### Installation Methods

#### Method 1: Auto-detect (Recommended)

Automatically builds both CPU and GPU backends if GPU support is detected:

```bash
pip install -e .
```

This will:
- Always build CPU backend
- Build GPU backend if NVIDIA HPC SDK and CUDA GPU are detected
- Set RPATH for runtime library discovery (no LD_LIBRARY_PATH needed)

#### Method 2: Force Specific Backend

Build only CPU backend:
```bash
SBD_BUILD_BACKEND=cpu pip install -e .
```

Build only GPU backend:
```bash
SBD_BUILD_BACKEND=gpu pip install -e .
```

Force build both:
```bash
SBD_BUILD_BACKEND=both pip install -e .
```

### Verification

Check which backends were built:

```bash
python -c "import sbd; print(f'Available backends: {sbd.available_backends()}')"
```

Expected output:
- CPU only: `Available backends: ['cpu']`
- Both: `Available backends: ['cpu', 'gpu']`

## Backend Architecture

### Design Philosophy

The dual backend architecture allows:
1. **Single Installation**: One `pip install` gets both backends
2. **No Symbol Collision**: Separate `.so` files (`_core_cpu.so`, `_core_gpu.so`)
3. **Runtime Selection**: Choose backend when running, not when building
4. **Automatic GPU Assignment**: Each MPI rank assigned to different GPU

### Backend Selection

The Python `__init__.py` dynamically loads the appropriate backend:

```python
import os
os.environ['SBD_BACKEND'] = 'gpu'  # Must be set BEFORE import
import sbd
```

Or use command-line:
```bash
SBD_BACKEND=gpu python script.py
```

### GPU Device Assignment

When using GPU backend with MPI:
- Each MPI rank automatically assigned to a GPU
- Assignment: `gpu_id = mpi_rank % num_gpus`
- Example: 8 ranks on 8 GPUs → each rank gets dedicated GPU
- Example: 16 ranks on 8 GPUs → 2 ranks per GPU

This is handled automatically in `bindings.cpp`:
```cpp
int mpi_rank;
MPI_Comm_rank(comm, &mpi_rank);
int myDevice = mpi_rank % numDevices;
cudaSetDevice(myDevice);
```

## Usage

### Basic Example

```python
from mpi4py import MPI
import sbd

# Get MPI communicator
comm = MPI.COMM_WORLD

# Configure calculation
config = sbd.TPB_SBD()
config.max_it = 100
config.eps = 1e-6
config.bit_length = 20

# Run diagonalization
results = sbd.tpb_diag_from_files(
    comm=comm,
    sbd_data=config,
    fcidumpfile="fcidump.txt",
    adetfile="alphadets.txt"
)

# Access results
energy = results['energy']
density = results['density']
```

### Backend Selection

```python
import os

# Set backend before importing sbd
os.environ['SBD_BACKEND'] = 'gpu'  # or 'cpu'

import sbd

# Verify backend
print(f"Using backend: {sbd.get_backend()}")
```

### MPI Configuration

Configure MPI communicator decomposition:

```python
config = sbd.TPB_SBD()
config.adet_comm_size = 2  # Alpha determinant parallelization
config.bdet_comm_size = 2  # Beta determinant parallelization  
config.task_comm_size = 2  # Task parallelization

# Total MPI ranks = 2 × 2 × 2 = 8
```

## Examples

### H2O Calculation

Located in `python/examples/h2o_cpu_gpu.py` - Comprehensive example with full command-line control over all TPB_SBD parameters.

**Basic CPU Backend:**
```bash
mpirun -np 8 -x OMP_NUM_THREADS=4 python h2o_cpu_gpu.py \
    --device cpu \
    --adet_comm_size 2 \
    --bdet_comm_size 2 \
    --task_comm_size 2 \
    --adetfile ../../data/h2o/h2o-1em4-alpha.txt \
    --eps 1e-4
```

**Basic GPU Backend:**
```bash
mpirun -np 8 python h2o_cpu_gpu.py \
    --device gpu \
    --adet_comm_size 2 \
    --bdet_comm_size 2 \
    --task_comm_size 2 \
    --adetfile ../../data/h2o/h2o-1em4-alpha.txt \
    --eps 1e-4
```

**Advanced Usage with All Options:**
```bash
mpirun -np 8 python h2o_cpu_gpu.py \
    --device gpu \
    --method 0 \
    --max_it 100 \
    --max_nb 20 \
    --eps 1e-6 \
    --max_time 3600 \
    --adet_comm_size 2 \
    --bdet_comm_size 2 \
    --task_comm_size 2 \
    --bit_length 20 \
    --do_rdm 1 \
    --carryover_type 1 \
    --ratio 0.1 \
    --threshold 1e-5 \
    --savename h2o_wf.dat \
    --adetfile ../../data/h2o/h2o-1em4-alpha.txt \
    --max_memory_gb_for_determinants 16
```

**Available Command-Line Options:**

All TPB_SBD parameters are configurable via command-line:

- **Device Selection:**
  - `--device {auto,cpu,gpu}` - Backend selection (default: auto)

- **Input Files:**
  - `--fcidump FILE` - FCIDUMP file path
  - `--adetfile FILE` - Alpha determinants file
  - `--bdetfile FILE` - Beta determinants file (optional)
  - `--loadname FILE` - Load initial wavefunction
  - `--savename FILE` - Save final wavefunction

- **MPI Configuration:**
  - `--task_comm_size N` - Task communicator size
  - `--adet_comm_size N` - Alpha determinant communicator size
  - `--bdet_comm_size N` - Beta determinant communicator size
  - `--h_comm_size N` - Helper communicator size

- **Diagonalization Method:**
  - `--method {0,1,2,3}` - 0=Davidson, 1=Davidson+Ham, 2=Lanczos, 3=Lanczos+Ham
  - `--max_it N` - Maximum iterations (default: 100)
  - `--max_nb N` - Maximum basis vectors (default: 10)
  - `--eps FLOAT` - Convergence tolerance (default: 1e-3)
  - `--max_time SECONDS` - Maximum time limit

- **Options:**
  - `--init N` - Initialization method
  - `--do_shuffle {0,1}` - Shuffle determinants
  - `--do_rdm {0,1}` - Calculate RDM (0=density only, 1=full RDM)
  - `--bit_length N` - Bit length for determinants (default: 20)

- **Carryover Selection:**
  - `--carryover_type N` - Carryover determinant selection type
  - `--ratio FLOAT` - Carryover ratio
  - `--threshold FLOAT` - Carryover threshold

- **Output:**
  - `--dump_matrix_form_wf FILE` - Dump wavefunction in matrix form

- **GPU-Specific:**
  - `--use_precalculated_dets {0,1}` - Use precalculated determinants
  - `--max_memory_gb_for_determinants N` - Max GPU memory in GB, integer (-1=auto)

**Expected Result:** Ground state energy ≈ -76.236 Hartree

See `python/examples/README.md` for more details and `python h2o_cpu_gpu.py --help` for full option list.

## API Reference

### Configuration Object

```python
config = sbd.TPB_SBD()
```

**Attributes:**
- `max_it` (int): Maximum iterations (default: 1)
- `eps` (float): Convergence tolerance (default: 1e-4)
- `method` (int): Diagonalization method (0=Davidson, 1=Davidson+Ham, 2=Lanczos, 3=Lanczos+Ham)
- `do_rdm` (int): Calculate RDM (0=density only, 1=full RDM)
- `bit_length` (int): Bit length for determinants (default: 20)
- `adet_comm_size` (int): Alpha determinant communicator size
- `bdet_comm_size` (int): Beta determinant communicator size
- `task_comm_size` (int): Task communicator size

### Main Functions

#### tpb_diag_from_files
```python
results = sbd.tpb_diag_from_files(
    comm,              # MPI communicator
    sbd_data,          # TPB_SBD configuration
    fcidumpfile,       # Path to FCIDUMP file
    adetfile,          # Path to alpha determinants
    loadname="",       # Load initial wavefunction (optional)
    savename=""        # Save final wavefunction (optional)
)
```

**Returns:** Dictionary with keys:
- `energy` (float): Ground state energy
- `density` (list): Orbital densities
- `carryover_adet` (list): Carryover alpha determinants
- `carryover_bdet` (list): Carryover beta determinants
- `one_p_rdm` (list): 1-particle RDM (if do_rdm=1)
- `two_p_rdm` (list): 2-particle RDM (if do_rdm=1)

#### tpb_diag
```python
results = sbd.tpb_diag(
    comm,              # MPI communicator
    sbd_data,          # TPB_SBD configuration
    fcidump,           # FCIDump object
    adet,              # Alpha determinants (list of lists)
    bdet,              # Beta determinants (list of lists)
    loadname="",       # Load initial wavefunction (optional)
    savename=""        # Save final wavefunction (optional)
)
```

### Utility Functions

```python
# Load FCIDUMP file
fcidump = sbd.LoadFCIDump("fcidump.txt")

# Load determinants
dets = sbd.LoadAlphaDets("alphadets.txt", bit_length=20, total_bit_length=26)

# Convert determinant to string
string = sbd.makestring(det, bit_length=20, total_bit_length=26)

# Backend information
backend = sbd.get_backend()           # Returns 'cpu' or 'gpu'
backends = sbd.available_backends()   # Returns list of available backends
sbd.print_backend_info()              # Prints backend information
```

## Troubleshooting

### Import Errors

**Problem:** `ImportError: cannot import name '_core_cpu'`

**Solution:** Backend not built. Check installation:
```bash
ls python/sbd/_core_*.so
pip install -e . -v  # Verbose output
```

### GPU Backend Not Building

**Problem:** Only CPU backend built despite having GPU

**Solutions:**
1. Check NVIDIA HPC SDK: `which nvc++`
2. Set NVHPC_HOME: `export NVHPC_HOME=/path/to/hpc_sdk/compilers`
3. Force GPU build: `SBD_BUILD_BACKEND=gpu pip install -e .`

### MPI Errors

**Problem:** MPI-related errors during runtime

**Solutions:**
1. Ensure MPI_HOME is set correctly
2. Check mpi4py: `python -c "from mpi4py import MPI; print(MPI.Get_version())"`
3. Use correct mpirun: `$MPI_HOME/bin/mpirun`

### Symbol Collision

**Problem:** CPU code runs on GPU or vice versa

**Solution:** This was an issue in older versions. Current version uses separate `.so` files to prevent symbol collision. Ensure you have the latest code.

## Performance Tips

### CPU Backend
- Set `OMP_NUM_THREADS` to number of cores per MPI rank
- Use fewer MPI ranks with more threads for memory-bound problems
- Example: 8 ranks × 4 threads = 32 cores

### GPU Backend
- One MPI rank per GPU is optimal
- Set `OMP_NUM_THREADS=1` (GPU does the work)
- Ensure CUDA-aware MPI for best performance
- Example: 8 ranks on 8 GPUs

### MPI Decomposition
- Balance `adet_comm_size`, `bdet_comm_size`, `task_comm_size`
- Total ranks = product of three sizes
- Larger problems benefit from more parallelization

## Contributing

See main repository for contribution guidelines.

## License

See LICENSE.txt

## Citation

If you use SBD in your research, please cite:
[Citation information to be added]

---

**Repository:** https://github.com/hfwen0502/sbd  
**Branch:** cpu-gpu-backend (dual backend support)  
**Main Branch:** main (CPU only, stable)