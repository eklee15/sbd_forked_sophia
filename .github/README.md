# SBD Python Bindings

Python bindings for the Selected Basis Diagonalization (SBD) library with dual CPU/GPU backend support.

## Overview

SBD (Selected Basis Diagonalization) is a high-performance library for quantum chemistry calculations. The Python bindings provide access to SBD's **Two-Particle Basis (TPB)** diagonalization method with support for both CPU and GPU backends.

**Key Features:**
- **TPB diagonalization** for quantum chemistry Hamiltonians
- Dual backend: CPU (OpenMP) and GPU (CUDA), switchable at runtime
- MPI parallelization
- Integration with [qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd) for SQD workflows

**Note:** The Python bindings expose the TPB method only. Other SBD methods (CAOP, GDB) are available through C++ CLI apps in `/apps`.

## Installation

### Prerequisites

**Required:** Python 3.8+, MPI (OpenMPI/MPICH), BLAS (OpenBLAS/MKL), pybind11, mpi4py, numpy.

**Optional (GPU):** NVIDIA HPC SDK (nvc++), CUDA-capable GPU, CUDA-aware MPI.

### Environment Variables

```bash
export MPI_HOME=/path/to/mpi
export BLAS_LIB_PATH=/path/to/blas/lib
export BLAS_LIBS=openblas  # or mkl_rt

# macOS: use system clang to match Python's libc++
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++

# GPU (optional)
export NVHPC_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/2025/compilers
export CC=nvc
export CXX=nvc++
```

### Build

```bash
# Auto-detect: builds CPU always, GPU if detected
pip install -e . --no-build-isolation

# Force specific backend
SBD_BUILD_BACKEND=cpu pip install -e . --no-build-isolation
SBD_BUILD_BACKEND=gpu pip install -e . --no-build-isolation
SBD_BUILD_BACKEND=both pip install -e . --no-build-isolation
```

### Verify

```bash
python -c "import sbd; print(sbd.available_backends())"
# CPU only: ['cpu']
# Both:     ['cpu', 'gpu']
```

## Usage

### Quick Start

```python
import sbd

# No explicit init() needed — auto-initializes on first use
config = sbd.TPB_SBD()
config.adet_comm_size = 2
config.bdet_comm_size = 2
config.max_it = 100
config.eps = 1e-4

results = sbd.tpb_diag_from_files(
    fcidumpfile='data/h2o/fcidump.txt',
    adetfile='data/h2o/h2o-1em4-alpha.txt',
    sbd_data=config,
)

print(f"Energy: {results['energy']:.10f} Hartree")
sbd.finalize()
```

### Runtime CPU/GPU Switching

Both backends are loaded at import time into separate namespaces. Switch per-call with the `device` parameter — no re-initialization needed:

```python
import sbd

# Override per call — auto-initializes on first use
result_cpu = sbd.tpb_diag(..., device='cpu')
result_gpu = sbd.tpb_diag(..., device='gpu')

# Or set a default device explicitly
sbd.init(device='gpu')   # optional — only if you want a non-auto default
result = sbd.tpb_diag(...)  # uses GPU

# Or get the backend module directly
backend = sbd.get_backend('gpu')
fcidump = backend.LoadFCIDump('fcidump.txt')
```

### Resource Cleanup

```python
results = sbd.tpb_diag_from_files(...)
sbd.finalize()  # optional — syncs GPU and resets state
```

`finalize()` calls `cudaDeviceSynchronize()` on GPU backends and resets Python state. It does **not** call `cudaDeviceReset()` (avoids CUDA-aware MPI conflicts) or `MPI_Finalize()` (handled by mpi4py).

## Integration with qiskit-addon-sqd

SBD can serve as the eigensolver backend for qiskit-addon-sqd's SQD workflow.

**Note:** Requires the `patch-ferminon-sbd` branch of [hfwen0502/qiskit-addon-sqd](https://github.com/hfwen0502/qiskit-addon-sqd) for MPI-aware solver support.

```python
from functools import partial
from mpi4py import MPI
from sbd.sbd_solver import solve_sci_batch
from sbd.device_config import DeviceConfig
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

# No sbd.init() needed — auto-initializes on first solver call
sbd_solver = partial(
    solve_sci_batch,
    mpi_comm=MPI.COMM_WORLD,
    sbd_config={"method": 0, "eps": 1e-8, "max_it": 100},
    device_config=DeviceConfig.gpu(),  # or .cpu()
)

result = diagonalize_fermionic_hamiltonian(
    hcore, eri, bit_array,
    sci_solver=sbd_solver,
    norb=norb, nelec=nelec,
    samples_per_batch=300, num_batches=3, max_iterations=5,
)
```

See `python/examples/run_sqd_sbd.py` for a complete example.

### Comparison with qiskit-addon-dice-solver

| Feature | dice-solver | SBD |
|---------|------------|-----|
| Solver | DICE (subprocess) | SBD (in-process) |
| GPU | No | Yes (CUDA) |
| MPI | Spawns processes | Direct integration |
| I/O | Temp files | In-memory |

## Examples

Located in `python/examples/`:

- **`run_sbd_diag.py`** — Standalone TPB diagonalization (no Qiskit dependency)
- **`run_sqd_sbd.py`** — SQD loop with SBD solver (random or hardware bitstrings)

See [python/examples/README.md](../python/examples/README.md) for usage details.

## API Reference

### Initialization

| Function | Description |
|----------|-------------|
| `sbd.init(device, comm_backend)` | **Optional.** Initialize MPI, set default device (`'cpu'`, `'gpu'`, `'auto'`). Auto-called on first use with defaults. |
| `sbd.finalize()` | Sync GPU, reset state. Does not call `MPI_Finalize` |
| `sbd.is_initialized()` | Check init status |

### Backend Access

| Function | Description |
|----------|-------------|
| `sbd.get_backend(device=None)` | Get backend module (`_core_cpu` or `_core_gpu`). `None` = default |
| `sbd.available_backends()` | List of compiled backends (`['cpu']` or `['cpu', 'gpu']`) |

### Query

| Function | Description |
|----------|-------------|
| `sbd.get_device()` | Default device name |
| `sbd.get_rank()` | MPI rank |
| `sbd.get_world_size()` | MPI world size |
| `sbd.get_comm()` | MPI communicator |
| `sbd.barrier()` | MPI barrier |

### Configuration

```python
config = sbd.TPB_SBD()
```

| Attribute | Default | Description |
|-----------|---------|-------------|
| `method` | 0 | 0=Davidson, 1=Davidson+Ham, 2=Lanczos, 3=Lanczos+Ham |
| `max_it` | 1 | Max iterations |
| `eps` | 1e-4 | Convergence tolerance |
| `max_nb` | 10 | Max basis vectors |
| `do_rdm` | 0 | 0=density only, 1=full RDM |
| `bit_length` | 20 | Bit length for determinants |
| `adet_comm_size` | 1 | Alpha determinant communicator size |
| `bdet_comm_size` | 1 | Beta determinant communicator size |
| `task_comm_size` | 1 | Task communicator size |

Total MPI ranks = `task_comm_size × adet_comm_size × bdet_comm_size`.

### Diagonalization

```python
# From files
results = sbd.tpb_diag_from_files(fcidumpfile, adetfile, sbd_data,
                                   loadname="", savename="", device=None)

# From data structures
results = sbd.tpb_diag(fcidump, adet, bdet, sbd_data,
                        loadname="", savename="", device=None)
```

**Returns:** `dict` with keys `energy`, `density`, `carryover_adet`, `carryover_bdet`, `one_p_rdm`, `two_p_rdm`.

The optional `device` parameter overrides the default set by `init()`.

### Utilities

```python
fcidump = sbd.LoadFCIDump("fcidump.txt", device=None)
dets = sbd.LoadAlphaDets("alphadets.txt", bit_length, total_bit_length, device=None)
string = sbd.makestring(det, bit_length, total_bit_length, device=None)
det = sbd.from_string(s, bit_length, total_bit_length, device=None)
sbd.print_info()
```

## Backend Architecture

- **`_core_cpu.so`** and **`_core_gpu.so`** are separate pybind11 modules with separate C++ namespaces — no symbol collision.
- Both are loaded eagerly at `import sbd` into `sbd._backends`.
- `get_backend(device)` returns the appropriate module; all wrapper functions accept an optional `device` parameter.
- GPU device assignment: `gpu_id = mpi_rank % num_gpus` (set per `tpb_diag()` call in `bindings.cpp`).

## Troubleshooting

**`ImportError` on macOS (symbol not found):** Python's libc++ and Homebrew clang's libc++ may differ. Use system clang: `CC=/usr/bin/clang CXX=/usr/bin/clang++`.

**`ImportError: _core_cpu`:** Backend not built. Rebuild: `pip install -e . --no-build-isolation -v`

**GPU not building:** Check `which nvc++` and set `NVHPC_HOME`.

**MPI errors:** Verify `MPI_HOME`, check `python -c "from mpi4py import MPI; print(MPI.Get_version())"`.

## Performance Tips

**CPU:** `OMP_NUM_THREADS` = cores per MPI rank.
**GPU:** 1 rank per GPU, `OMP_NUM_THREADS=1`, use method 0 (matrix-free Davidson).

---

**Repository:** https://github.com/hfwen0502/sbd
