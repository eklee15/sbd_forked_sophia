# SBD Python Examples

Examples demonstrating SBD's capabilities for quantum chemistry calculations.

## Overview

- **Communication:** MPI for distributed computing
- **Backends:** CPU (OpenMP) and GPU (CUDA), switchable at runtime via `device` parameter

## Examples

### 1. run_sbd_diag.py — Standalone SBD Diagonalization

Runs a single TPB diagonalization from an FCIDUMP file and alpha determinant
file. No SQD loop, no Qiskit dependency.

```bash
# H2O with 2 MPI ranks
mpirun -np 2 python run_sbd_diag.py \
    --device cpu \
    --fcidump ../../data/h2o/fcidump.txt \
    --adetfile ../../data/h2o/h2o-1em3-alpha.txt \
    --adet_comm_size 2

# N2 with GPU
mpirun -np 8 python run_sbd_diag.py \
    --device gpu \
    --fcidump ../../data/n2/fcidump.txt \
    --adetfile ../../data/n2/1em3-alpha.txt \
    --adet_comm_size 2 --bdet_comm_size 2 --task_comm_size 2
```

**Key options:** `--device`, `--fcidump`, `--adetfile`, `--adet_comm_size`,
`--bdet_comm_size`, `--task_comm_size`, `--method`, `--eps`, `--max_it`.
Run `python run_sbd_diag.py --help` for the full list.

**Requirements:** `sbd`, `mpi4py`

### 2. run_sqd_sbd.py — SQD Loop with SBD Solver

Runs the self-consistent SQD workflow (qiskit-addon-sqd) using SBD as the
eigensolver backend. Supports two bitstring input modes:

- `--counts FILE` — load hardware bitstrings from a count_dict.json
- `--samples N` — generate N uniform random bitstrings (default)

```bash
# H2O with random samples (default)
mpirun -np 4 python run_sqd_sbd.py \
    --fcidump ../../data/h2o/fcidump.txt \
    --device cpu \
    --adet_comm_size 2 --bdet_comm_size 2

# Custom system with hardware bitstrings
mpirun -np 8 python run_sqd_sbd.py \
    --fcidump /path/to/fci_dump.txt \
    --counts /path/to/count_dict.json \
    --samples_per_batch 800 --num_batches 3 --max_iterations 10 \
    --device gpu \
    --adet_comm_size 2 --bdet_comm_size 2 --task_comm_size 2
```

**Key options:** `--fcidump` (required), `--counts`, `--samples`,
`--samples_per_batch`, `--num_batches`, `--max_iterations`, `--device`,
MPI decomposition flags, SBD solver flags.
Run `python run_sqd_sbd.py --help` for the full list.

**Requirements:** `sbd`, `mpi4py`, `qiskit-addon-sqd`, `pyscf`, `numpy`

## MPI Decomposition

Total MPI ranks must equal `task_comm_size × adet_comm_size × bdet_comm_size`.

When using more than one rank, specify at least `--adet_comm_size`. Examples:

| Ranks | Decomposition |
|-------|---------------|
| 1 | default (all = 1) |
| 2 | `--adet_comm_size 2` |
| 4 | `--adet_comm_size 2 --bdet_comm_size 2` |
| 8 | `--adet_comm_size 2 --bdet_comm_size 2 --task_comm_size 2` |

## Backend Selection

Both backends are loaded at import time. Select per-call via `--device`:

```bash
--device cpu    # OpenMP (default)
--device gpu    # CUDA (requires NVIDIA GPU + HPC SDK build)
--device auto   # GPU if available, else CPU
```

Within Python, backends can also be switched at runtime:

```python
import sbd
sbd.init(device='cpu')

# Override default for a single call
result_gpu = sbd.tpb_diag(..., device='gpu')
result_cpu = sbd.tpb_diag(..., device='cpu')
```

## Available Test Data

**H2O** (`../../data/h2o/`): `h2o-1em3` through `h2o-1em8` alpha determinant files.
**N2** (`../../data/n2/`): `1em3` through `1em7` and `3em4` through `3em7` alpha determinant files.

Smaller thresholds = more determinants = higher accuracy.

## Expected Results

- **H2O**: ground state energy ≈ **-76.236 Hartree**
- **N2**: ground state energy ≈ **-109.042 Hartree** (with 1e-3 dets)

## Performance Tips

**CPU:** Set `OMP_NUM_THREADS` to cores per MPI rank (e.g., 8 ranks × 4 threads = 32 cores).

**GPU:** One MPI rank per GPU, `OMP_NUM_THREADS=1`. Each rank auto-assigned: `gpu_id = rank % num_gpus`. Use method 0 (matrix-free Davidson) for best GPU performance.

## See Also

- [Python Bindings README](../../README_PYTHON.md) — Installation, API reference
- [SBD README](../../README.md) — C++ library overview
