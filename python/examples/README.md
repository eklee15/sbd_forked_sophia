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
`--bdet_comm_size`, `--task_comm_size`, `--method`, `--tolerance`, `--iteration`.
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

# H2O with example hardware bitstrings (FCIDUMP from ../../data/h2o/)
mpirun -np 4 python run_sqd_sbd.py \
    --fcidump ../../data/h2o/fcidump.txt \
    --counts count_dict_h2o.json \
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

**count_dict.json format:** A JSON object mapping bitstrings to shot counts, as
produced by a quantum device or simulator. Each bitstring has length `2 × NORB` —
the first `NORB` bits are alpha (spin-up) orbitals and the last `NORB` are beta
(spin-down):

```json
{
  "010000000010001010000001010000000001000010100100": 16,
  "000010001110000000000010001001000110000000000100": 12,
  "000101000000010011000000000010000000001001000110": 8
}
```

An example is provided at `count_dict_h2o.json` (matches `../../data/h2o/fcidump.txt`,
NORB=24, 5α+5β electrons).

**Key options:** `--fcidump` (required), `--counts`, `--samples`,
`--samples_per_batch`, `--num_batches`, `--max_iterations`, `--device`,
MPI decomposition flags. SBD solver flags (`--method`, `--tolerance`,
`--iteration`, etc.) have sensible defaults; run `python run_sqd_sbd.py --help`
for the full list.

**Requirements:** `sbd`, `mpi4py`, `pyscf`, `numpy`, and the MPI-aware fork of qiskit-addon-sqd:
[hfwen0502/qiskit-addon-sqd](https://github.com/hfwen0502/qiskit-addon-sqd) (`patch-ferminon-sbd` branch)

#### SQD Parameter Guide

SQD samples bitstrings from a quantum device, uses **configuration recovery** to
correct noisy samples using an orbital occupancy vector, then subsamples into
batches for diagonalization. Occupancies are averaged across batches and fed back
to configuration recovery — this self-consistent loop typically converges in 3–5
iterations. On the first iteration, no occupancies are available yet, so the raw
samples are simply filtered by correct electron count (Hamming weight
postselection).

| Parameter | What it controls | Typical values |
|-----------|-----------------|----------------|
| `--counts FILE` | Load hardware bitstrings from a JSON file (use one or the other) | 10K–1M+ shots |
| `--samples N` | Generate N uniform random bitstrings for testing (default) | 10K–1M+ |
| `--samples_per_batch` | Subspace dimension per batch (accuracy vs. cost) | 300–800 (small), 1M+ (production) |
| `--num_batches` | Independent subsamples for averaging occupancies | 3–10 (small), up to 100 (large) |
| `--max_iterations` | SQD self-consistent loop iterations (not SBD `--iteration`) | 3–5 |

**MPI work distribution:** All ranks diagonalize each batch together, then move
to the next batch sequentially. Within each diagonalization, ranks form a 3D grid:
`adet_comm_size × bdet_comm_size × task_comm_size = total ranks`. More batches
increases wall time linearly but does not require more ranks.

### 3. run_sqd_fulqrum.py — SQD Loop with Fulqrum Eigensolver

Alternative to `run_sqd_sbd.py` using [Fulqrum](https://github.com/qiskit-community/fulqrum)
as the eigensolver instead of SBD. Single-process (no MPI), useful for comparison
and smaller systems.

```bash
# H2O with example bitstrings
python run_sqd_fulqrum.py \
    --fcidump ../../data/h2o/fcidump.txt \
    --counts count_dict_h2o.json

# N2 with hardware bitstrings
python run_sqd_fulqrum.py \
    --fcidump /path/to/fcidump.txt \
    --counts /path/to/count_dict.json
```

**Requirements:** `fulqrum`, `numpy`, `scipy`

**RHEL 9 / Fedora note:** Fulqrum's C++ extensions crash on distros with
`_GLIBCXX_ASSERTIONS` enabled by default. Rebuild with:

```bash
CXXFLAGS='-O3 -std=c++17 -ffast-math -fopenmp -U_GLIBCXX_ASSERTIONS -DNDEBUG' \
pip install -e . --no-build-isolation --force-reinstall --no-deps
```

macOS and Ubuntu/Debian are not affected.

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

Within Python, backends can also be switched at runtime without re-initialization:

```python
import sbd

# No init() needed — auto-initializes on first call
result_cpu = sbd.tpb_diag(..., device='cpu')
result_gpu = sbd.tpb_diag(..., device='gpu')
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

- [Python Bindings README](../../.github/README.md) — Installation, API reference
- [SBD README](../../README.md) — C++ library overview
