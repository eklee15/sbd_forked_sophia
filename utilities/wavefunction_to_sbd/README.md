# Converting Wavefunctions to SBD Restart Format

This directory contains tools to convert wavefunctions (determinants + coefficients) into SBD's binary restart format, enabling round-trip workflows with external solvers like PySCF.

## Motivation

SBD can export wavefunctions via `--dump_matrix_form_wf` (coefficient matrix) and `--savename` (binary restart files), but it cannot import wavefunctions that have been modified externally. This utility bridges that gap for workflows like:

1. **SBD → PySCF → SBD**: Run SBD diag, export wavefunction, use PySCF's `enlarge_space()` to grow the determinant space via Hamiltonian-connected excitations, convert back to SBD format, then reload for refinement.

2. **PySCF → SBD**: Convert any PySCF CI wavefunction (FCI, Selected CI) to SBD format for distributed computation.

3. **Custom initial guess**: Construct a wavefunction in Python and load it as SBD's starting point.

## Files

- **`wavefunction_to_sbd.py`** — Conversion utility (library + CLI)
- **`test_roundtrip.py`** — Round-trip validation: SBD save → Python convert → SBD load

## Quick Start

### As a CLI tool

```bash
# From text determinants + binary coefficient matrix
python wavefunction_to_sbd.py \
    --adetfile alpha_dets.txt \
    --coefficients wavefunction.bin \
    --norb 24 \
    --output_prefix my_wf_ \
    --adet_comm_size 1 \
    --bdet_comm_size 1

# From .npy files (e.g., saved from PySCF)
python wavefunction_to_sbd.py \
    --adetfile alpha_dets.npy \
    --bdetfile beta_dets.npy \
    --coefficients coefficients.npy \
    --norb 24 \
    --output_prefix my_wf_

# Then load in SBD:
python run_sbd_diag.py --loadname my_wf_ --adetfile alpha_dets.txt ...
```

### As a Python library

```python
from wavefunction_to_sbd import integer_to_sbd_words, write_restart_files

# Convert PySCF integer determinants to SBD word-packed format
alpha_dets = [integer_to_sbd_words(int(v), norb=24, bit_length=20) for v in alpha_ints]
beta_dets  = [integer_to_sbd_words(int(v), norb=24, bit_length=20) for v in beta_ints]

# Write SBD restart files
write_restart_files(
    output_prefix="my_wf_",
    alpha_dets=alpha_dets,
    beta_dets=beta_dets,
    coefficients=coef_matrix.ravel(),
    adet_comm_size=1,
    bdet_comm_size=1,
)
```

## Round-Trip Test

`test_roundtrip.py` validates the conversion by checking byte-identical output:

```bash
python test_roundtrip.py
```

The test:
1. Runs SBD diag on H2O (275 dets) and saves the wavefunction
2. Reads the matrix-form dump and determinants into Python
3. Converts back to SBD restart format via `wavefunction_to_sbd.py`
4. Compares files byte-for-byte (must be identical)
5. SBD loads the Python-produced restart file and verifies same energy

## PySCF `enlarge_space` Workflow

The primary use case is growing the determinant space via PySCF, then feeding it back to SBD. The CLI already accepts `.npy` files for both determinants (integer arrays) and coefficients, which is what PySCF produces:

```python
import numpy as np
from pyscf.fci import select_ci

# --- After SBD diag, read wavefunction into PySCF ---
# alpha_ints: sorted integer determinants (bit i = orbital i)
# coef_matrix: shape (n_alpha, n_beta)
civec = select_ci._as_SCIvector(coef_matrix, (alpha_ints, beta_ints))

# --- Enlarge the determinant space ---
from pyscf import tools
eri = tools.fcidump.read("fcidump.txt")["H2"]
myci = select_ci.SCI()
myci.ci_coeff_cutoff = 1e-4
myci.select_cutoff = 1e-4
civec_enlarged = select_ci.enlarge_space(myci, civec, eri, norb, (nelec_a, nelec_b))

# --- Save in formats wavefunction_to_sbd.py accepts ---
np.save("enlarged_alpha.npy", civec_enlarged._strs[0])
np.save("enlarged_beta.npy", civec_enlarged._strs[1])
np.save("enlarged_coefficients.npy", np.asarray(civec_enlarged))
```

```bash
# Convert to SBD restart format
python wavefunction_to_sbd.py \
    --adetfile enlarged_alpha.npy \
    --bdetfile enlarged_beta.npy \
    --coefficients enlarged_coefficients.npy \
    --norb 24 \
    --output_prefix enlarged_wf_

# Also write text determinants for SBD's --adetfile
# (wavefunction_to_sbd.py writes binary restart only;
#  use integers_to_bitstrings() or a simple script for text files)
```

```bash
# Load into SBD
mpirun -np 1 python run_sbd_diag.py \
    --fcidump fcidump.txt \
    --adetfile enlarged_alpha.txt \
    --loadname enlarged_wf_ \
    --bit_length 20
```

## SBD Restart File Format

Reference: [`../../include/sbd/chemistry/tpb/restart.h`](../../include/sbd/chemistry/tpb/restart.h)

Each MPI rank in the basis communicator (`adet_comm_size * bdet_comm_size` ranks) gets one binary file:

| Section | Type | Count | Description |
|---------|------|-------|-------------|
| Header | uint64 | 3 | `adet_range`, `bdet_range`, `det_length` |
| Alpha dets | uint64 | `adet_range * det_length` | Word-packed determinants |
| Beta dets | uint64 | `bdet_range * det_length` | Word-packed determinants |
| Coefficients | float64 | `adet_range * bdet_range` | Row-major (alpha varies slowest) |

### Determinant Packing

SBD stores each determinant as a vector of `size_t` words (default `bit_length=20`):
- Orbital `i` is stored in `word[i // bit_length]`, bit position `i % bit_length`
- `det_length = ceil(norb / bit_length)`
- Example: `norb=24, bit_length=20` → `det_length=2` words per determinant

### MPI Partitioning

Determinants are distributed across ranks using `get_mpi_range()` (see [`../../include/sbd/framework/mpi_utility.h`](../../include/sbd/framework/mpi_utility.h)):
- Alpha dets partitioned across `adet_comm_size` ranks
- Beta dets partitioned across `bdet_comm_size` ranks
- Total files = `adet_comm_size * bdet_comm_size`

## Important Notes

- **`bit_length` must match**: The `bit_length` parameter must be the same value used by SBD (default 20). A mismatch causes corrupted determinants and NaN energies. This was the likely cause of a team member's NaN issue with a C++ converter that computed `bit_length` differently.
- **Determinant ordering**: SBD sorts determinants using `sort_bitarray()` (lexicographic on word vectors from the last word). For standard `bit_length` values, this matches PySCF's integer sort order.
- **Symmetric alpha/beta**: When `--bdetfile` is not given, SBD uses the alpha determinants for both alpha and beta.

## Requirements

- Python 3.10+
- NumPy
- SBD Python bindings (for `test_roundtrip.py`)
- PySCF (only for the `enlarge_space` workflow)
