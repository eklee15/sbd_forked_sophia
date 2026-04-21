# Converting Qiskit Sampler Output to SBD Determinant Files

This directory contains tools to convert Qiskit Sampler JSON output (`count_dict.json`) into determinant files compatible with the SBD (Selected Basis Diagonalization) solver.

## Overview

The Qiskit Sampler produces a JSON file with bitstrings and their counts. These need to be converted into alpha and beta determinant files that the SBD solver can read. This conversion:

1. Splits each bitstring into alpha (right half) and beta (left half) parts
2. Converts them to the binary string format expected by SBD
3. Optionally extracts unique determinants sorted by marginal probability

## Files

- **`convert_sampler_to_sbd.py`** - Python version using qiskit-addon-sqd utilities
- **`convert_sampler_to_sbd.c`** - C version for standalone use
- **`sbd_output/`** - Example output from Python version
- **`sbd_output_c/`** - Example output from C version

## Python Version (Recommended)

### Features
- Uses `qiskit-addon-sqd` utility functions for proper conversion
- Generates both full and unique determinant lists
- Sorts unique determinants by marginal probability
- Provides detailed statistics

### Requirements
```bash
pip install qiskit-addon-sqd numpy
```

### Usage
```bash
# Activate your virtual environment
source ~/.venv/bin/activate

# Basic usage
python convert_sampler_to_sbd.py ../count_dict.json --norb 29

# With custom output directory
python convert_sampler_to_sbd.py ../count_dict.json --norb 29 --output-dir ./sbd_input

# Specify output prefix
python convert_sampler_to_sbd.py ../count_dict.json --norb 29 --prefix my_determinants
```

### Output Files
- `determinants_alpha.txt` - All alpha determinants (binary strings)
- `determinants_beta.txt` - All beta determinants (binary strings)
- `determinants_alpha_unique.txt` - Unique alpha determinants sorted by probability
- `determinants_beta_unique.txt` - Unique beta determinants sorted by probability
- `determinants_counts.txt` - Counts for each bitstring

## C Version

### Features
- Standalone C implementation with no dependencies
- Fast and lightweight
- Produces identical output to Python version

### Compilation
```bash
gcc -O3 -o convert_sampler_to_sbd convert_sampler_to_sbd.c
```

### Usage
```bash
# Basic usage
./convert_sampler_to_sbd ../count_dict.json 29 sbd_output_c/determinants

# The third argument is the output prefix (optional, default: "determinants")
```

### Output Files
- `determinants_alpha.txt` - Alpha determinants (binary strings)
- `determinants_beta.txt` - Beta determinants (binary strings)
- `determinants_counts.txt` - Counts for each bitstring

## Output Format

Both versions produce determinant files in the format expected by SBD:
- One binary string per line
- Each string has exactly `norb` bits (0s and 1s)
- No spaces or other characters

Example (for norb=29):
```
00011011000001100010001100101
10110110000010000111010010101
00000000001011110000110100010
...
```

## Using with SBD Solver

The generated files can be used directly with the SBD solver:

```bash
mpirun -np 4 sbd_diag \
  --fcidump fcidump.txt \
  --adetfile determinants_alpha.txt \
  --bdetfile determinants_beta.txt \
  --method 0 \
  --tolerance 1e-8
```

Or use the unique determinants for a smaller subspace:

```bash
mpirun -np 4 sbd_diag \
  --fcidump fcidump.txt \
  --adetfile determinants_alpha_unique.txt \
  --bdetfile determinants_beta_unique.txt
```

## Integration with qiskit-addon-sqd

The Python version uses the same utility functions as `qiskit-addon-sqd`:
- `counts_to_arrays()` - Converts counts dict to bitstring matrix
- `bitstring_matrix_to_integers()` - Converts bitstrings to integers

This ensures compatibility with the full SQD workflow. You can also use these determinants directly in the SQD workflow by loading them with the `load_determinants_from_file()` function (see `../sbd/python/examples/sqd_integration_sbd.py`).

## Bitstring Convention

Following the qiskit-addon-sqd convention:
- **Full bitstring**: `[beta_bits | alpha_bits]`
- **Beta bits**: Left half (indices 0 to norb-1)
- **Alpha bits**: Right half (indices norb to 2*norb-1)

For a 58-bit string with norb=29:
- Bits 0-28: Beta (spin-down) electrons
- Bits 29-57: Alpha (spin-up) electrons

## Example Output

From the test with `count_dict.json` (norb=29):

```
Loading ../count_dict.json...
Loaded 100 unique bitstrings
Bitstring length: 58 bits
Expected: 58 bits (2 * 29 orbitals)

Converted to determinants:
  Alpha determinants: 100
  Beta determinants: 100
  Sample alpha det: 56673381 (binary: 00011011000001100010001100101)
  Sample beta det: 268838870 (binary: 10000000001100010011111010110)

Unique determinants:
  Unique alpha: 100
  Unique beta: 99

Saved full determinant lists (SBD format):
  Alpha: sbd_output/determinants_alpha.txt (100 determinants)
  Beta:  sbd_output/determinants_beta.txt (100 determinants)

Saved unique determinants (sorted by marginal probability):
  Alpha: sbd_output/determinants_alpha_unique.txt (100 determinants)
  Beta:  sbd_output/determinants_beta_unique.txt (99 determinants)
```

## Verification

Both Python and C versions produce identical output:
```bash
diff sbd_output/determinants_alpha.txt sbd_output_c/determinants_alpha.txt
# No differences - files are identical!
```

## References

- **qiskit-addon-sqd**: https://github.com/Qiskit/qiskit-addon-sqd
- **SBD solver**: ../sbd/
- **Integration examples**: ../sbd/python/examples/sqd_integration_sbd.py
- **SBD solver wrapper**: ../qiskit-addon-dice-solver/qiskit_addon_dice_solver/sbd_solver.py