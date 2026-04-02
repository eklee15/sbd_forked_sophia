# SBD Utilities

This directory contains utility tools for working with the SBD (Selected Basis Diagonalization) solver.

## Sampler-to-SBD Conversion Tools

Tools to convert Qiskit Sampler output (JSON format) into determinant files compatible with the SBD solver.

### Files

- **`convert_sampler_to_sbd.py`** - Python implementation using `qiskit-addon-sqd`
- **`convert_sampler_to_sbd.cpp`** - C++ implementation using `qiskit-addon-sqd-hpc` (30x faster)
- **`count_dict.json`** - Sample input file with 1000 bitstrings (65 KB)
- **`README_SAMPLER_TO_SBD.md`** - Detailed documentation for the conversion tools

### Quick Start

#### Python Version
```bash
# Install dependencies
pip install qiskit-addon-sqd numpy

# Run conversion with sample file (all determinants)
python convert_sampler_to_sbd.py count_dict.json --norb 29 --nelec 5 5

# Run with subsampling to cap computational cost
python convert_sampler_to_sbd.py count_dict.json --norb 29 --nelec 5 5 --max-dets 1000
```

#### C++ Version (Recommended for Large Datasets)
```bash
# Set paths to dependencies (adjust as needed)
export SQD_HPC_PATH=${SQD_HPC_PATH:-../../qiskit-addon-sqd-hpc}
export BOOST_PATH=${BOOST_PATH:-/path/to/boost}

# Compile (requires boost and qiskit-addon-sqd-hpc)
g++ -std=c++17 -O3 \
    -I${SQD_HPC_PATH}/include \
    -I${BOOST_PATH} \
    -o convert_sampler_to_sbd_cpp convert_sampler_to_sbd.cpp

# Run conversion (30x faster than Python)
# Basic usage: all determinants
./convert_sampler_to_sbd_cpp count_dict.json 29 5 5 output_prefix

# With subsampling to cap computational cost
./convert_sampler_to_sbd_cpp count_dict.json 29 5 5 output_prefix 1000 42
#                                                                  ^max  ^seed
```

**Note**: You need to install or point to:
- `qiskit-addon-sqd-hpc` library
- Boost C++ library (tested with Boost 1.85.0)

### Performance

Tested with 1M samples (29 orbitals, 5+5 electrons):
- **Python**: 1m 41s
- **C++**: 3.3s (30x speedup)

Both implementations produce valid SBD input files with ~99.9% Hilbert space coverage.

### Output

Generates two files:
- `determinants_alpha.txt` - Alpha electron determinants (binary strings)
- `determinants_beta.txt` - Beta electron determinants (binary strings)

These files can be directly used with the SBD solver:
```bash
mpirun -np 4 sbd_diag --fcidump fcidump.txt \
    --adetfile determinants_alpha.txt \
    --bdetfile determinants_beta.txt
```

## See Also

- [README_SAMPLER_TO_SBD.md](README_SAMPLER_TO_SBD.md) - Detailed documentation
- [../python/sbd_solver.py](../python/sbd_solver.py) - SBD solver Python interface
- [../python/examples/](../python/examples/) - Example usage