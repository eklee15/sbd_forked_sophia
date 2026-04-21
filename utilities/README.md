# SBD Utilities

Utility tools for working with the SBD (Selected Basis Diagonalization) solver.

## [`sampler_to_sbd/`](sampler_to_sbd/)

Convert Qiskit Sampler output (JSON count dict) into SBD determinant files. Python and C++ implementations (C++ is 30x faster for large datasets).

## [`wavefunction_to_sbd/`](wavefunction_to_sbd/)

Convert wavefunctions (determinants + coefficients) to SBD's binary restart format (`--loadname`). Enables round-trip workflows with external solvers like PySCF (e.g., SBD → PySCF `enlarge_space` → SBD).

## See Also

- [`../python/sbd_solver.py`](../python/sbd_solver.py) — SBD solver Python interface
- [`../python/examples/`](../python/examples/) — Example scripts (diag, SQD, variance)
- [`../include/sbd/chemistry/tpb/restart.h`](../include/sbd/chemistry/tpb/restart.h) — C++ restart file format reference
