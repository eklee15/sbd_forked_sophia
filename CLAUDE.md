# SBD Python Bindings — Development Notes

## Repository

- **Upstream (RIKEN):** https://github.com/r-ccs-cms/sbd (remote: `upstream`)
- **Fork:** https://github.com/hfwen0502/sbd (remote: `origin`)
- **Active branch:** `sbd-python-cleanup` — refactored Python bindings with runtime CPU/GPU switching

## Build

### Local (macOS)
```bash
source ~/bin/setup_env_sbd.sh   # sets CC=/usr/bin/clang, CXX=/usr/bin/clang++, MPI, BLAS
source ~/.venv/bin/activate
pip install -e . --no-build-isolation
```
**macOS gotcha:** Must use system clang (`/usr/bin/clang++`), not Homebrew clang — libc++ mismatch causes symbol errors.

### Remote server (ssh -J root@150.239.225.32 root@10.241.128.40)
```bash
source /mnt/data/myenv/bin/activate
SBD_BUILD_BACKEND=both CC=nvc CXX=nvc++ CFLAGS='' CXXFLAGS='' \
  PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/2025/compilers/bin:$PATH \
  pip install -e . --no-build-isolation
```

## Architecture

- **Eager backend loading:** Both `_core_cpu.so` and `_core_gpu.so` loaded at `import sbd` into `_backends` dict
- **`py::module_local()`** in `bindings.cpp` prevents pybind11 type registry collision when both backends coexist
- **Lazy init:** `sbd.init()` is optional — auto-called on first API use with `device='auto'`
- **Per-call device switching:** `sbd.tpb_diag(..., device='gpu')` overrides default without re-init
- **MPI decomposition:** Total ranks = `task_comm_size × adet_comm_size × bdet_comm_size`

## Key Design Decisions

### SCIState amplitudes are redundant for SBD
SBD returns carryover determinants directly (`carryover_adet/bdet`). The `SCIState.amplitudes` in `sbd_solver.py` are either read from a wf dump file or faked as uniform values — they exist only to satisfy the `SCIResult` interface from qiskit-addon-sqd. The wf dump can be skipped (`dump_matrix_form_wf=""`) and amplitudes faked; qiskit-addon-sqd places carryover determinants at the front of the next iteration's subspace regardless.

### qiskit-addon-sqd fork required
The SQD integration (`run_sqd_sbd.py`, `sbd_solver.py`) requires the MPI-aware fork:
`pip install git+https://github.com/hfwen0502/qiskit-addon-sqd@patch-ferminon-sbd`

## Known Issues

### Upstream UHF merge breaks GPU (2026-04-03)
RIKEN upstream commits (ba07d7b — UHF integrals, thrust refactoring) cause GPU Davidson to diverge to wrong energy (-84.69 vs -76.24 for H2O). CPU unaffected. Reverted in our branch. Do not merge upstream/main until fixed.

### Fulqrum RHEL 9 crash
Fulqrum's C++ extensions crash on RHEL 9 / Fedora / CentOS Stream 9 due to `_GLIBCXX_ASSERTIONS`. Rebuild with:
```bash
CXXFLAGS='-O3 -std=c++17 -ffast-math -fopenmp -U_GLIBCXX_ASSERTIONS -DNDEBUG' \
pip install -e . --no-build-isolation --force-reinstall --no-deps
```

## Test Data

- **H2O:** `data/h2o/fcidump.txt` (NORB=24, NELEC=10, MS2=0), expected energy ≈ -76.236 Ha
- **N2:** `data/n2/fcidump.txt` (NORB=60, NELEC=14), expected energy ≈ -109.042 Ha
- **Example bitstrings:** `python/examples/count_dict_h2o.json`
