# Python Bindings Quick Start Guide

## Overview
This guide provides a quick reference for implementing Python bindings for the SBD library's TPB diagonalization functionality.

## File Checklist

### Core Files to Create
- [ ] `setup.py` - Build configuration
- [ ] `pyproject.toml` - Modern Python packaging
- [ ] `python/__init__.py` - Package initialization
- [ ] `python/bindings.cpp` - Main pybind11 bindings
- [ ] `python/tests/test_basic.py` - Basic tests
- [ ] `python/tests/test_h2o.py` - Integration test
- [ ] `README_PYTHON.md` - Python documentation
- [ ] `examples/python/simple_h2o.py` - Example script

## Implementation Checklist

### Phase 1: Project Setup (30 min)
- [ ] Create `python/` directory structure
- [ ] Create `setup.py` with basic configuration
- [ ] Create `pyproject.toml`
- [ ] Create empty `python/__init__.py`
- [ ] Test that structure is correct

### Phase 2: Basic Bindings (2 hours)
- [ ] Create `python/bindings.cpp` skeleton
- [ ] Add pybind11 module definition
- [ ] Bind `FCIDump` structure
- [ ] Bind `LoadFCIDump` function
- [ ] Bind `TPB_SBD` configuration struct
- [ ] Test compilation

### Phase 3: Utility Functions (1 hour)
- [ ] Bind `makestring` function
- [ ] Bind `from_string` function
- [ ] Bind `LoadAlphaDets` function
- [ ] Add helper functions for data conversion
- [ ] Test utility functions

### Phase 4: MPI Integration (2 hours)
- [ ] Add mpi4py headers
- [ ] Implement MPI communicator conversion
- [ ] Test MPI functionality
- [ ] Handle edge cases

### Phase 5: Main Diagonalization (2 hours)
- [ ] Bind `sbd::tpb::diag` function
- [ ] Handle input/output parameters
- [ ] Return results as Python dict
- [ ] Add error handling
- [ ] Test with simple case

### Phase 6: Testing (2 hours)
- [ ] Write basic unit tests
- [ ] Write H2O integration test
- [ ] Test MPI parallelization
- [ ] Validate results against C++ version
- [ ] Fix any issues

### Phase 7: Documentation (1 hour)
- [ ] Write `README_PYTHON.md`
- [ ] Create example scripts
- [ ] Add docstrings to bindings
- [ ] Document installation process

## Key Code Snippets

### Minimal bindings.cpp Structure
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <mpi4py/mpi4py.h>
#include "sbd/sbd.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "SBD Python bindings";
    
    // Initialize mpi4py
    if (import_mpi4py() < 0) {
        throw std::runtime_error("Failed to import mpi4py");
    }
    
    // Bind classes and functions here
    
    // FCIDump
    py::class_<sbd::FCIDump>(m, "FCIDump")
        .def(py::init<>());
    
    // LoadFCIDump
    m.def("LoadFCIDump", &sbd::LoadFCIDump);
    
    // TPB_SBD
    py::class_<sbd::tpb::SBD>(m, "TPB_SBD")
        .def(py::init<>())
        .def_readwrite("max_it", &sbd::tpb::SBD::max_it);
    
    // tpb_diag
    m.def("tpb_diag", [](py::object py_comm, ...) {
        // Implementation
    });
}
```

### Minimal setup.py
```python
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'sbd._core',
        ['python/bindings.cpp'],
        include_dirs=[
            pybind11.get_include(),
            'include',
        ],
        libraries=['mpi', 'lapack', 'blas'],
        language='c++',
        extra_compile_args=['-std=c++17'],
    ),
]

setup(
    name='sbd',
    version='1.2.0',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.6.0', 'mpi4py>=3.0.0'],
)
```

### Minimal Test
```python
import sbd
from mpi4py import MPI

def test_basic():
    comm = MPI.COMM_WORLD
    fcidump = sbd.LoadFCIDump("../data/h2o/fcidump.txt")
    assert fcidump is not None
```

## Build and Test Commands

```bash
# Install in development mode
cd /path/to/sbd
pip install -e .

# Run tests
cd python/tests
python test_basic.py
mpirun -np 4 python test_h2o.py

# Build wheel
python setup.py bdist_wheel

# Install from wheel
pip install dist/sbd-1.2.0-*.whl
```

## Common Issues and Solutions

### Issue: MPI not found
**Solution**: Set MPI_INCLUDE_PATH environment variable
```bash
export MPI_INCLUDE_PATH=/usr/include/mpi
pip install -e .
```

### Issue: BLAS/LAPACK not found
**Solution**: Install development packages
```bash
# Ubuntu/Debian
sudo apt-get install libblas-dev liblapack-dev

# macOS
brew install openblas lapack
```

### Issue: pybind11 compilation errors
**Solution**: Ensure C++17 support
```bash
# Check compiler version
g++ --version  # Should be >= 7.0
```

### Issue: mpi4py import fails
**Solution**: Install mpi4py with same MPI
```bash
pip install mpi4py --no-binary mpi4py
```

## Testing Strategy

### Level 1: Compilation Test
```bash
python setup.py build_ext --inplace
python -c "import sbd; print(sbd.__version__)"
```

### Level 2: Function Test
```python
import sbd
fcidump = sbd.LoadFCIDump("test.txt")
print(type(fcidump))
```

### Level 3: MPI Test
```bash
mpirun -np 2 python -c "from mpi4py import MPI; import sbd; print(MPI.COMM_WORLD.Get_rank())"
```

### Level 4: Integration Test
```bash
cd python/tests
mpirun -np 4 python test_h2o.py
```

## Performance Validation

Compare Python vs C++ performance:

```bash
# C++ version
cd apps/chemistry_tpb_selected_basis_diagonalization
mpirun -np 4 ./main --fcidump ../../data/h2o/fcidump.txt \
                     --adetfile ../../data/h2o/h2o-1em5-alpha.txt \
                     --iteration 50

# Python version
cd examples/python
mpirun -np 4 python simple_h2o.py
```

Expected: Python overhead < 5%

## Debugging Tips

### Enable verbose output
```python
import sbd
sbd.set_verbose(True)  # If implemented
```

### Check MPI communicator
```python
from mpi4py import MPI
comm = MPI.COMM_WORLD
print(f"Rank: {comm.Get_rank()}, Size: {comm.Get_size()}")
```

### Verify data loading
```python
import sbd
fcidump = sbd.LoadFCIDump("fcidump.txt")
print(f"NORB: {fcidump.header.get('NORB')}")
print(f"NELEC: {fcidump.header.get('NELEC')}")
```

### Profile performance
```python
import cProfile
import sbd

cProfile.run('sbd.tpb_diag(...)', 'profile.stats')
```

## Next Steps After Implementation

1. **Validate Results**: Compare with C++ version
2. **Optimize Performance**: Profile and optimize hot paths
3. **Extend Coverage**: Add more functions if needed
4. **Write Documentation**: Complete user guide
5. **Create Examples**: Add more example scripts
6. **Package Distribution**: Create wheels for PyPI
7. **CI/CD Setup**: Automate testing and deployment

## Resources

- [pybind11 Documentation](https://pybind11.readthedocs.io/)
- [mpi4py Documentation](https://mpi4py.readthedocs.io/)
- [Python Packaging Guide](https://packaging.python.org/)
- [SBD User Guide](docs/user-guide.md)

## Success Metrics

- ✅ Compiles without errors
- ✅ All tests pass
- ✅ MPI parallelization works
- ✅ Results match C++ version
- ✅ Performance overhead < 5%
- ✅ Can install via pip
- ✅ Documentation complete

## Estimated Timeline

| Phase | Time | Cumulative |
|-------|------|------------|
| Setup | 30 min | 30 min |
| Basic Bindings | 2 hours | 2.5 hours |
| Utilities | 1 hour | 3.5 hours |
| MPI Integration | 2 hours | 5.5 hours |
| Main Function | 2 hours | 7.5 hours |
| Testing | 2 hours | 9.5 hours |
| Documentation | 1 hour | 10.5 hours |
| **Total** | **~11 hours** | |

Add buffer for debugging and refinement: **~15 hours total**