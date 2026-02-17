# Python Bindings Implementation Plan for SBD Library

## Overview
Create Python bindings for the SBD (Selected Basis Diagonalization) library, focusing on the tensor-product basis (TPB) diagonalization functionality from `apps/chemistry_tpb_selected_basis_diagonalization`.

## Project Structure

```
sbd/
├── python/
│   ├── __init__.py
│   ├── bindings.cpp          # Main pybind11 bindings file
│   └── tests/
│       ├── test_basic.py     # Basic functionality tests
│       └── test_h2o.py       # Integration test with H2O data
├── setup.py                   # Python package build configuration
├── pyproject.toml            # Modern Python project metadata
└── README_PYTHON.md          # Python-specific documentation
```

## Implementation Steps

### 1. Project Setup
**Files to create:**
- `setup.py` - Build configuration using setuptools and pybind11
- `pyproject.toml` - Modern Python packaging metadata
- `python/__init__.py` - Python package initialization
- `README_PYTHON.md` - Python usage documentation

**Key dependencies:**
- pybind11 (for C++ bindings)
- mpi4py (for MPI communicator wrapping)
- numpy (for array handling)

### 2. Core Data Structure Bindings

#### 2.1 FCIDump Structure (`python/bindings.cpp`)
Expose the `sbd::FCIDump` structure:
```cpp
py::class_<sbd::FCIDump>(m, "FCIDump")
    .def(py::init<>())
    .def_readwrite("header", &sbd::FCIDump::header)
    .def_readwrite("one_electron_integrals", &sbd::FCIDump::one_electron_integrals)
    .def_readwrite("two_electron_integrals", &sbd::FCIDump::two_electron_integrals);
```

**Python API:**
```python
fcidump = sbd.LoadFCIDump("fcidump.txt")
```

#### 2.2 TPB SBD Configuration (`sbd::tpb::SBD`)
Expose configuration structure with all parameters:
```cpp
py::class_<sbd::tpb::SBD>(m, "TPB_SBD")
    .def(py::init<>())
    .def_readwrite("task_comm_size", &sbd::tpb::SBD::task_comm_size)
    .def_readwrite("adet_comm_size", &sbd::tpb::SBD::adet_comm_size)
    .def_readwrite("bdet_comm_size", &sbd::tpb::SBD::bdet_comm_size)
    .def_readwrite("method", &sbd::tpb::SBD::method)
    .def_readwrite("max_it", &sbd::tpb::SBD::max_it)
    .def_readwrite("max_nb", &sbd::tpb::SBD::max_nb)
    .def_readwrite("eps", &sbd::tpb::SBD::eps)
    .def_readwrite("max_time", &sbd::tpb::SBD::max_time)
    .def_readwrite("do_shuffle", &sbd::tpb::SBD::do_shuffle)
    .def_readwrite("do_rdm", &sbd::tpb::SBD::do_rdm)
    .def_readwrite("ratio", &sbd::tpb::SBD::ratio)
    .def_readwrite("threshold", &sbd::tpb::SBD::threshold)
    .def_readwrite("bit_length", &sbd::tpb::SBD::bit_length);
```

**Python API:**
```python
config = sbd.TPB_SBD()
config.max_it = 100
config.eps = 1e-6
config.do_rdm = 1
```

### 3. Utility Function Bindings

#### 3.1 Bitstring Utilities
```cpp
m.def("makestring", &sbd::makestring,
      "Convert bitstring to string representation",
      py::arg("config"), py::arg("bit_length"), py::arg("total_bit_length"));

m.def("from_string", &sbd::from_string,
      "Convert string to bitstring representation",
      py::arg("s"), py::arg("bit_length"), py::arg("total_bit_length"));
```

**Python API:**
```python
bitstring = sbd.from_string("01001100", bit_length=4, total_bit_length=8)
string_repr = sbd.makestring(bitstring, bit_length=4, total_bit_length=8)
```

#### 3.2 Determinant Loading
```cpp
m.def("LoadAlphaDets", 
      [](const std::string& filename, size_t bit_length, size_t total_bit_length) {
          std::vector<std::vector<size_t>> dets;
          sbd::LoadAlphaDets(filename, dets, bit_length, total_bit_length);
          return dets;
      },
      "Load alpha determinants from file",
      py::arg("filename"), py::arg("bit_length"), py::arg("total_bit_length"));
```

**Python API:**
```python
alpha_dets = sbd.LoadAlphaDets("alphadets.txt", bit_length=20, total_bit_length=36)
```

#### 3.3 FCIDUMP Loading
```cpp
m.def("LoadFCIDump", &sbd::LoadFCIDump,
      "Load FCIDUMP file",
      py::arg("filename"));
```

### 4. Main Diagonalization Function

#### 4.1 MPI Communicator Handling
Use mpi4py to handle MPI communicators:
```cpp
#include <mpi4py/mpi4py.h>

// In module initialization
if (import_mpi4py() < 0) {
    throw std::runtime_error("Failed to import mpi4py");
}
```

#### 4.2 TPB Diag Function Binding
```cpp
m.def("tpb_diag",
    [](py::object py_comm,
       const sbd::tpb::SBD& sbd_data,
       const sbd::FCIDump& fcidump,
       const std::vector<std::vector<size_t>>& adet,
       const std::vector<std::vector<size_t>>& bdet,
       const std::string& loadname,
       const std::string& savename) {
        
        // Convert mpi4py communicator to MPI_Comm
        PyObject* py_comm_ptr = py_comm.ptr();
        MPI_Comm* comm_ptr = PyMPIComm_Get(py_comm_ptr);
        if (!comm_ptr) throw std::runtime_error("Invalid MPI communicator");
        MPI_Comm comm = *comm_ptr;
        
        // Output variables
        double energy;
        std::vector<double> density;
        std::vector<std::vector<size_t>> co_adet;
        std::vector<std::vector<size_t>> co_bdet;
        std::vector<std::vector<double>> one_p_rdm;
        std::vector<std::vector<double>> two_p_rdm;
        
        // Call the actual diag function
        sbd::tpb::diag(comm, sbd_data, fcidump, adet, bdet,
                       loadname, savename, energy, density,
                       co_adet, co_bdet, one_p_rdm, two_p_rdm);
        
        // Return results as a dictionary
        py::dict results;
        results["energy"] = energy;
        results["density"] = density;
        results["carryover_adet"] = co_adet;
        results["carryover_bdet"] = co_bdet;
        results["one_p_rdm"] = one_p_rdm;
        results["two_p_rdm"] = two_p_rdm;
        
        return results;
    },
    "Perform tensor-product basis diagonalization",
    py::arg("comm"),
    py::arg("sbd_data"),
    py::arg("fcidump"),
    py::arg("adet"),
    py::arg("bdet"),
    py::arg("loadname") = "",
    py::arg("savename") = "");
```

**Python API:**
```python
from mpi4py import MPI
import sbd

comm = MPI.COMM_WORLD

# Load data
fcidump = sbd.LoadFCIDump("fcidump.txt")
alpha_dets = sbd.LoadAlphaDets("alphadets.txt", bit_length=20, total_bit_length=36)

# Configure
config = sbd.TPB_SBD()
config.max_it = 100
config.eps = 1e-6
config.do_rdm = 1

# Run diagonalization
results = sbd.tpb_diag(
    comm=comm,
    sbd_data=config,
    fcidump=fcidump,
    adet=alpha_dets,
    bdet=alpha_dets,  # Use same for closed shell
    loadname="",
    savename="wavefunction.dat"
)

print(f"Ground state energy: {results['energy']}")
print(f"Density: {results['density']}")
```

### 5. Build Configuration

#### 5.1 setup.py
```python
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import os

class get_pybind_include(object):
    def __str__(self):
        return pybind11.get_include()

ext_modules = [
    Extension(
        'sbd._core',
        ['python/bindings.cpp'],
        include_dirs=[
            get_pybind_include(),
            'include',
            os.environ.get('MPI_INCLUDE_PATH', '/usr/include/mpi'),
        ],
        libraries=['mpi', 'lapack', 'blas'],
        language='c++',
        extra_compile_args=['-std=c++17', '-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
]

setup(
    name='sbd',
    version='1.2.0',
    author='Tomonori Shirakawa',
    description='Python bindings for Selected Basis Diagonalization library',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.6.0', 'mpi4py>=3.0.0', 'numpy>=1.19.0'],
    python_requires='>=3.7',
    zip_safe=False,
)
```

#### 5.2 pyproject.toml
```toml
[build-system]
requires = ["setuptools>=45", "wheel", "pybind11>=2.6.0", "mpi4py>=3.0.0"]
build-backend = "setuptools.build_meta"

[project]
name = "sbd"
version = "1.2.0"
description = "Python bindings for Selected Basis Diagonalization library"
readme = "README_PYTHON.md"
requires-python = ">=3.7"
dependencies = [
    "pybind11>=2.6.0",
    "mpi4py>=3.0.0",
    "numpy>=1.19.0",
]
```

### 6. Testing Strategy

#### 6.1 Basic Functionality Test (`python/tests/test_basic.py`)
```python
import sbd
import numpy as np

def test_bitstring_conversion():
    """Test bitstring utility functions"""
    bit_length = 4
    total_length = 8
    
    # Test from_string and makestring
    original = "01001100"
    bitstring = sbd.from_string(original, bit_length, total_length)
    result = sbd.makestring(bitstring, bit_length, total_length)
    
    assert result == original, f"Expected {original}, got {result}"

def test_fcidump_loading():
    """Test FCIDUMP loading"""
    # Assumes test data exists
    fcidump = sbd.LoadFCIDump("../data/h2o/fcidump.txt")
    assert fcidump is not None
    assert len(fcidump.header) > 0

def test_config_creation():
    """Test SBD configuration"""
    config = sbd.TPB_SBD()
    config.max_it = 100
    config.eps = 1e-6
    
    assert config.max_it == 100
    assert config.eps == 1e-6
```

#### 6.2 Integration Test (`python/tests/test_h2o.py`)
```python
from mpi4py import MPI
import sbd
import os

def test_h2o_diagonalization():
    """Test full diagonalization workflow with H2O data"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Paths to test data
    data_dir = "../data/h2o"
    fcidump_file = os.path.join(data_dir, "fcidump.txt")
    adet_file = os.path.join(data_dir, "h2o-1em5-alpha.txt")
    
    # Load data
    fcidump = sbd.LoadFCIDump(fcidump_file)
    alpha_dets = sbd.LoadAlphaDets(adet_file, bit_length=20, total_bit_length=26)
    
    # Configure
    config = sbd.TPB_SBD()
    config.max_it = 50
    config.eps = 1e-5
    config.do_rdm = 0
    
    # Run diagonalization
    results = sbd.tpb_diag(
        comm=comm,
        sbd_data=config,
        fcidump=fcidump,
        adet=alpha_dets,
        bdet=alpha_dets,
    )
    
    if rank == 0:
        print(f"H2O Ground State Energy: {results['energy']}")
        assert results['energy'] < 0, "Energy should be negative"
        assert len(results['density']) > 0, "Density should not be empty"

if __name__ == "__main__":
    test_h2o_diagonalization()
```

### 7. Documentation

#### 7.1 README_PYTHON.md
Create comprehensive documentation covering:
- Installation instructions
- Basic usage examples
- API reference
- MPI usage guidelines
- Troubleshooting

#### 7.2 Example Scripts
Create `examples/python/` directory with:
- `simple_h2o.py` - Basic H2O calculation
- `n2_calculation.py` - N2 molecule example
- `custom_determinants.py` - Using custom determinant sets

### 8. Key Technical Considerations

#### 8.1 Memory Management
- Use `py::return_value_policy::take_ownership` for large data structures
- Consider using numpy arrays for better Python integration
- Handle MPI-distributed data carefully

#### 8.2 Error Handling
- Wrap C++ exceptions and convert to Python exceptions
- Provide clear error messages for MPI-related issues
- Validate input parameters before calling C++ functions

#### 8.3 Performance
- Minimize Python/C++ boundary crossings
- Use move semantics where possible
- Consider releasing GIL for long-running operations

#### 8.4 MPI Compatibility
- Ensure mpi4py and C++ MPI implementations are compatible
- Test with different MPI implementations (OpenMPI, MPICH, Intel MPI)
- Handle MPI initialization/finalization properly

### 9. Testing and Validation

#### 9.1 Unit Tests
- Test each binding function independently
- Verify data type conversions
- Check error handling

#### 9.2 Integration Tests
- Compare Python results with C++ executable results
- Test with various problem sizes
- Verify MPI parallelization works correctly

#### 9.3 Performance Tests
- Benchmark Python overhead vs pure C++
- Profile memory usage
- Test scalability with MPI

### 10. Deployment

#### 10.1 Installation Methods
```bash
# Development installation
pip install -e .

# Regular installation
pip install .

# From source with MPI
MPI_INCLUDE_PATH=/path/to/mpi/include pip install .
```

#### 10.2 Distribution
- Create wheel packages for common platforms
- Document system requirements (MPI, BLAS, LAPACK)
- Provide conda recipe as alternative

## Success Criteria

1. ✅ Python bindings compile successfully with pybind11
2. ✅ All core functions (LoadFCIDump, LoadAlphaDets, tpb_diag) are accessible from Python
3. ✅ MPI functionality works correctly with mpi4py
4. ✅ Test suite passes with H2O and N2 data
5. ✅ Python results match C++ executable results
6. ✅ Documentation is complete and clear
7. ✅ Package can be installed via pip

## Timeline Estimate

- **Setup & Infrastructure**: 2-3 hours
- **Core Bindings**: 4-6 hours
- **MPI Integration**: 2-3 hours
- **Testing**: 3-4 hours
- **Documentation**: 2-3 hours
- **Total**: ~15-20 hours

## Next Steps

After reviewing this plan, proceed to implementation in Code mode to:
1. Create the project structure
2. Implement the pybind11 bindings
3. Set up the build system
4. Write tests
5. Create documentation