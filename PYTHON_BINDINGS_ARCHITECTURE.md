# Python Bindings Architecture for SBD Library

## System Architecture

```mermaid
graph TB
    subgraph "Python Layer"
        A[Python User Code] --> B[sbd Python Package]
        B --> C[mpi4py]
    end
    
    subgraph "Binding Layer"
        B --> D[pybind11 Bindings<br/>python/bindings.cpp]
        C --> D
    end
    
    subgraph "C++ Library Layer"
        D --> E[sbd::tpb::diag]
        D --> F[sbd::LoadFCIDump]
        D --> G[sbd::LoadAlphaDets]
        D --> H[Bitstring Utils]
        
        E --> I[sbd::tpb::SBD Config]
        E --> J[sbd::FCIDump]
        E --> K[MPI Communicator]
    end
    
    subgraph "Dependencies"
        E --> L[MPI Library]
        E --> M[BLAS/LAPACK]
        E --> N[OpenMP]
    end
```

## Data Flow Diagram

```mermaid
sequenceDiagram
    participant User as Python User
    participant Pkg as sbd Package
    participant Bind as pybind11 Layer
    participant CPP as C++ SBD Library
    participant MPI as MPI Runtime
    
    User->>Pkg: Load FCIDUMP
    Pkg->>Bind: LoadFCIDump(filename)
    Bind->>CPP: sbd::LoadFCIDump()
    CPP-->>Bind: FCIDump object
    Bind-->>Pkg: Python FCIDump
    Pkg-->>User: fcidump
    
    User->>Pkg: Load determinants
    Pkg->>Bind: LoadAlphaDets(file, params)
    Bind->>CPP: sbd::LoadAlphaDets()
    CPP-->>Bind: vector<vector<size_t>>
    Bind-->>Pkg: Python list of lists
    Pkg-->>User: alpha_dets
    
    User->>Pkg: Configure SBD
    Pkg->>Bind: Create TPB_SBD()
    Bind->>CPP: new sbd::tpb::SBD
    CPP-->>Bind: SBD object
    Bind-->>Pkg: Python SBD config
    Pkg-->>User: config
    
    User->>Pkg: Run diagonalization
    Pkg->>Bind: tpb_diag(comm, config, ...)
    Bind->>MPI: Convert mpi4py comm
    MPI-->>Bind: MPI_Comm
    Bind->>CPP: sbd::tpb::diag()
    
    Note over CPP,MPI: Parallel computation<br/>across MPI ranks
    
    CPP-->>Bind: Results (energy, density, rdm)
    Bind-->>Pkg: Python dict
    Pkg-->>User: results
```

## Module Structure

```
sbd (Python Package)
├── __init__.py
│   ├── Imports core bindings
│   ├── Defines high-level API
│   └── Version info
│
├── _core (C++ Extension Module)
│   ├── FCIDump class
│   ├── TPB_SBD class
│   ├── LoadFCIDump()
│   ├── LoadAlphaDets()
│   ├── makestring()
│   ├── from_string()
│   └── tpb_diag()
│
└── tests/
    ├── test_basic.py
    └── test_h2o.py
```

## Type Mapping

| C++ Type | Python Type | Notes |
|----------|-------------|-------|
| `std::string` | `str` | Direct mapping |
| `double` | `float` | Direct mapping |
| `int` | `int` | Direct mapping |
| `size_t` | `int` | Automatic conversion |
| `std::vector<double>` | `list[float]` | Can convert to numpy array |
| `std::vector<std::vector<size_t>>` | `list[list[int]]` | Nested lists |
| `std::map<std::string, std::string>` | `dict[str, str]` | Direct mapping |
| `MPI_Comm` | `mpi4py.MPI.Comm` | Via mpi4py API |
| `sbd::FCIDump` | `sbd.FCIDump` | Custom class binding |
| `sbd::tpb::SBD` | `sbd.TPB_SBD` | Custom class binding |

## Function Signatures

### Python API
```python
# Load FCIDUMP file
fcidump: FCIDump = sbd.LoadFCIDump(filename: str) -> FCIDump

# Load determinants
alpha_dets: list[list[int]] = sbd.LoadAlphaDets(
    filename: str,
    bit_length: int,
    total_bit_length: int
) -> list[list[int]]

# Bitstring utilities
bitstring: list[int] = sbd.from_string(
    s: str,
    bit_length: int,
    total_bit_length: int
) -> list[int]

string_repr: str = sbd.makestring(
    config: list[int],
    bit_length: int,
    total_bit_length: int
) -> str

# Main diagonalization
results: dict = sbd.tpb_diag(
    comm: MPI.Comm,
    sbd_data: TPB_SBD,
    fcidump: FCIDump,
    adet: list[list[int]],
    bdet: list[list[int]],
    loadname: str = "",
    savename: str = ""
) -> dict[str, Any]
```

### Return Dictionary Structure
```python
{
    'energy': float,                    # Ground state energy
    'density': list[float],             # Orbital densities
    'carryover_adet': list[list[int]], # Important alpha determinants
    'carryover_bdet': list[list[int]], # Important beta determinants
    'one_p_rdm': list[list[float]],   # 1-particle RDM (if do_rdm=1)
    'two_p_rdm': list[list[float]]    # 2-particle RDM (if do_rdm=1)
}
```

## Build Process

```mermaid
graph LR
    A[setup.py] --> B[setuptools]
    B --> C[Compile bindings.cpp]
    C --> D[Link with MPI]
    D --> E[Link with BLAS/LAPACK]
    E --> F[Create _core.so]
    F --> G[Install sbd package]
    
    H[pybind11] --> C
    I[SBD headers] --> C
    J[mpi4py] --> D
```

## MPI Integration Details

### Communicator Conversion
```cpp
// In bindings.cpp
#include <mpi4py/mpi4py.h>

// Initialize mpi4py
if (import_mpi4py() < 0) {
    throw std::runtime_error("Failed to import mpi4py");
}

// Convert Python MPI.Comm to C MPI_Comm
PyObject* py_comm_ptr = py_comm.ptr();
MPI_Comm* comm_ptr = PyMPIComm_Get(py_comm_ptr);
MPI_Comm comm = *comm_ptr;
```

### Python Usage
```python
from mpi4py import MPI

# Get communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Pass to C++ function
results = sbd.tpb_diag(comm=comm, ...)

# Results available on all ranks
if rank == 0:
    print(f"Energy: {results['energy']}")
```

## Error Handling Strategy

```mermaid
graph TD
    A[Python Call] --> B{Input Validation}
    B -->|Invalid| C[Raise ValueError]
    B -->|Valid| D[Call C++ Function]
    D --> E{C++ Exception?}
    E -->|Yes| F[Catch in pybind11]
    F --> G[Convert to Python Exception]
    G --> H[Raise in Python]
    E -->|No| I[Return Results]
    I --> J[Convert to Python Types]
    J --> K[Return to User]
```

## Performance Considerations

### Memory Management
- **Zero-copy where possible**: Use `py::array_t` for numpy arrays
- **Move semantics**: Transfer ownership of large vectors
- **Reference counting**: Let Python manage object lifetimes

### GIL Handling
```cpp
// Release GIL for long computations
py::gil_scoped_release release;
sbd::tpb::diag(...);  // C++ computation
py::gil_scoped_acquire acquire;
// Convert results to Python
```

### Data Locality
- Minimize Python/C++ boundary crossings
- Batch operations when possible
- Use contiguous memory layouts

## Testing Strategy

```mermaid
graph TB
    A[Unit Tests] --> B[Test Bindings]
    A --> C[Test Data Conversion]
    A --> D[Test Error Handling]
    
    E[Integration Tests] --> F[Test H2O System]
    E --> G[Test N2 System]
    E --> H[Test MPI Parallelism]
    
    I[Validation Tests] --> J[Compare with C++ Results]
    I --> K[Verify Energy Values]
    I --> L[Check RDM Properties]
    
    B --> M[CI/CD Pipeline]
    C --> M
    D --> M
    F --> M
    G --> M
    H --> M
    J --> M
    K --> M
    L --> M
```

## Deployment Workflow

```mermaid
graph LR
    A[Source Code] --> B[Build Wheel]
    B --> C[Test Installation]
    C --> D{Tests Pass?}
    D -->|No| E[Fix Issues]
    E --> A
    D -->|Yes| F[Upload to PyPI]
    F --> G[User pip install]
    
    H[Documentation] --> I[Build Docs]
    I --> J[Deploy to GitHub Pages]
```

## Example Usage Patterns

### Pattern 1: Simple Calculation
```python
import sbd
from mpi4py import MPI

comm = MPI.COMM_WORLD
fcidump = sbd.LoadFCIDump("fcidump.txt")
dets = sbd.LoadAlphaDets("dets.txt", 20, 36)

config = sbd.TPB_SBD()
config.max_it = 100

results = sbd.tpb_diag(comm, config, fcidump, dets, dets)
print(f"Energy: {results['energy']}")
```

### Pattern 2: With RDM Calculation
```python
config = sbd.TPB_SBD()
config.do_rdm = 1
config.max_it = 100

results = sbd.tpb_diag(comm, config, fcidump, dets, dets)

# Access RDMs
one_rdm = results['one_p_rdm']
two_rdm = results['two_p_rdm']
```

### Pattern 3: Iterative Refinement
```python
# First pass
results1 = sbd.tpb_diag(comm, config, fcidump, dets1, dets1,
                        savename="wf1.dat")

# Use carryover determinants for second pass
co_dets = results1['carryover_adet']
results2 = sbd.tpb_diag(comm, config, fcidump, co_dets, co_dets,
                        loadname="wf1.dat", savename="wf2.dat")
```

## Future Enhancements

1. **NumPy Integration**: Convert vectors to numpy arrays automatically
2. **Async Support**: Enable async/await for long computations
3. **GPU Support**: Expose CUDA/HIP functionality if compiled with thrust
4. **Streaming Results**: Yield intermediate results during iteration
5. **Checkpoint Management**: High-level API for restart files
6. **Visualization**: Built-in plotting for densities and RDMs