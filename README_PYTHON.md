# SBD Python Bindings

Python bindings for the SBD (Selected Basis Diagonalization) library, providing access to Tensor Product Basis (TPB) diagonalization functionality.

## Features

- **Two API styles**: File-based (convenient) and data structure-based (flexible)
- **MPI parallelization**: Full support via mpi4py
- **Complete TPB functionality**: Access to all configuration parameters
- **Efficient**: Minimal Python overhead, GIL released during computation
- **Type-safe**: Proper Python type hints and error handling

## Installation

### Prerequisites

- Python 3.7 or later
- C++17 compatible compiler
- MPI implementation (OpenMPI, MPICH, or Intel MPI)
- BLAS/LAPACK libraries
- pybind11 (≥2.6.0)
- mpi4py (≥3.0.0)
- numpy (≥1.19.0)

### Building from Source

```bash
cd /path/to/sbd

# Install dependencies
pip install pybind11 mpi4py numpy

# Build and install
pip install -e .

# Or for development
python setup.py build_ext --inplace
```

### Setting MPI Include Path

If MPI headers are not in the default location:

```bash
export MPI_INCLUDE_PATH=/path/to/mpi/include
pip install -e .
```

## Quick Start

### Example 1: File-Based API (Recommended)

```python
from mpi4py import MPI
import sbd

# Get MPI communicator
comm = MPI.COMM_WORLD

# Configure calculation
config = sbd.TPB_SBD()
config.max_it = 100
config.eps = 1e-6
config.do_rdm = 0  # 0=density only, 1=full RDM

# Run diagonalization
results = sbd.tpb_diag_from_files(
    comm=comm,
    sbd_data=config,
    fcidumpfile="fcidump.txt",
    adetfile="alphadets.txt"
)

# Access results
if comm.Get_rank() == 0:
    print(f"Energy: {results['energy']}")
    print(f"Density: {results['density']}")
```

### Example 2: Data Structure API (More Flexible)

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

if comm.Get_rank() == 0:
    print(f"Energy: {results['energy']}")
```

## API Reference

### Classes

#### `TPB_SBD`

Configuration for TPB diagonalization.

**Attributes:**
- `task_comm_size` (int): Task communicator size (default: 1)
- `adet_comm_size` (int): Alpha determinant communicator size (default: 1)
- `bdet_comm_size` (int): Beta determinant communicator size (default: 1)
- `h_comm_size` (int): Helper communicator size (default: 1)
- `method` (int): Diagonalization method (default: 0)
  - 0: Davidson without storing Hamiltonian
  - 1: Davidson with storing Hamiltonian
  - 2: Lanczos without storing Hamiltonian
  - 3: Lanczos with storing Hamiltonian
- `max_it` (int): Maximum iterations (default: 1)
- `max_nb` (int): Maximum basis vectors (default: 10)
- `eps` (float): Convergence tolerance (default: 1e-4)
- `max_time` (float): Maximum time in seconds (default: 86400)
- `init` (int): Initialization method (default: 0)
- `do_shuffle` (int): Shuffle determinants flag (default: 0)
- `do_rdm` (int): Calculate RDM (0=density only, 1=full RDM) (default: 0)
- `carryover_type` (int): Carryover selection type (default: 0)
- `ratio` (float): Carryover ratio (default: 0.0)
- `threshold` (float): Carryover threshold (default: 0.01)
- `bit_length` (size_t): Bit length for determinants (default: 20)
- `dump_matrix_form_wf` (str): Filename to dump wavefunction (default: "")

#### `FCIDump`

FCIDUMP data structure.

**Attributes:**
- `header` (dict[str, str]): Header information
- `one_electron_integrals`: One-electron integrals
- `two_electron_integrals`: Two-electron integrals

### Functions

#### `LoadFCIDump(filename: str) -> FCIDump`

Load FCIDUMP file.

**Parameters:**
- `filename`: Path to FCIDUMP file

**Returns:**
- FCIDump object

#### `LoadAlphaDets(filename: str, bit_length: int, total_bit_length: int) -> list[list[int]]`

Load alpha determinants from file.

**Parameters:**
- `filename`: Path to determinants file
- `bit_length`: Bit length for each word
- `total_bit_length`: Total number of orbitals

**Returns:**
- List of determinants (each determinant is a list of integers)

#### `makestring(config: list[int], bit_length: int, total_bit_length: int) -> str`

Convert bitstring to string representation.

**Parameters:**
- `config`: Bitstring as list of integers
- `bit_length`: Bit length for each word
- `total_bit_length`: Total number of orbitals

**Returns:**
- String representation

#### `tpb_diag(comm, sbd_data, fcidump, adet, bdet, loadname="", savename="") -> dict`

Perform TPB diagonalization with pre-loaded data structures.

**Parameters:**
- `comm` (MPI.Comm): MPI communicator
- `sbd_data` (TPB_SBD): Configuration object
- `fcidump` (FCIDump): FCIDUMP data
- `adet` (list[list[int]]): Alpha determinants
- `bdet` (list[list[int]]): Beta determinants
- `loadname` (str): Load wavefunction from file (optional)
- `savename` (str): Save wavefunction to file (optional)

**Returns:**
- Dictionary with keys:
  - `energy` (float): Ground state energy
  - `density` (list[float]): Orbital densities
  - `carryover_adet` (list[list[int]]): Important alpha determinants
  - `carryover_bdet` (list[list[int]]): Important beta determinants
  - `one_p_rdm` (list[list[float]]): 1-particle RDM (if do_rdm=1)
  - `two_p_rdm` (list[list[float]]): 2-particle RDM (if do_rdm=1)

#### `tpb_diag_from_files(comm, sbd_data, fcidumpfile, adetfile, loadname="", savename="") -> dict`

Perform TPB diagonalization from files (convenience function).

**Parameters:**
- `comm` (MPI.Comm): MPI communicator
- `sbd_data` (TPB_SBD): Configuration object
- `fcidumpfile` (str): Path to FCIDUMP file
- `adetfile` (str): Path to determinants file
- `loadname` (str): Load wavefunction from file (optional)
- `savename` (str): Save wavefunction to file (optional)

**Returns:**
- Same dictionary as `tpb_diag`

## Running with MPI

```bash
# Single process
python script.py

# Multiple processes
mpirun -np 4 python script.py

# With specific MPI implementation
mpiexec -n 8 python script.py
```

## Examples

See the `python/examples/` directory for complete examples:
- `simple_h2o.py`: Basic H2O calculation

## Testing

```bash
# Run basic tests
cd python/tests
python test_basic.py

# Run integration tests with MPI
mpirun -np 4 python test_h2o.py
```

## Troubleshooting

### Import Error: "No module named 'sbd'"

Make sure the module is built and installed:
```bash
pip install -e .
```

### MPI Import Error

Install mpi4py with the same MPI implementation:
```bash
pip install mpi4py --no-binary mpi4py
```

### Compilation Errors

Check that you have:
- C++17 compatible compiler
- MPI headers installed
- BLAS/LAPACK libraries installed

Set the MPI include path if needed:
```bash
export MPI_INCLUDE_PATH=/usr/include/mpi
```

### Runtime Errors

Make sure all MPI processes can access the input files and have write permissions for output files.

## Performance Tips

1. **Use file-based API** for convenience unless you need to manipulate data
2. **Set appropriate MPI decomposition** via `adet_comm_size` and `bdet_comm_size`
3. **Disable RDM calculation** (`do_rdm=0`) if not needed - significantly faster
4. **Use method=0** (Davidson without storing) for large systems
5. **Adjust convergence tolerance** (`eps`) based on accuracy needs

## Citation

If you use this software in your research, please cite:

```
[Add citation information here]
```

## License

[Add license information here]

## Support

For issues and questions:
- GitHub Issues: [repository URL]
- Email: [contact email]