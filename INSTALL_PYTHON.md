# Installation Guide for SBD Python Bindings

This guide provides detailed instructions for building and installing the SBD Python bindings.

## Prerequisites

### Required Software

1. **Python 3.7 or later**
   ```bash
   python --version  # Should be >= 3.7
   ```

2. **C++17 Compatible Compiler**
   - GCC 7.0 or later
   - Clang 5.0 or later
   - MSVC 2017 or later

3. **MPI Implementation**
   - OpenMPI 3.0 or later (recommended)
   - MPICH 3.2 or later
   - Intel MPI 2018 or later

4. **BLAS/LAPACK Libraries**
   - OpenBLAS (recommended)
   - Intel MKL
   - ATLAS
   - System BLAS/LAPACK

### Python Dependencies

```bash
pip install pybind11>=2.6.0 mpi4py>=3.0.0 numpy>=1.19.0
```

## Installation Methods

### Method 1: Development Installation (Recommended for Development)

This method creates a symbolic link, so changes to the source are immediately reflected.

```bash
cd /path/to/sbd
pip install -e .
```

### Method 2: Standard Installation

```bash
cd /path/to/sbd
pip install .
```

### Method 3: Build in Place (For Testing)

```bash
cd /path/to/sbd
python setup.py build_ext --inplace
```

This creates the `_core.so` (or `_core.pyd` on Windows) file in the `python/` directory.

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

1. **Install system dependencies:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y \
       build-essential \
       python3-dev \
       libopenmpi-dev \
       openmpi-bin \
       libopenblas-dev \
       liblapack-dev
   ```

2. **Install Python dependencies:**
   ```bash
   pip install pybind11 mpi4py numpy
   ```

3. **Build and install:**
   ```bash
   cd /path/to/sbd
   pip install -e .
   ```

### Linux (CentOS/RHEL/Fedora)

1. **Install system dependencies:**
   ```bash
   sudo yum install -y \
       gcc-c++ \
       python3-devel \
       openmpi-devel \
       openblas-devel \
       lapack-devel
   
   # Load MPI module
   module load mpi/openmpi-x86_64
   ```

2. **Install Python dependencies:**
   ```bash
   pip install pybind11 mpi4py numpy
   ```

3. **Build and install:**
   ```bash
   cd /path/to/sbd
   pip install -e .
   ```

### macOS

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install system dependencies:**
   ```bash
   brew install gcc openmpi openblas lapack
   ```

3. **Install Python dependencies:**
   ```bash
   pip install pybind11 mpi4py numpy
   ```

4. **Build and install:**
   ```bash
   cd /path/to/sbd
   pip install -e .
   ```

### HPC Systems (with Module System)

Many HPC systems use environment modules. Here's a typical workflow:

```bash
# Load required modules
module load gcc/9.3.0
module load openmpi/4.0.5
module load python/3.9.0
module load openblas/0.3.10

# Install Python dependencies in user space
pip install --user pybind11 mpi4py numpy

# Build and install
cd /path/to/sbd
pip install --user -e .
```

## Configuration Options

### Setting MPI Include Path

If MPI headers are not in the default location:

```bash
export MPI_INCLUDE_PATH=/path/to/mpi/include
pip install -e .
```

### Using Intel MKL

If you want to use Intel MKL instead of OpenBLAS:

```bash
# Load Intel MKL module (if using modules)
module load intel-mkl

# Or set library paths manually
export MKLROOT=/opt/intel/mkl
pip install -e .
```

### Custom Compiler

```bash
export CC=gcc-9
export CXX=g++-9
pip install -e .
```

### Debug Build

For debugging, you can modify `setup.py` to add debug flags:

```python
extra_compile_args=['-std=c++17', '-fopenmp', '-g', '-O0']
```

## Verification

### Test Installation

```bash
python -c "import sbd; print(sbd.__version__)"
```

Expected output: `1.2.0`

### Run Basic Tests

```bash
cd /path/to/sbd/python/tests
python test_basic.py
```

### Run MPI Tests

```bash
cd /path/to/sbd/python/tests
mpirun -np 4 python test_h2o.py
```

## Troubleshooting

### Problem: "ImportError: No module named 'sbd'"

**Solution:**
```bash
# Make sure you're in the sbd directory
cd /path/to/sbd
pip install -e .
```

### Problem: "ImportError: cannot import name '_core'"

**Cause:** The C++ extension module wasn't built successfully.

**Solution:**
```bash
# Check for build errors
python setup.py build_ext --inplace

# Look for error messages in the output
```

### Problem: "fatal error: mpi.h: No such file or directory"

**Cause:** MPI headers not found.

**Solution:**
```bash
# Find MPI include directory
mpicc --showme:compile

# Set MPI_INCLUDE_PATH
export MPI_INCLUDE_PATH=/usr/include/mpi
pip install -e .
```

### Problem: "undefined reference to BLAS/LAPACK functions"

**Cause:** BLAS/LAPACK libraries not found.

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev liblapack-dev

# CentOS/RHEL
sudo yum install openblas-devel lapack-devel

# macOS
brew install openblas lapack
```

### Problem: mpi4py compilation fails

**Cause:** mpi4py needs to be compiled with the same MPI as the bindings.

**Solution:**
```bash
# Install mpi4py from source
pip install mpi4py --no-binary mpi4py
```

### Problem: "version `GLIBCXX_3.4.XX' not found"

**Cause:** C++ standard library version mismatch.

**Solution:**
```bash
# Update libstdc++
sudo apt-get install libstdc++6

# Or use a newer compiler
export CC=gcc-9
export CXX=g++-9
pip install -e . --force-reinstall
```

### Problem: Segmentation fault when running

**Possible causes:**
1. MPI version mismatch between mpi4py and bindings
2. Memory issues
3. Incompatible BLAS/LAPACK

**Solution:**
```bash
# Rebuild everything with the same compiler and MPI
pip uninstall sbd mpi4py
pip install mpi4py --no-binary mpi4py
pip install -e .
```

## Advanced Configuration

### Building with CUDA/HIP Support

If SBD was compiled with THRUST support:

```bash
# For CUDA
export CUDACXX=nvcc
pip install -e .

# For HIP
export HIPCXX=hipcc
pip install -e .
```

### Building with Custom Flags

Edit `setup.py` and modify the `extra_compile_args` and `extra_link_args`:

```python
extra_compile_args=['-std=c++17', '-fopenmp', '-O3', '-march=native']
extra_link_args=['-fopenmp']
```

### Creating a Wheel Package

```bash
pip install wheel
python setup.py bdist_wheel

# Install the wheel
pip install dist/sbd-1.2.0-*.whl
```

## Uninstallation

```bash
pip uninstall sbd
```

## Getting Help

If you encounter issues not covered here:

1. Check the main README: `README.md`
2. Check Python-specific docs: `README_PYTHON.md`
3. Open an issue on GitHub
4. Contact the developers

## Next Steps

After successful installation:

1. Read `README_PYTHON.md` for API documentation
2. Try the examples in `python/examples/`
3. Run the test suite to verify everything works
4. Start using SBD in your projects!