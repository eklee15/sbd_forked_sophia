#!/bin/bash
export MPI_HOME=/home/leeeun/miniconda3/envs/sbd/bin
export BLAS_LIB_PATH=/home/leeeun/miniconda3/envs/sbd/libblas.so
export BLAS_LIBS=/home/leeeun/miniconda3/envs/sbd/lib  # or mkl_rt
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/leeeun/miniconda3/envs/sbd/lib  

# macOS: use system clang to match Python's libc++
export CC=/usr/bin/cc
export CXX=/usr/bin/c++

# GPU (optional)
export NVHPC_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/2026/compilers
export CC=/opt/nvidia/hpc_sdk/Linux_x86_64/2026/compilers/bin/nvcc
export CXX=/opt/nvidia/hpc_sdk/Linux_x86_64/2026/compilers/bin/nvc++
# Clear these — inherited flags (e.g. from conda) may be gcc-specific
# and break nvc++ compilation
export CFLAGS=''
export CXXFLAGS=''



# 1. Define the base path for your Conda environment
export CONDA_PREFIX=/home/leeeun/miniconda3/envs/sbd

# 2. Add the include path (for mpi.h) and library path (for linking)
export CFLAGS="-I$CONDA_PREFIX/include"
export CPPFLAGS="-I$CONDA_PREFIX/include"
export LDFLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"

# 3. Ensure nvc++ also gets these via the build system's environment
export CXXFLAGS="-I$CONDA_PREFIX/include"

# 4. Retry the install
#pip install -e . --no-build-isolation
