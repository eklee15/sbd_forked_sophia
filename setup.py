"""
Setup script for SBD Python bindings - Unified CPU/GPU backend
Automatically builds:
  - CPU backend (_core_cpu.so): Always built
  - GPU backend (_core_gpu.so): Built if NVIDIA HPC SDK is detected

Override with SBD_BUILD_BACKEND environment variable:
  - SBD_BUILD_BACKEND=cpu: Build only CPU backend
  - SBD_BUILD_BACKEND=gpu: Build only GPU backend
  - SBD_BUILD_BACKEND=both: Build both backends (default if GPU available)
"""
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess
import pybind11

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path"""
    def __str__(self):
        return pybind11.get_include()

def get_mpi4py_include():
    """Get mpi4py include path"""
    try:
        import mpi4py
        return mpi4py.get_include()
    except (ImportError, AttributeError):
        # Fallback: try to find it in site-packages
        import site
        for site_dir in site.getsitepackages():
            mpi4py_inc = os.path.join(site_dir, 'mpi4py', 'include')
            if os.path.exists(mpi4py_inc):
                return mpi4py_inc
        return None

def get_mpi_config():
    """Get MPI configuration from MPI_HOME or mpicc"""
    mpi_home = os.environ.get('MPI_HOME', None)
    
    if mpi_home:
        # Use MPI_HOME if provided (important for CUDA-aware MPI)
        mpi_include = os.path.join(mpi_home, 'include')
        mpi_lib = os.path.join(mpi_home, 'lib')
        print(f"Using MPI from MPI_HOME: {mpi_home}")
        return [mpi_include], [mpi_lib], ['mpi']
    else:
        # Try to detect from mpicc
        try:
            compile_flags = subprocess.check_output(['mpicc', '--showme:compile'],
                                                    universal_newlines=True).strip().split()
            link_flags = subprocess.check_output(['mpicc', '--showme:link'],
                                                universal_newlines=True).strip().split()
            
            include_dirs = [flag[2:] for flag in compile_flags if flag.startswith('-I')]
            library_dirs = [flag[2:] for flag in link_flags if flag.startswith('-L')]
            libraries = [flag[2:] for flag in link_flags if flag.startswith('-l')]
            
            print("Using MPI detected from mpicc")
            return include_dirs, library_dirs, libraries
        except Exception as e:
            print(f"Error: Could not detect MPI. Please set MPI_HOME environment variable.")
            print(f"Error details: {e}")
            sys.exit(1)

def find_nvidia_hpc_sdk():
    """Find NVIDIA HPC SDK compilers (nvc/nvc++) and add to PATH if needed
    Returns: (compiler_path, is_available)
    """
    nvhpc_home = os.environ.get('NVHPC_HOME', None)

    if nvhpc_home:
        # Check if nvc++ exists in NVHPC_HOME
        nvcxx_path = os.path.join(nvhpc_home, 'bin', 'nvc++')
        if os.path.exists(nvcxx_path):
            print(f"Found NVIDIA HPC SDK at: {nvhpc_home}")
            # Add bin directory to PATH so compiler can be found
            nvhpc_bin = os.path.join(nvhpc_home, 'bin')
            current_path = os.environ.get('PATH', '')
            if nvhpc_bin not in current_path:
                os.environ['PATH'] = f"{nvhpc_bin}:{current_path}"
                print(f"Added {nvhpc_bin} to PATH")
            return nvcxx_path, True
        else:
            print(f"Warning: NVHPC_HOME set to {nvhpc_home} but nvc++ not found")

    # Try to find nvc++ in PATH
    import shutil
    nvcxx_path = shutil.which('nvc++')
    if nvcxx_path:
        print(f"Found nvc++ in PATH: {nvcxx_path}")
        return nvcxx_path, True

    # Not found
    return None, False

def has_cuda_gpu():
    """Check if CUDA GPU is available"""
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              timeout=5)
        return result.returncode == 0
    except:
        return False

# Get MPI configuration
mpi_includes, mpi_lib_dirs, mpi_libs = get_mpi_config()

# Get mpi4py include path
mpi4py_inc = get_mpi4py_include()
if not mpi4py_inc:
    print("Warning: Could not find mpi4py include path")

# Build include directories
include_dirs = [get_pybind_include(), 'include'] + mpi_includes
if mpi4py_inc:
    include_dirs.append(mpi4py_inc)

# Build library directories
library_dirs = mpi_lib_dirs.copy()

# Add BLAS library path
blas_lib_path = os.environ.get('BLAS_LIB_PATH', None)
if blas_lib_path:
    library_dirs.append(blas_lib_path)
    print(f"Using BLAS from: {blas_lib_path}")
else:
    print("Warning: BLAS_LIB_PATH not set. Assuming BLAS is in system path.")

# Get BLAS library names (can be openblas, blas+lapack, mkl, etc.)
blas_libs = os.environ.get('BLAS_LIBS', 'openblas').split(',')
print(f"Using BLAS libraries: {blas_libs}")

# Build libraries list
libraries = mpi_libs + blas_libs

# Build extra link args with RPATH
extra_link_args = ['-fopenmp']

# Add RPATH so libraries can be found at runtime without LD_LIBRARY_PATH
for lib_dir in library_dirs:
    extra_link_args.append(f'-Wl,--rpath,{lib_dir}')

print(f"RPATH will be set to: {library_dirs}")

# Detect GPU availability
gpu_compiler, has_nvhpc = find_nvidia_hpc_sdk()
has_gpu = has_cuda_gpu()

# Determine which backends to build
build_backend = os.environ.get('SBD_BUILD_BACKEND', 'auto').lower()

if build_backend == 'auto':
    # Auto-detect: build both if GPU available, otherwise CPU only
    build_cpu = True
    build_gpu = has_nvhpc and has_gpu
    if build_gpu:
        print("\nAuto-detected GPU support - will build both CPU and GPU backends")
    else:
        print("\nNo GPU support detected - will build CPU backend only")
        if not has_nvhpc:
            print("  (NVIDIA HPC SDK not found)")
        if not has_gpu:
            print("  (No CUDA GPU detected)")
elif build_backend == 'cpu':
    build_cpu = True
    build_gpu = False
    print("\nBuilding CPU backend only (SBD_BUILD_BACKEND=cpu)")
elif build_backend == 'gpu':
    build_cpu = False
    build_gpu = True
    print("\nBuilding GPU backend only (SBD_BUILD_BACKEND=gpu)")
    if not has_nvhpc:
        print("Warning: NVIDIA HPC SDK not found, GPU build may fail")
elif build_backend == 'both':
    build_cpu = True
    build_gpu = True
    print("\nBuilding both CPU and GPU backends (SBD_BUILD_BACKEND=both)")
    if not has_nvhpc:
        print("Warning: NVIDIA HPC SDK not found, GPU build may fail")
else:
    print(f"Error: Invalid SBD_BUILD_BACKEND='{build_backend}'")
    print("Valid values: auto, cpu, gpu, both")
    sys.exit(1)

# Build extension modules list
ext_modules = []

# CPU backend
if build_cpu:
    print("\nConfiguring CPU backend (_core_cpu)")
    cpu_ext = Extension(
        'sbd._core_cpu',
        ['python/bindings.cpp'],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language='c++',
        extra_compile_args=[
            '-std=c++17',
            '-fopenmp',
            '-O2',
            '-Wno-sign-compare',
            '-Wno-unused-variable',
            '-fPIC',
            '-DSBD_MODULE_NAME=_core_cpu'
        ],
        extra_link_args=extra_link_args,
    )
    ext_modules.append(cpu_ext)

# GPU backend
if build_gpu:
    print("\nConfiguring GPU backend (_core_gpu)")
    
    if not gpu_compiler:
        print("Error: GPU backend requested but NVIDIA HPC SDK not found")
        sys.exit(1)
    
    print(f"Using compiler: {gpu_compiler}")
    
    gpu_ext = Extension(
        'sbd._core_gpu',
        ['python/bindings.cpp'],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language='c++',
        extra_compile_args=[
            "-DSBD_THRUST",
            "-mp",
            "-cuda",
            "-fast",
            "-Minfo=accel",
            "--diag_suppress=declared_but_not_referenced,set_but_not_used",
            "-fmax-errors=0",
            "-fPIC",
            "-gpu=sm_90",
            "-DSBD_MODULE_NAME=_core_gpu"
        ],
        extra_link_args=extra_link_args + ["-mp", "-cuda", "-cudalib"],
    )
    ext_modules.append(gpu_ext)

# Custom build_ext for GPU backend
class CustomBuildExt(build_ext):
    def build_extensions(self):
        # For each extension, configure compiler appropriately
        for ext in self.extensions:
            if '_core_gpu' in ext.name:
                # GPU backend: use NVIDIA HPC SDK
                self.compiler.set_executable("compiler_so", gpu_compiler)
                self.compiler.set_executable("compiler_cxx", gpu_compiler)
                self.compiler.set_executable("linker_so", f"{gpu_compiler} -shared")
                
                # Remove unwanted GCC flags
                unwanted = [
                    "-grecord-gcc-switches",
                    "-Werror=format-security",
                    "-Wsign-compare",
                    "-fstack-protector-strong",
                ]
                
                for attr in ["compiler_so", "compiler", "compiler_cxx", "linker_so"]:
                    flags = getattr(self.compiler, attr, None)
                    if flags:
                        cleaned = [f for f in flags if f not in unwanted]
                        setattr(self.compiler, attr, cleaned)
        
        build_ext.build_extensions(self)

# Setup
setup(
    name='sbd',
    version='1.2.0',
    author='Tomonori Shirakawa',
    author_email='',
    description='Python bindings for Selected Basis Diagonalization (SBD) library',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/hfwen0502/sbd',
    packages=['sbd'],
    package_dir={'sbd': 'python'},
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt} if build_gpu else {},
    install_requires=[
        'pybind11>=2.6.0',
        'mpi4py>=3.0.0',
        'numpy>=1.19.0',
    ],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
    ],
    zip_safe=False,
)

print("\nSetup complete!")
if build_cpu:
    print("  - CPU backend: sbd._core_cpu")
if build_gpu:
    print("  - GPU backend: sbd._core_gpu")
print()

# Made with Bob
