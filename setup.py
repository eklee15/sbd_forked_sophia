from setuptools import setup, Extension
import sys
import os
import subprocess
import pybind11


class get_pybind_include(object):
    def __str__(self):
        return pybind11.get_include()


def get_mpi4py_include():
    try:
        import mpi4py
        return mpi4py.get_include()
    except (ImportError, AttributeError):
        import site
        for site_dir in site.getsitepackages():
            mpi4py_inc = os.path.join(site_dir, 'mpi4py', 'include')
            if os.path.exists(mpi4py_inc):
                return mpi4py_inc
        return None


def get_mpi_config():
    mpi_home = os.environ.get('MPI_HOME', None)
    if mpi_home:
        mpi_include = os.path.join(mpi_home, 'include')
        mpi_lib = os.path.join(mpi_home, 'lib')
        print(f"Using MPI from MPI_HOME: {mpi_home}")
        return [mpi_include], [mpi_lib], ['mpi']
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
    nvhpc_home = os.environ.get('NVHPC_HOME', None)
    if nvhpc_home:
        nvcxx_path = os.path.join(nvhpc_home, 'bin', 'nvc++')
        if os.path.exists(nvcxx_path):
            print(f"Found NVIDIA HPC SDK at: {nvhpc_home}")
            nvhpc_bin = os.path.join(nvhpc_home, 'bin')
            current_path = os.environ.get('PATH', '')
            if nvhpc_bin not in current_path:
                os.environ['PATH'] = f"{nvhpc_bin}:{current_path}"
            return nvcxx_path, True
        else:
            print(f"Warning: NVHPC_HOME set to {nvhpc_home} but nvc++ not found")
    import shutil
    nvcxx_path = shutil.which('nvc++')
    if nvcxx_path:
        print(f"Found nvc++ in PATH: {nvcxx_path}")
        return nvcxx_path, True
    return None, False


# Get MPI configuration
mpi_includes, mpi_lib_dirs, mpi_libs = get_mpi_config()

# Get mpi4py include path
mpi4py_inc = get_mpi4py_include()
if not mpi4py_inc:
    print("Warning: Could not find mpi4py include path")

# Build include/library directories
include_dirs = [get_pybind_include(), 'include'] + mpi_includes
if mpi4py_inc:
    include_dirs.append(mpi4py_inc)

library_dirs = mpi_lib_dirs.copy()

blas_lib_path = os.environ.get('BLAS_LIB_PATH', None)
if blas_lib_path:
    library_dirs.append(blas_lib_path)
    print(f"Using BLAS from: {blas_lib_path}")
else:
    print("Warning: BLAS_LIB_PATH not set. Assuming BLAS is in system path.")

blas_libs = os.environ.get('BLAS_LIBS', 'openblas').split(',')
print(f"Using BLAS libraries: {blas_libs}")

libraries = mpi_libs + blas_libs

# RPATH so libraries are found at runtime without LD_LIBRARY_PATH
extra_link_args = ['-fopenmp']
for lib_dir in library_dirs:
    extra_link_args.append(f'-Wl,--rpath,{lib_dir}')
print(f"RPATH will be set to: {library_dirs}")

# Detect GPU compiler
gpu_compiler, has_nvhpc = find_nvidia_hpc_sdk()

# Determine which backends to build
build_backend = os.environ.get('SBD_BUILD_BACKEND', 'auto').lower()

if build_backend == 'auto':
    build_cpu = True
    build_gpu = has_nvhpc
    if build_gpu:
        print("\nAuto-detected nvc++ - will build both CPU and GPU backends")
    else:
        print("\nnvc++ not found - will build CPU backend only")
elif build_backend == 'cpu':
    build_cpu = True
    build_gpu = False
    print("\nBuilding CPU backend only (SBD_BUILD_BACKEND=cpu)")
elif build_backend == 'gpu':
    build_cpu = False
    build_gpu = True
    print("\nBuilding GPU backend only (SBD_BUILD_BACKEND=gpu)")
    if not has_nvhpc:
        print("Warning: nvc++ not found, GPU build may fail")
elif build_backend == 'both':
    build_cpu = True
    build_gpu = True
    print("\nBuilding both CPU and GPU backends (SBD_BUILD_BACKEND=both)")
    if not has_nvhpc:
        print("Warning: nvc++ not found, GPU build may fail")
else:
    print(f"Error: Invalid SBD_BUILD_BACKEND='{build_backend}'")
    print("Valid values: auto, cpu, gpu, both")
    sys.exit(1)

ext_modules = []

if build_cpu:
    print("\nConfiguring CPU backend (_core_cpu)")
    import platform
    if platform.system() == 'Darwin':
        omp_inc = '/opt/homebrew/opt/libomp/include'
        omp_lib = '/opt/homebrew/opt/libomp/lib'
        openblas_lib = '/opt/homebrew/opt/openblas/lib'
        cpu_compile_args = [
            '-DSBD_TRADMODE',
            '-std=c++17', '-Xpreprocessor', '-fopenmp', '-O3',
            '-Wno-sign-compare', '-Wno-unused-variable', '-fPIC',
            '-DSBD_MODULE_NAME=_core_cpu', f'-I{omp_inc}',
        ]
        cpu_link_args = [f'-L{omp_lib}', f'-L{openblas_lib}', '-lomp']
        cpu_inc = include_dirs + [omp_inc]
        cpu_lib_dirs = library_dirs + [omp_lib, openblas_lib]
        cpu_libs = libraries + ['omp']
    else:
        cpu_compile_args = [
            '-DSBD_TRADMODE',
            '-std=c++17', '-fopenmp', '-O3',
            '-Wno-sign-compare', '-Wno-unused-variable', '-fPIC',
            '-DSBD_MODULE_NAME=_core_cpu',
        ]
        cpu_link_args = extra_link_args
        cpu_inc = include_dirs
        cpu_lib_dirs = library_dirs
        cpu_libs = libraries

    cpu_ext = Extension(
        'sbd._core_cpu',
        ['python/bindings.cpp'],
        include_dirs=cpu_inc,
        libraries=cpu_libs,
        library_dirs=cpu_lib_dirs,
        language='c++',
        extra_compile_args=cpu_compile_args,
        extra_link_args=cpu_link_args,
    )
    ext_modules.append(cpu_ext)

if build_gpu:
    print("\nConfiguring GPU backend (_core_gpu)")
    if not gpu_compiler:
        print("Error: GPU backend requested but nvc++ not found")
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
            '-DSBD_THRUST',
            '-DSBD_TRADMODE',
            '-mp',
            '-cuda',
            '-fast',
            '-Minfo=accel',
            '--diag_suppress=declared_but_not_referenced,set_but_not_used',
            '-fmax-errors=0',
            '-fPIC',
            '-gpu=sm_90',
            '-DSBD_MODULE_NAME=_core_gpu',
        ],
        extra_link_args=extra_link_args + ['-mp', '-cuda', '-cudalib'],
    )
    ext_modules.append(gpu_ext)


setup(
    name='sbd',
    version='1.3.0',
    author='Tomonori Shirakawa',
    author_email='',
    description='Python bindings for Selected Basis Diagonalization (SBD) library',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/hfwen0502/sbd',
    packages=['sbd'],
    package_dir={'sbd': 'python'},
    ext_modules=ext_modules,
    install_requires=[
        'pybind11>=2.6.0',
        'mpi4py>=3.0.0',
        'numpy>=1.19.0',
    ],
    python_requires='>=3.7',
    zip_safe=False,
)

print("\nSetup complete!")
if build_cpu:
    print("  - CPU backend: sbd._core_cpu")
if build_gpu:
    print("  - GPU backend: sbd._core_gpu")
print()
