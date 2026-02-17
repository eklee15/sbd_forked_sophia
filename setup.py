"""
Setup script for SBD Python bindings
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

def get_mpi_flags():
    """Get MPI compiler and linker flags"""
    try:
        # Try to get MPI compile flags
        compile_flags = subprocess.check_output(['mpicc', '--showme:compile'],
                                                universal_newlines=True).strip().split()
        link_flags = subprocess.check_output(['mpicc', '--showme:link'],
                                            universal_newlines=True).strip().split()
        
        include_dirs = [flag[2:] for flag in compile_flags if flag.startswith('-I')]
        library_dirs = [flag[2:] for flag in link_flags if flag.startswith('-L')]
        libraries = [flag[2:] for flag in link_flags if flag.startswith('-l')]
        extra_link = [flag for flag in link_flags if not flag.startswith('-l') and not flag.startswith('-L')]
        
        return include_dirs, library_dirs, libraries, extra_link
    except:
        # Fallback to environment variable or default
        mpi_include = os.environ.get('MPI_INCLUDE_PATH', '/usr/include/mpi')
        return [mpi_include], [], ['mpi'], []

# Get MPI configuration
mpi_includes, mpi_lib_dirs, mpi_libs, mpi_link_flags = get_mpi_flags()

# Get MPI include path from environment or use detected
if 'MPI_INCLUDE_PATH' in os.environ:
    mpi_includes = [os.environ['MPI_INCLUDE_PATH']]

# Build include directories
include_dirs = [get_pybind_include(), 'include'] + mpi_includes

# Build library directories
library_dirs = mpi_lib_dirs

# Build libraries list - use MPI libs + standard math libs
libraries = mpi_libs + ['lapack', 'blas']

# Build extra link args
extra_link_args = ['-fopenmp'] + mpi_link_flags

ext_modules = [
    Extension(
        'sbd._core',
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
            '-fPIC'
        ],
        extra_link_args=extra_link_args,
    ),
]

setup(
    name='sbd',
    version='1.2.0',
    author='Tomonori Shirakawa',
    author_email='',
    description='Python bindings for Selected Basis Diagonalization (SBD) library',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/your-repo/sbd',
    packages=['sbd'],
    package_dir={'sbd': 'python'},
    ext_modules=ext_modules,
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

# Made with Bob
