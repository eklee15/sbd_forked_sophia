"""
Setup script for SBD Python bindings
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import pybind11

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path"""
    def __str__(self):
        return pybind11.get_include()

# Get MPI include path from environment or use default
mpi_include = os.environ.get('MPI_INCLUDE_PATH', '/usr/include/mpi')

ext_modules = [
    Extension(
        'sbd._core',
        ['python/bindings.cpp'],
        include_dirs=[
            get_pybind_include(),
            'include',
            mpi_include,
        ],
        libraries=['mpi', 'lapack', 'blas'],
        library_dirs=[],
        language='c++',
        extra_compile_args=['-std=c++17', '-fopenmp', '-O3'],
        extra_link_args=['-fopenmp'],
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
