#!/usr/bin/env python3
"""
HPC QuantSim - High-Frequency Market Simulator with Distributed GPU Compute
Setup script for building C++ extensions and configuring the package.
"""

import os
import sys
import subprocess
from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

from setuptools import setup, Extension, find_packages

# Package metadata
__version__ = "0.1.0"
__author__ = "HPC QuantSim Team"
__email__ = "team@hpcquantsim.com"

# Check for CUDA availability
def has_cuda():
    """Check if CUDA is available on the system."""
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

# Check for MPI availability
def has_mpi():
    """Check if MPI is available on the system."""
    try:
        result = subprocess.run(['mpicc', '--version'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

# Compiler flags and includes
def get_cuda_flags():
    """Get CUDA compilation flags and include directories."""
    if not has_cuda():
        return [], [], []
    
    cuda_include_dirs = [
        '/usr/local/cuda/include',
        '/opt/cuda/include',
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/include'
    ]
    
    cuda_library_dirs = [
        '/usr/local/cuda/lib64',
        '/opt/cuda/lib64',
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/lib/x64'
    ]
    
    cuda_libraries = ['cudart', 'curand', 'cublas', 'cusolver']
    
    # Filter existing directories
    cuda_include_dirs = [d for d in cuda_include_dirs if os.path.exists(d)]
    cuda_library_dirs = [d for d in cuda_library_dirs if os.path.exists(d)]
    
    return cuda_include_dirs, cuda_library_dirs, cuda_libraries

def get_mpi_flags():
    """Get MPI compilation flags and include directories."""
    if not has_mpi():
        return [], [], []
    
    try:
        # Get MPI compile flags
        result = subprocess.run(['mpicc', '--showme:compile'], 
                              capture_output=True, text=True)
        compile_flags = result.stdout.strip().split()
        
        # Get MPI link flags
        result = subprocess.run(['mpicc', '--showme:link'], 
                              capture_output=True, text=True)
        link_flags = result.stdout.strip().split()
        
        include_dirs = [flag[2:] for flag in compile_flags if flag.startswith('-I')]
        library_dirs = [flag[2:] for flag in link_flags if flag.startswith('-L')]
        libraries = [flag[2:] for flag in link_flags if flag.startswith('-l')]
        
        return include_dirs, library_dirs, libraries
    except:
        return [], [], []

# Get system-specific flags
cuda_includes, cuda_lib_dirs, cuda_libs = get_cuda_flags()
mpi_includes, mpi_lib_dirs, mpi_libs = get_mpi_flags()

# Define C++ extensions
cpp_extensions = []

# Core simulation engine
core_sources = [
    "hpc_quantsim/core/src/simulation_engine.cpp",
    "hpc_quantsim/core/src/order_book.cpp",
    "hpc_quantsim/core/src/strategy_interface.cpp",
    "hpc_quantsim/core/src/python_bindings.cpp"
]

core_extension = Pybind11Extension(
    "hpc_quantsim.core._core",
    sources=core_sources,
    include_dirs=[
        "hpc_quantsim/core/include",
        pybind11.get_include()
    ] + mpi_includes,
    library_dirs=mpi_lib_dirs,
    libraries=mpi_libs,
    language='c++',
    cxx_std=17,
)

cpp_extensions.append(core_extension)

# GPU kernels (if CUDA available)
if has_cuda():
    gpu_sources = [
        "hpc_quantsim/gpu/src/cuda_kernels.cu",
        "hpc_quantsim/gpu/src/gpu_utils.cpp",
        "hpc_quantsim/gpu/src/gpu_bindings.cpp"
    ]
    
    gpu_extension = Pybind11Extension(
        "hpc_quantsim.gpu._gpu",
        sources=gpu_sources,
        include_dirs=[
            "hpc_quantsim/gpu/include",
            pybind11.get_include()
        ] + cuda_includes,
        library_dirs=cuda_lib_dirs,
        libraries=cuda_libs,
        language='c++',
        cxx_std=17,
    )
    
    cpp_extensions.append(gpu_extension)

# HPC collectives extension
if has_mpi():
    hpc_sources = [
        "hpc_quantsim/hpc/src/mpi_collectives.cpp",
        "hpc_quantsim/hpc/src/ucx_transport.cpp",
        "hpc_quantsim/hpc/src/hpc_bindings.cpp"
    ]
    
    hpc_extension = Pybind11Extension(
        "hpc_quantsim.hpc._hpc",
        sources=hpc_sources,
        include_dirs=[
            "hpc_quantsim/hpc/include",
            pybind11.get_include()
        ] + mpi_includes,
        library_dirs=mpi_lib_dirs,
        libraries=mpi_libs + ['ucx', 'ucs'],
        language='c++',
        cxx_std=17,
    )
    
    cpp_extensions.append(hpc_extension)

# Custom build command
class CustomBuildExt(build_ext):
    """Custom build extension to handle CUDA compilation."""
    
    def build_extensions(self):
        # Set compiler flags
        if has_cuda():
            for ext in self.extensions:
                if '_gpu' in ext.name:
                    ext.extra_compile_args = ['-O3', '-DWITH_CUDA']
                    if sys.platform == 'win32':
                        ext.extra_compile_args.extend(['/std:c++17'])
                    else:
                        ext.extra_compile_args.extend(['-std=c++17', '-fPIC'])
        
        super().build_extensions()

# Read long description from README
long_description = ""
readme_path = Path("README.md")
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()

# Setup configuration
setup(
    name="hpc-quantsim",
    version=__version__,
    author=__author__,
    author_email=__email__,
    description="High-Frequency Market Simulator with Distributed GPU Compute",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hpc-quantsim",
    packages=find_packages(),
    ext_modules=cpp_extensions,
    cmdclass={"build_ext": CustomBuildExt},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "mpi4py>=3.1.4",
        "pyarrow>=13.0.0",
        "fastapi>=0.104.0",
        "pydantic>=2.5.0",
        "click>=8.1.7",
        "pybind11>=2.11.1",
    ],
    extras_require={
        "gpu": ["cupy-cuda12x>=12.2.0", "cudf>=23.10.0"],
        "hpc": ["ucx-py>=0.36.0"],
        "viz": ["plotly>=5.17.0", "dash>=2.14.0", "bokeh>=3.3.0"],
        "dev": ["pytest>=7.4.0", "black>=23.10.0", "mypy>=1.7.0"],
        "all": [
            "cupy-cuda12x>=12.2.0", "cudf>=23.10.0", "ucx-py>=0.36.0",
            "plotly>=5.17.0", "dash>=2.14.0", "bokeh>=3.3.0",
            "pytest>=7.4.0", "black>=23.10.0", "mypy>=1.7.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "hpc-quantsim=hpc_quantsim.cli:main",
            "quantsim-cluster=hpc_quantsim.deployment.cluster:main",
        ],
    },
    package_data={
        "hpc_quantsim": [
            "deployment/*.yml",
            "deployment/*.json",
            "dashboard/templates/*.html",
            "dashboard/static/*",
        ],
    },
    zip_safe=False,
)
