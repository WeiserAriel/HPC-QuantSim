# HPC QuantSim Docker Container
# Multi-stage build for production-ready quantitative simulation platform

ARG PYTHON_VERSION=3.11
ARG UBUNTU_VERSION=22.04

# Build stage
FROM nvidia/cuda:11.8-devel-ubuntu${UBUNTU_VERSION} as builder

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    libhdf5-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create Python alias
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Production stage
FROM nvidia/cuda:11.8-runtime-ubuntu${UBUNTU_VERSION} as production

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=all
ENV PATH="/usr/local/cuda/bin:${PATH}"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python3-pip \
    libhdf5-103 \
    libssl3 \
    libffi8 \
    openssh-client \
    libopenmpi3 \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# Create Python alias
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/dist-packages /usr/local/lib/python${PYTHON_VERSION}/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create application user and directory
RUN useradd --create-home --shell /bin/bash --uid 1001 quantsim
WORKDIR /app
RUN chown quantsim:quantsim /app

# Copy application code
COPY --chown=quantsim:quantsim . /app/

# Install the application
USER quantsim
RUN python -m pip install --user --no-deps -e .

# Create data and config directories
RUN mkdir -p /app/data /app/config /app/results /app/logs

# Expose dashboard port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import hpc_quantsim; print('OK')" || exit 1

# Default command runs the dashboard
CMD ["hpc-quantsim", "dashboard", "--host", "0.0.0.0", "--port", "8000"]

# Multi-stage build for MPI cluster deployment
FROM production as mpi-cluster

USER root

# Install additional MPI and networking tools
RUN apt-get update && apt-get install -y \
    libopenmpi-dev \
    openmpi-common \
    libnccl2 \
    libnccl-dev \
    infiniband-diags \
    ibverbs-utils \
    rdma-core \
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Configure OpenMPI for containerized environments
RUN echo "btl_vader_single_copy_mechanism = none" >> /usr/etc/openmpi/openmpi-mca-params.conf && \
    echo "btl_base_warn_component_unused = 0" >> /usr/etc/openmpi/openmpi-mca-params.conf

USER quantsim

# Set MPI environment variables
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none
ENV OMPI_MCA_btl_base_warn_component_unused=0

# Default command for MPI cluster
CMD ["mpirun", "--allow-run-as-root", "-np", "1", "python", "-m", "hpc_quantsim.cli", "run"]

# GPU-optimized stage
FROM production as gpu-optimized

USER root

# Install additional GPU libraries
RUN apt-get update && apt-get install -y \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

USER quantsim

# GPU-specific environment variables
ENV CUDA_CACHE_PATH=/tmp/cuda-cache
ENV NUMBA_CACHE_DIR=/tmp/numba-cache

# Default command for GPU workloads
CMD ["hpc-quantsim", "run", "--gpu", "--scenarios", "1000"]
